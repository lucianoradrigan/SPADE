"""B4 OOD map (2026-11-16): where the dc_motor classifiers answer, abstain or are confidently wrong, over speed and load.

**Exploratory and burned.** These data characterize the shipped and candidate classifiers; they must never inform a model
decision (training, thresholds, promotion): instructions 2026-11-16, B4. Simulated (dc_perm_ex_v2, dcm_exact, 10 kHz).

Plan (fixed before generating): speeds 20-400 rad/s in steps of 20 (20 speeds) x 2 load levels (no extra load, +5 N·m)
x 5 classes (normal, outer_race, inner_race, ball, cage; severity x1.0; noise-free; nominal psi_e and j_rotor) = 200 runs
of 2.0 s, one seed each from the block SEED_BLOCK (300001-300200), disjoint from every earlier block
(tests/test_ood_map_seeds.py). Budget: 2 CPU-h.

    python experiments/ood_map_2026_11_16.py --workers 8     # data/ood_map_2026-11-16.parquet + docs/audit/ai/ood_map_2026-11-16.{md,json}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEEDS_RAD_S = tuple(float(s) for s in range(20, 401, 20))
LOADS_NM = (0.0, 5.0)
FAULTS = (None, "outer_race", "inner_race", "ball", "cage")
DURATION_S = 2.0
SEED_BLOCK = range(300001, 300001 + len(SPEEDS_RAD_S) * len(LOADS_NM) * len(FAULTS))
PLANT = "dc_perm_ex_v2"
DATA = REPO / "data" / "ood_map_2026-11-16.parquet"
OUT = REPO / "docs" / "audit" / "ai" / "ood_map_2026-11-16"
MODELS = {  # name -> (run, artifact kind for report_ai_n27_regen._predict)
    "PC shipped": ("configs/classifiers/pc_server_5class_w2048/2026-11-04_run01", "keras"),
    "PC B4 candidate": ("configs/classifiers/pc_server_5class_w2048/2026-11-10_run01", "keras"),
    "RPi5 shipped": ("configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01", "tflite"),
    "RPi5 B4 candidate": ("configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-10_run01", "tflite"),
    "ESP32 shipped": ("configs/classifiers/esp32_dscnn/2026-10-18_run01", "tflite"),
}
SETTLE_S = 0.5


def jobs() -> list:
    out, seeds = [], iter(SEED_BLOCK)
    for w in SPEEDS_RAD_S:
        for load in LOADS_NM:
            for fault in FAULTS:
                out.append({"speed": w, "load_nm": load, "fault": fault, "seed": next(seeds)})
    return out


def _simulate(job: dict):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    f = job["fault"]
    kw = {} if f is None else {"electrical_severity": 8.0, "mechanical_severity": CALIBRATED_MECHANICAL_SEVERITY[f]}
    if job["load_nm"]:
        kw["load_torque_nm"] = job["load_nm"]
    sid = f"oodmap_{f or 'normal'}_w{job['speed']:.0f}_l{job['load_nm']:g}_seed{job['seed']}"
    df = records_to_dataframe(run_scenario(Scenario(scenario_id=sid, plant_config_id=PLANT, fault_type=f, seed=job["seed"], duration_s=DURATION_S,
                                                    omega_ref_rad_s=job["speed"], electrical_noise_pct=0.0, mechanical_noise_pct=0.0, **kw)))
    df["map_speed"], df["map_load_nm"] = job["speed"], job["load_nm"]
    return df


def generate(workers: int):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        frames = list(ex.map(_simulate, jobs(), chunksize=2))
    df = pd.concat(frames, ignore_index=True)
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA)
    return df, {"wall_s": round(time.time() - t0, 1), "children_cpu_h": round((ch.ru_utime + ch.ru_stime) / 3600, 3), "workers": workers}


def evaluate(df) -> dict:
    import numpy as np

    sys.path.insert(0, str(REPO / "experiments"))
    import report_ai_n27_regen as RG
    from driveflow.ai import envelope

    df = df[df["timestamp_s"] >= SETTLE_S]
    res = {}
    for name, (run, kind) in MODELS.items():
        info = RG._model_info(run)
        classes = info["metrics"]["classes"]
        thr = info["metrics"]["calibration"]["abstain_below"]
        env = envelope.trained_envelope(REPO / run)
        cells = []
        for (w, load), part in df.groupby(["map_speed", "map_load_nm"]):
            part = part[part["label"].isin(classes)]
            X, ys, g = RG._windows(part, classes, info, 100_000)
            if len(ys) == 0:
                continue
            P = RG._predict(run, kind, X)
            y = np.array([classes.index(c) for c in ys])
            pred, conf = P.argmax(axis=1), P.max(axis=1)
            answered = conf >= thr
            healthy = y == classes.index("normal")
            cells.append({"speed": float(w), "load_nm": float(load), "n_windows": int(len(y)),
                          "inside_envelope": bool(env and env["speed_rad_s"][0] - 1e-6 <= w <= env["speed_rad_s"][1] + 1e-6),
                          "abstention": float(1 - answered.mean()),
                          "accuracy_on_answered": float((pred[answered] == y[answered]).mean()) if answered.any() else None,
                          "confident_wrong": float((answered & (pred != y)).mean()),
                          "healthy_confident_wrong": float((answered & healthy & (pred != y)).sum() / max(1, healthy.sum())),
                          "per_class_recall": {c: (float((pred[(y == k) & answered] == k).sum() / max(1, (y == k).sum()))) for k, c in enumerate(classes)}})
        res[name] = {"run": run, "artifact": kind, "classes": classes, "abstain_below": thr, "envelope": env["speed_rad_s"] if env else None, "cells": cells}
        print(name, len(cells), "cells", flush=True)
    return res


def _md(r: dict) -> str:
    L = ["# OOD map of the dc_motor classifiers (B4, 2026-11-16)", "",
         "**Exploratory and burned: these data must never inform a model decision.** Simulated (`dc_perm_ex_v2`, `dcm_exact`, 10 kHz); "
         f"{len(SPEEDS_RAD_S)} speeds × {len(LOADS_NM)} loads × {len(FAULTS)} classes, one 2.0 s noise-free run each, seeds {SEED_BLOCK.start}–{SEED_BLOCK.stop - 1}. "
         f"Command: `{r['command']}`. Cost: {r['cost']}. Raw: `ood_map_2026-11-16.json`.", "",
         "Each cell: share of steady-state windows (after 0.5 s) answered **confidently wrong** (above the model's abstention threshold and wrong), "
         "and in brackets the abstention. `·` marks speeds inside the model's trained envelope. Healthy-only numbers are in the JSON.", ""]
    for name, m in r["models"].items():
        L += [f"## {name} (`{m['run']}`, envelope {m['envelope']} rad/s)", "", "| speed (rad/s) | " + " | ".join(f"load +{l:g} N·m" for l in LOADS_NM) + " |", "|---|" + "---|" * len(LOADS_NM)]
        by = {(c["speed"], c["load_nm"]): c for c in m["cells"]}
        for w in SPEEDS_RAD_S:
            row = []
            for l in LOADS_NM:
                c = by.get((w, l))
                row.append("n/a" if c is None else f"{c['confident_wrong']:.2f} ({c['abstention']:.2f})")
            mark = "·" if any(by.get((w, l), {}).get("inside_envelope") for l in LOADS_NM) else ""
            L.append(f"| {w:.0f}{mark} | " + " | ".join(row) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--evaluate-only", action="store_true", help="reuse data/ood_map_2026-11-16.parquet")
    args = ap.parse_args(argv)
    import pandas as pd

    if args.evaluate_only:
        df, cost = pd.read_parquet(DATA), json.loads(OUT.with_suffix(".json").read_text())["cost"] if OUT.with_suffix(".json").exists() else {}
    else:
        df, cost = generate(args.workers)
    if cost.get("children_cpu_h", 0) > 2.0:
        print("CPU budget exceeded: stop (nothing deleted)")
    r = {"date": "2026-11-16", "task": "B4", "simulated": True, "exploratory_burned": True, "command": "python experiments/ood_map_2026_11_16.py",
         "plan": {"speeds_rad_s": SPEEDS_RAD_S, "loads_nm": LOADS_NM, "faults": [f or "normal" for f in FAULTS], "duration_s": DURATION_S,
                  "seed_block": [SEED_BLOCK.start, SEED_BLOCK.stop - 1], "plant": PLANT, "settle_s": SETTLE_S},
         "cost": cost, "host": platform.platform(), "models": evaluate(df)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    OUT.with_suffix(".md").write_text(_md(r), encoding="utf-8")
    print("written", OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
