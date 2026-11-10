"""B4 evaluation (docs/audit/b4/b4_preregistration.md, approved 2026-11-09 before any data; plan configs/b4_plan.yaml).

    python experiments/report_b4.py --pc <run> --rpi5 <run> --esp32 <variant dir> --esp32-keras <run> --teacher3 <run>

Per tier: the candidate's deployed artifact against the currently promoted run, on the same windows:
  E1  the candidate's own recorded test split of its B4 grid (G5-v2 for PC/RPi5, G3-v2 + off-grid healthy for ESP32);
  E3  report_classifier_evaluation.CONDITIONS regenerated on dc_perm_ex_v2 (data/b4_eval_cache_5class.parquet), pooled;
  E4  held-out low speeds 50 and 70 rad/s, steady-state windows (t >= 0.5 s), never trained on.
Criteria (fixed before any data): the AI-N27 rule (gate D1 on E1; delta macro-F1 >= -1 pt, delta recall >= -5 pt per class,
|delta abstention| <= 5 pt on E1, E3 and E4); low speed for PC and RPi5 (E4: macro-F1 >= 0.80, recall >= 0.50 per class);
ESP32 int8 against its own Keras model on E1, E3, E4 (recall drop <= 5 pt per class, macro-F1 drop <= 3 pt).
Nothing is promoted (decision N-D4). Report: docs/audit/b4/b4_results.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import report_ai_n27_regen as RG  # noqa: E402

PLAN = yaml.safe_load((REPO / "configs" / "b4_plan.yaml").read_text())
OUT = REPO / "docs" / "audit" / "b4" / "b4_results"
DATA = {"G3": "data/b4_train_3class.parquet", "G5": "data/b4_grid_5class.parquet", "E3": "data/b4_eval_cache_5class.parquet",
        "E4_3class": "data/b4_e4_3class.parquet", "E4_5class": "data/b4_e4_5class.parquet"}
PROMOTED = {"PC": "configs/classifiers/pc_server_5class_w2048/2026-11-04_run01",
            "RPi5": "configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01",
            "ESP32": "configs/classifiers/esp32_dscnn/2026-10-18_run01"}
KIND = {"PC": "keras", "RPi5": "tflite", "ESP32": "tflite"}
SETS = ("E1", "E3", "E4")
LOW = PLAN["criteria"]["low_speed_e4"]
INT8 = PLAN["criteria"]["int8_vs_keras"]


def _rel(p: str) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _e3(ev: pd.DataFrame, classes, info):
    from report_classifier_evaluation import _SHARES_NOMINAL, CONDITIONS

    parts = []
    for cond, (_o, _s, win) in CONDITIONS.items():
        df = ev[ev["condition"] == _SHARES_NOMINAL.get(cond, cond)]
        df = df[df["timestamp_s"] >= RG.SETTLE_S] if win == "steady" else df[df["timestamp_s"] < RG.SETTLE_S]
        X, y, g = RG._windows(df, classes, info, 10_000)
        parts.append((X, y, np.array([f"{cond}/{x}" for x in g])))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def _sets(tier: str, cand: dict, prom: dict, frames: dict) -> dict:
    classes = prom["metrics"]["classes"]
    grid = frames["G3" if tier == "ESP32" else "G5"]
    X, y, g = RG._windows(grid, sorted(grid["label"].unique()), cand, cand["cap"])
    test = set(cand["metrics"]["split"]["group_hashes"]["test"])
    m = np.array([RG._hash(x) in test for x in g])
    e1 = (X[m], y[m], g[m])
    e3 = _e3(frames["E3"], classes, prom)
    e4df = frames["E4_3class" if tier == "ESP32" else "E4_5class"]
    e4 = RG._windows(e4df[e4df["timestamp_s"] >= RG.SETTLE_S], classes, prom, 100_000)
    return {"E1": e1, "E3": e3, "E4": e4}


def _checks(t: dict, classes: list, tier: str) -> dict:
    c = {}
    rule = RG.RULE
    e1c = t["sets"]["E1"]["candidate"]
    c["gate_D1_E1_macro_f1"] = {"value": e1c["macro_f1"], "threshold": rule["gate_min_macro_f1"], "pass": e1c["macro_f1"] >= rule["gate_min_macro_f1"]}
    worst = min(v for v in e1c["recall"].values() if v is not None)
    c["gate_D1_E1_min_recall"] = {"value": worst, "threshold": rule["gate_min_recall"], "pass": worst >= rule["gate_min_recall"]}
    for s in SETS:
        d = t["sets"][s]["delta_points"]
        c[f"{s}_delta_macro_f1"] = {"value": d["macro_f1"], "threshold": rule["min_delta_macro_f1_points"], "pass": d["macro_f1"] >= rule["min_delta_macro_f1_points"]}
        rec = {k: v for k, v in d["recall"].items() if v is not None}
        w = min(rec, key=rec.get)
        c[f"{s}_worst_delta_recall"] = {"class": w, "value": rec[w], "threshold": rule["min_delta_recall_points"], "pass": rec[w] >= rule["min_delta_recall_points"]}
        c[f"{s}_delta_abstention"] = {"value": d["abstention"], "threshold": rule["max_abs_delta_abstention_points"], "pass": abs(d["abstention"]) <= rule["max_abs_delta_abstention_points"]}
    if tier in ("PC", "RPi5"):
        e4c = t["sets"]["E4"]["candidate"]
        c["E4_low_speed_macro_f1"] = {"value": e4c["macro_f1"], "threshold": LOW["macro_f1_min"], "pass": e4c["macro_f1"] >= LOW["macro_f1_min"]}
        w4 = min(v for v in e4c["recall"].values() if v is not None)
        c["E4_low_speed_min_recall"] = {"value": w4, "threshold": LOW["recall_min_per_class"], "pass": w4 >= LOW["recall_min_per_class"]}
    if tier == "ESP32":
        for s in SETS:
            k, q = t["int8_vs_keras"][s]["keras"], t["int8_vs_keras"][s]["int8"]
            drop_f1 = 100.0 * (k["macro_f1"] - q["macro_f1"])
            drops = {cl: 100.0 * (k["recall"][cl] - q["recall"][cl]) for cl in classes if k["recall"][cl] is not None}
            wc = max(drops, key=drops.get)
            c[f"{s}_int8_macro_f1_drop"] = {"value": drop_f1, "threshold": INT8["max_macro_f1_drop_pts"], "pass": drop_f1 <= INT8["max_macro_f1_drop_pts"]}
            c[f"{s}_int8_worst_recall_drop"] = {"class": wc, "value": drops[wc], "threshold": INT8["max_recall_drop_pts_per_class"], "pass": drops[wc] <= INT8["max_recall_drop_pts_per_class"]}
    return c


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pc", required=True)
    ap.add_argument("--rpi5", required=True)
    ap.add_argument("--esp32", required=True, help="the selected ESP32 variant directory (int8 model.tflite + its Keras files)")
    ap.add_argument("--esp32-keras", required=True)
    ap.add_argument("--teacher3", required=True)
    ap.add_argument("--extra", default="{}", help="JSON merged into the report (selection, timings, deviations)")
    args = ap.parse_args(argv)
    t0 = time.time()
    frames = {k: pd.read_parquet(REPO / p) for k, p in DATA.items()}
    cands = {"PC": _rel(args.pc), "RPi5": _rel(args.rpi5), "ESP32": _rel(args.esp32)}
    res = {"command": "python experiments/report_b4.py " + " ".join(argv if argv is not None else sys.argv[1:]), "preregistration": "docs/audit/b4/b4_preregistration.md",
           "plan": "configs/b4_plan.yaml", "host": platform.platform(), "python": platform.python_version(), "rule_ai_n27": RG.RULE, "low_speed": LOW, "int8_vs_keras": INT8,
           "data": DATA, "promoted": PROMOTED, "candidates": cands, "tiers": {}, **json.loads(args.extra)}
    for tier in ("PC", "RPi5", "ESP32"):
        prom, cand = RG._model_info(PROMOTED[tier]), RG._model_info(cands[tier])
        classes = prom["metrics"]["classes"]
        for k in ("classes", "channels", "level_scaling"):
            assert prom["metrics"].get(k) == cand["metrics"].get(k), f"{tier}: {k} differs between promoted and candidate"
        assert prom["window"] == cand["window"]
        sets = _sets(tier, cand, prom, frames)
        t = {"promoted": PROMOTED[tier], "candidate": cands[tier], "artifact": KIND[tier], "classes": classes, "sets": {}, "candidate_dataset": cand["dataset"]}
        gh = cand["metrics"]["split"]["group_hashes"]  # Q2 (§2): no acquisition in two splits
        t["Q2_split_by_acquisition"] = {"overlaps": {f"{a}~{b}": len(set(gh[a]) & set(gh[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}}
        t["Q2_split_by_acquisition"]["pass"] = not any(t["Q2_split_by_acquisition"]["overlaps"].values())
        for s, (X, y, g) in sets.items():
            t["sets"][s] = RG._compare(prom, cand, KIND[tier], X, y, g, classes)
        assert t["sets"]["E1"]["candidate"]["n_windows"] == cand["metrics"]["n_test"], f"{tier}: E1 differs from the candidate's recorded test set"
        if tier == "ESP32":
            keras_info = RG._model_info(_rel(args.esp32_keras))
            t["int8_vs_keras"] = {}
            for s, (X, y, g) in sets.items():
                keep = np.isin(y, classes)
                yy = np.array([classes.index(c) for c in y[keep]])
                thr = cand["metrics"]["calibration"]["abstain_below"]
                t["int8_vs_keras"][s] = {"int8": RG._score(RG._predict(cand["run"], "tflite", X[keep]), yy, classes, thr),
                                         "keras": RG._score(RG._predict(keras_info["run"], "keras", X[keep]), yy, classes, thr)}
        if KIND[tier] != "keras":
            t["keras_descriptive"] = {s: RG._compare(prom, cand, "keras", *v, classes) for s, v in sets.items()}
        t["checks"] = _checks(t, classes, tier)
        t["verdict"] = "MEETS THE CRITERION" if all(c["pass"] for c in t["checks"].values()) else "DOES NOT MEET THE CRITERION"
        res["tiers"][tier] = t
        print(tier, t["verdict"], flush=True)
    tm = RG._model_info(_rel(args.teacher3))["metrics"]
    res["teacher_3class"] = {"run": _rel(args.teacher3), "test_macro_f1": tm["test_macro_f1"], "test_accuracy": tm["test_accuracy"], "n_test": tm["n_test"]}
    res["evaluation_wall_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in res["tiers"].items()}))


if __name__ == "__main__":
    main()
