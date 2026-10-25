"""R3 (round PLECS/OpenModelica, after R2; user decision 2026-10-25): the classifier training datasets
regenerated with the DC plant integrated by the converged RK45 reference, compared with the Euler
originals the promoted models were trained on.

Per dataset pair: SHA-256, rows, runs and runs per label; per run (paired by source_file), NRMS of
current_r, rpm, torque_nm and acc_x of RK45 vs Euler (RMS difference / range of the RK45 run),
start-up (t < 0.5 s) and steady state separately; summary by label. Nothing is trained.

Usage: python experiments/report_r3_datasets_rk45.py [--out docs/audit/plecs_om]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
PAIRS = {"5-class (PC, RPi5)": ("data/diagnosis_grid_dataset_5class.parquet", "data/diagnosis_grid_dataset_5class_rk45.parquet"),
         "3-class (ESP32)": ("data/diagnosis_grid_dataset.parquet", "data/diagnosis_grid_dataset_rk45.parquet")}
CHANNELS = ("current_r", "rpm", "torque_nm", "acc_x")
SETTLE_S = 0.5


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _nrms(a, b):
    r = float(np.ptp(b))
    return float(np.sqrt(np.mean((a - b) ** 2)) / r) if r > 0 else float("nan")


def compare(euler_path: Path, rk_path: Path) -> dict:
    cols = ["source_file", "label", "timestamp_s", *CHANNELS]
    e = pq.read_table(euler_path, columns=cols).to_pandas()
    r = pq.read_table(rk_path, columns=cols + ["grid_dc_plant_solver"]).to_pandas()
    assert set(r["grid_dc_plant_solver"].unique()) == {"rk45"}
    ge, gr = dict(tuple(e.groupby("source_file", sort=False))), dict(tuple(r.groupby("source_file", sort=False)))
    assert set(ge) == set(gr), "the two datasets do not contain the same runs"
    rows = []
    for sid, a in ge.items():
        a, b = a.sort_values("timestamp_s"), gr[sid].sort_values("timestamp_s")
        assert len(a) == len(b) and np.array_equal(a["timestamp_s"].to_numpy(), b["timestamp_s"].to_numpy())
        t = b["timestamp_s"].to_numpy()
        rec = {"run": sid, "label": a["label"].iloc[0]}
        for seg, m in (("startup", t < SETTLE_S), ("steady", t >= SETTLE_S)):
            for ch in CHANNELS:
                x, y = a[ch].to_numpy(float), b[ch].to_numpy(float)
                rec[f"{ch}_{seg}"] = float(np.sqrt(np.mean((x[m] - y[m]) ** 2)) / np.ptp(y)) if np.ptp(y) > 0 else float("nan")
        rows.append(rec)
    by_label = {}
    for lab in sorted({x["label"] for x in rows}):
        sel = [x for x in rows if x["label"] == lab]
        by_label[lab] = {"runs": len(sel), **{k: {"median": float(np.nanmedian([x[k] for x in sel])), "max": float(np.nanmax([x[k] for x in sel]))}
                                              for k in sel[0] if k not in ("run", "label")}}
    worst = sorted(rows, key=lambda x: -x["current_r_steady"])[:5]
    return {"euler": {"path": str(euler_path.relative_to(REPO)), "sha256": sha256(euler_path), "rows": len(e)},
            "rk45": {"path": str(rk_path.relative_to(REPO)), "sha256": sha256(rk_path), "rows": len(r)},
            "runs": len(rows), "runs_per_label": {k: v["runs"] for k, v in by_label.items()}, "by_label": by_label,
            "all_runs": {k: {"median": float(np.nanmedian([x[k] for x in rows])), "max": float(np.nanmax([x[k] for x in rows]))} for k in rows[0] if k not in ("run", "label")},
            "worst_current_steady": [{"run": x["run"], "label": x["label"], "current_r_steady": x["current_r_steady"]} for x in worst]}


def _md(res: dict, cmd: str) -> str:
    L = ["# R3 — Classifier training datasets regenerated with the RK45 reference", "",
         f"Command: `{cmd}`. Generation: `python experiments/generate_diagnosis_grid_dataset.py --dc-plant-solver rk45 [--faults outer_race,inner_race,ball,cage] --out data/<name>_rk45.parquet --workers 11` "
         "(same grid, seeds, noise and plant-drift draws as the originals: `--seeds 3 --duration-s 2.5 --seed 7`).", "",
         "Decision: user, 2026-10-25, after R2 (S-16 affects diagnosis, S-18): regenerate the datasets with the reference. **Not done:** retraining, promotion, "
         "registry or default changes. The Euler datasets are kept unchanged; the promoted models' `run_metadata.json` still point to them.", "",
         "**Reproduction check before generating** (`r3_euler_reproduction.json`): the generator in Euler mode reproduces 6 sampled runs of the stored datasets "
         "with current_r, rpm, torque_nm and voltage_v bit-identical; acc_x/y/z differ by ≤ 4.7e-15 (floating-point rounding). So the RK45 datasets differ from the "
         "originals only by the plant integration.", ""]
    for name, r in res["pairs"].items():
        L += [f"## {name}", "", "| | Path | Rows | SHA-256 |", "|---|---|---|---|",
              f"| Euler (original) | `{r['euler']['path']}` | {r['euler']['rows']} | `{r['euler']['sha256'][:16]}…` |",
              f"| RK45 (new) | `{r['rk45']['path']}` | {r['rk45']['rows']} | `{r['rk45']['sha256'][:16]}…` |", "",
              f"{r['runs']} paired runs ({', '.join(f'{k}: {v}' for k, v in r['runs_per_label'].items())}). NRMS RK45 vs Euler per run (RMS difference / RK45 range of the run):", "",
              "| Label | current_r start-up median / max | current_r steady median / max | rpm steady median / max | acc_x steady median / max |", "|---|---|---|---|---|"]
        for lab, s in list(r["by_label"].items()) + [("all", r["all_runs"])]:
            f = lambda k: f"{s[k]['median']:.2e} / {s[k]['max']:.2e}"
            L.append(f"| {lab} | {f('current_r_startup')} | {f('current_r_steady')} | {f('rpm_steady')} | {f('acc_x_steady')} |")
        L += ["", "Largest steady-state current differences: " + "; ".join(f"`{x['run']}` {x['current_r_steady']:.3f}" for x in r["worst_current_steady"]) + ".", ""]
    L += ["**Next step (needs the user's decision):** retraining the classifiers on these datasets (and a promotion decision) is not part of this task.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "audit" / "plecs_om")
    args = ap.parse_args(argv)
    cmd = "python experiments/report_r3_datasets_rk45.py " + " ".join(argv if argv is not None else sys.argv[1:])
    res = {"command": cmd, "host": platform.platform(), "python": platform.python_version(),
           "pairs": {name: compare(REPO / e, REPO / r) for name, (e, r) in PAIRS.items()}}
    (args.out / "r3_datasets_rk45.json").write_text(json.dumps(res, indent=2) + "\n")
    (args.out / "r3_datasets_rk45.md").write_text(_md(res, cmd.strip()))
    print(_md(res, cmd.strip()))


if __name__ == "__main__":
    main()
