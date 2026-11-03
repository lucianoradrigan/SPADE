"""AI-N27 option 2: preconditions P1-P3 of docs/audit/assistant/ai_n27_regen_preregistration.md §1.3.

  p1  before generating: the grid generator, in default mode, reproduces the first, middle and last
      job of each stored grid (current_r, rpm, voltage_v, torque_nm bit-identical; acc_x/y/z max
      relative difference <= 1e-12, as amendment 2).
  p2  after generating: in each dcm grid every noise-free acquisition has min current_r >= -1e-6 A,
      every row is tagged dcm / 1000, and acquisitions, order, rows and timestamps equal the stored grid.
  p3  after training: each candidate's split (group hashes per split, n_windows) equals its reference.

Each writes docs/audit/assistant/ai_n27_regen_<p>.json and exits 1 if the precondition fails.

Usage: python experiments/report_ai_n27_regen_preconditions.py p1|p2
       python experiments/report_ai_n27_regen_preconditions.py p3 --pairs <reference run>=<candidate run> ...
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
OUT = REPO / "docs" / "audit" / "assistant"
GRIDS = {"3class": ("data/diagnosis_grid_dataset.parquet", "data/diagnosis_grid_dataset_dcm.parquet", (None, "outer_race", "inner_race")),
         "5class": ("data/diagnosis_grid_dataset_5class.parquet", "data/diagnosis_grid_dataset_5class_dcm.parquet", (None, "outer_race", "inner_race", "ball", "cage"))}
PLANT_COLS = ("current_r", "rpm", "voltage_v", "torque_nm")
VIB_COLS = ("acc_x", "acc_y", "acc_z")
VIB_REL_TOL = 1e-12
NEG_CURRENT_TOL_A = -1e-6


def p1() -> dict:
    import pandas as pd

    import generate_diagnosis_grid_dataset as g

    res = {}
    for name, (stored, _new, faults) in GRIDS.items():
        jobs = g.build_jobs(3, 2.5, 7, faults)
        for j in (jobs[0], jobs[len(jobs) // 2], jobs[-1]):
            new = g._simulate(j).sort_values("timestamp_s").reset_index(drop=True)
            sid = new["source_file"].iloc[0]
            old = pq.read_table(REPO / stored, filters=[("source_file", "==", sid)]).to_pandas().sort_values("timestamp_s").reset_index(drop=True)
            rec = {"rows": [len(old), len(new)], "plant_bit_identical": {}, "vib_max_rel_diff": {}}
            for c in PLANT_COLS:
                rec["plant_bit_identical"][c] = bool(len(old) == len(new) and np.array_equal(old[c].to_numpy(), new[c].to_numpy(), equal_nan=True))
            for c in VIB_COLS:
                a, b = old[c].to_numpy(float), new[c].to_numpy(float)
                rec["vib_max_rel_diff"][c] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300)) if len(a) == len(b) else float("inf")
            rec["ok"] = all(rec["plant_bit_identical"].values()) and all(v <= VIB_REL_TOL for v in rec["vib_max_rel_diff"].values())
            res[f"{name}:{sid}"] = rec
    return {"cases": res, "ok": all(r["ok"] for r in res.values())}


def p2() -> dict:
    res = {}
    for name, (stored, new, _f) in GRIDS.items():
        cols = ["source_file", "timestamp_s", "current_r", "grid_electrical_noise_pct", "grid_dc_converter_model", "grid_dc_plant_substeps"]
        d = pq.read_table(REPO / new, columns=cols).to_pandas()
        s = pq.read_table(REPO / stored, columns=["source_file", "timestamp_s"]).to_pandas()
        same_order = list(dict.fromkeys(d["source_file"])) == list(dict.fromkeys(s["source_file"]))
        same_rows = len(d) == len(s) and np.array_equal(d["source_file"].to_numpy(), s["source_file"].to_numpy()) and np.array_equal(d["timestamp_s"].to_numpy(), s["timestamp_s"].to_numpy())
        nf = d[d["grid_electrical_noise_pct"] == 0.0]
        mins = nf.groupby("source_file")["current_r"].min()
        tags = set(d["grid_dc_converter_model"]) == {"dcm"} and set(d["grid_dc_plant_substeps"]) == {1000}
        res[name] = {"path": new, "acquisitions": int(d["source_file"].nunique()), "rows": len(d), "same_acquisitions_order": same_order,
                     "same_rows_and_timestamps": bool(same_rows), "tagged_dcm_1000": bool(tags), "noise_free_acquisitions": int(len(mins)),
                     "noise_free_min_current_a": float(mins.min()), "noise_free_below_tol": int((mins < NEG_CURRENT_TOL_A).sum())}
        res[name]["ok"] = bool(same_order and same_rows and tags and res[name]["noise_free_below_tol"] == 0)
    return {"grids": res, "ok": all(r["ok"] for r in res.values())}


def p3(pairs: list) -> dict:
    res = {}
    for pair in pairs:
        ref, cand = pair.split("=", 1)
        a, b = (json.loads((REPO / r / "metrics.json").read_text())["split"] for r in (ref, cand))
        res[cand] = {"reference": ref, "group_hashes_equal": a["group_hashes"] == b["group_hashes"], "n_windows": [a["n_windows"], b["n_windows"]],
                     "ok": a["group_hashes"] == b["group_hashes"] and a["n_windows"] == b["n_windows"]}
    return {"candidates": res, "ok": all(r["ok"] for r in res.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=("p1", "p2", "p3"))
    ap.add_argument("--pairs", nargs="*", default=[])
    args = ap.parse_args(argv)
    res = {"p1": p1, "p2": p2}[args.which]() if args.which != "p3" else p3(args.pairs)
    res = {"command": "python experiments/report_ai_n27_regen_preconditions.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "host": platform.platform(), "python": platform.python_version(), **res}
    (OUT / f"ai_n27_regen_{args.which}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
