"""B4 §2 preconditions (pre-registration approved 2026-11-09), checked before any B4 dataset is generated.

Q1: the first, middle and last acquisition (build_jobs order) of the stored 5-class dcm grid
    (data/diagnosis_grid_dataset_5class_dcm.parquet, dcm x1000) regenerated with --plant dc_perm_ex_v2:
    NRMS <= 1e-4 for current_r, rpm, torque_nm (§2 as written), and min current_r >= -1e-6 A for noise-free runs.
Q3: the E4 (90001-) and E5 (95001-) run seeds do not overlap the B4 grid seeds (80001-), the off-grid seeds
    (85001-), any earlier grid seed (1001-) or the evaluation seeds (report_classifier_evaluation, 200-203).
Q2 (split by acquisition, on the windows) is checked on each trained candidate in report_b4.py.

    python experiments/b4_preconditions.py
Writes docs/audit/b4/b4_preconditions.{md,json} (simulated).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "docs" / "audit" / "b4" / "b4_preconditions"
NRMS_MAX = 1e-4
CHANNELS = ("current_r", "rpm", "torque_nm")


def _nrms(a, ref):
    rng = float(ref.max() - ref.min())
    d = float(np.sqrt(np.mean((a - ref) ** 2)))
    return d / rng if rng > 0 else d


def main() -> None:
    from generate_diagnosis_grid_dataset import _simulate, build_jobs

    jobs = build_jobs(3, 2.5, 7, (None, "outer_race", "inner_race", "ball", "cage"))
    sel = [jobs[0], jobs[len(jobs) // 2], jobs[-1]]
    with mp.get_context("spawn").Pool(3) as pool:
        frames = pool.map(_simulate, [{**j, "plant_config_id": "dc_perm_ex_v2"} for j in sel])
    stored = pd.read_parquet(REPO / "data" / "diagnosis_grid_dataset_5class_dcm.parquet")
    runs = []
    for j, new in zip(sel, frames):
        sid = str(new["source_file"].iloc[0])
        old = stored[stored["source_file"].astype(str) == sid].reset_index(drop=True)
        assert len(old) == len(new) and len(old) > 0, f"{sid}: {len(old)} stored rows vs {len(new)}"
        noise_free = j["electrical_noise_pct"] == 0 and j["mechanical_noise_pct"] == 0
        r = {"source": sid, "fault": j["fault"], "speed": j["speed"], "severity_scale": j["severity_scale"], "noise_free": noise_free,
             "nrms": {c: _nrms(new[c].to_numpy(), old[c].to_numpy()) for c in CHANNELS}, "min_current_r_a": float(new["current_r"].min())}
        r["pass"] = all(v <= NRMS_MAX for v in r["nrms"].values()) and (not noise_free or r["min_current_r_a"] >= -1e-6)
        runs.append(r)
    plan = yaml.safe_load((REPO / "configs" / "b4_plan.yaml").read_text())
    n_off = len(plan["esp32_offgrid_healthy"]) * plan["esp32_offgrid_healthy_seeds_per_condition"]
    seeds = {"b4_grid_5class": range(80001, 80001 + 675), "b4_grid_3class": range(80001, 80001 + 405), "offgrid": range(85001, 85001 + n_off),
             "e4_5class": range(90001, 90001 + 60), "e4_3class": range(90001, 90001 + 36), "e5": range(95001, 95001 + 30),
             "earlier_grids": range(1001, 1001 + 675), "evaluation": range(200, 204)}
    held = {"e4_5class", "e4_3class", "e5"}
    overlaps = {f"{a}~{b}": len(set(seeds[a]) & set(seeds[b])) for a in sorted(held) for b in seeds if b not in held}
    res = {"command": "python experiments/b4_preconditions.py", "simulated": True, "preregistration": "docs/audit/b4/b4_preregistration.md §2",
           "Q1": {"runs": runs, "nrms_max": NRMS_MAX, "pass": all(r["pass"] for r in runs)},
           "Q3": {"seed_ranges": {k: [v.start, v.stop - 1] for k, v in seeds.items()}, "overlaps": overlaps, "pass": not any(overlaps.values())}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")
    L = ["# B4 preconditions (pre-registration §2)", "", f"**Simulated.** Command: `{res['command']}`. Raw: `b4_preconditions.json`.", "",
         f"## Q1: the generator reproduces the stored 5-class dcm grid with `dc_perm_ex_v2` — **{'pass' if res['Q1']['pass'] else 'FAIL'}**", "",
         f"NRMS ≤ {NRMS_MAX:g} per channel; noise-free runs: min `current_r` ≥ −1e-6 A.", "",
         "| Run | Noise-free | NRMS current_r | NRMS rpm | NRMS torque_nm | min current_r (A) | Pass |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        n = r["nrms"]
        L.append(f"| `{r['source']}` | {r['noise_free']} | {n['current_r']:.2e} | {n['rpm']:.2e} | {n['torque_nm']:.2e} | {r['min_current_r_a']:.3g} | {r['pass']} |")
    L += ["", f"## Q3: held-out seeds do not overlap — **{'pass' if res['Q3']['pass'] else 'FAIL'}**", "", "| Set | First | Last |", "|---|---|---|"]
    L += [f"| {k} | {a} | {b} |" for k, (a, b) in res["Q3"]["seed_ranges"].items()]
    L += ["", "Overlapping seeds between a held-out set and any other set: " + ", ".join(f"{k} {v}" for k, v in overlaps.items()), "",
          "## Q2", "", "Checked on the windows of each trained candidate (`report_b4.py`)."]
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print(json.dumps({"Q1": res["Q1"]["pass"], "Q3": res["Q3"]["pass"]}))


if __name__ == "__main__":
    main()
