"""AI-N27 option 2, evaluation set E2a (docs/audit/assistant/ai_n27_regen_preregistration.md §4.2):
new held-out runs in the regime where the S-21 converter artifact appears, simulated with the
discontinuous-conduction converter (S21-2) at 1000 substeps. Never used for training.

Cells: every (speed, severity) cell of the training grids where the S21-1 scan
(docs/audit/assistant/s21_dataset_scan.md) found affected acquisitions. Every class of the tier per
cell, healthy included. Noise and plant drift are drawn as in generate_diagnosis_grid_dataset.
build_jobs (one third noise-free, the rest U(0, 100) %; psi_e x U(0.9, 1.1); J x U(0.8, 1.5)) from
RNG seed 20261103; run seeds start at 60001 (the grid uses 1001-1525, the evaluation cache 200-203).

Usage:
    python experiments/generate_ai_n27_heldout.py --grid 3class --out data/ai_n27_heldout_3class_dcm.parquet --workers 10
    python experiments/generate_ai_n27_heldout.py --grid 5class --out data/ai_n27_heldout_5class_dcm.parquet --workers 10

New draws (AI-N28 repair round, E2a', esp32_repair_preregistration.md section 5.1): the same cells and
distributions with other seeds; the defaults keep the original set reproducible:
    python experiments/generate_ai_n27_heldout.py --grid 3class --rng-seed 20261108 --first-run-seed 70001 \
        --set-name esp32_repair_E2a_prime --out data/esp32_repair_heldout_3class_dcm.parquet --workers 10
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_diagnosis_grid_dataset import MAX_NOISE_PCT, NOISE_FREE_SHARE, NOMINAL_J_ROTOR, NOMINAL_PSI_E, _simulate  # noqa: E402

RNG_SEED = 20261103
FIRST_RUN_SEED = 60001
DURATION_S = 2.5
PLANT = {"dc_converter_model": "dcm", "dc_plant_substeps": 1000}
#: grid -> (classes besides healthy, seeds per class and cell, affected (speed rad/s, severity scale) cells of s21_dataset_scan)
GRIDS = {
    "3class": (("outer_race", "inner_race"), 3,
               ((80.0, 1.0), (80.0, 1.5), (80.0, 2.0), (100.0, 1.5), (100.0, 2.0))),
    "5class": (("outer_race", "inner_race", "ball", "cage"), 1,
               ((80.0, 0.75), (80.0, 1.0), (80.0, 1.5), (80.0, 2.0), (100.0, 0.75), (100.0, 1.0), (100.0, 1.5), (100.0, 2.0),
                (125.0, 1.0), (125.0, 1.5), (125.0, 2.0), (150.0, 1.5), (150.0, 2.0), (175.0, 1.5), (175.0, 2.0),
                (200.0, 1.5), (200.0, 2.0), (220.0, 2.0))),
}


def build_jobs(grid: str, rng_seed: int = RNG_SEED, first_run_seed: int = FIRST_RUN_SEED) -> list:
    faults, n_seeds, cells = GRIDS[grid]
    rng = np.random.default_rng(rng_seed)
    jobs, run_seed = [], first_run_seed - 1
    for speed, sev in cells:
        for _ in range(n_seeds):
            for fault in (None, *faults):
                run_seed += 1
                noisy = rng.random() >= NOISE_FREE_SHARE
                e_noise, m_noise = float(rng.uniform(0, MAX_NOISE_PCT)), float(rng.uniform(0, MAX_NOISE_PCT))
                jobs.append({
                    "fault": fault, "speed": speed, "severity_scale": sev if fault else 0.0, "seed": run_seed, "duration_s": DURATION_S,
                    "electrical_noise_pct": e_noise if noisy else 0.0, "mechanical_noise_pct": m_noise if noisy else 0.0,
                    "psi_e": NOMINAL_PSI_E * float(rng.uniform(0.9, 1.1)), "j_rotor": NOMINAL_J_ROTOR * float(rng.uniform(0.8, 1.5)),
                    **PLANT,
                })
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=tuple(GRIDS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED, help="draws of noise and plant drift (default: the original E2a)")
    ap.add_argument("--first-run-seed", type=int, default=FIRST_RUN_SEED, help="first run seed (default: the original E2a)")
    ap.add_argument("--set-name", default="ai_n27_E2a", help="value of the heldout_set column")
    args = ap.parse_args(argv)
    jobs = build_jobs(args.grid, args.rng_seed, args.first_run_seed)
    print(f"{len(jobs)} held-out runs ({args.grid}), {DURATION_S} s each, plant {PLANT}")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        frames = pool.map(_simulate, jobs)
    df = pd.concat(frames, ignore_index=True)
    df["heldout_set"] = args.set_name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows to {args.out} in {time.time() - t0:.0f} s")
    print(df.groupby("label")["source_file"].nunique())


if __name__ == "__main__":
    main()
