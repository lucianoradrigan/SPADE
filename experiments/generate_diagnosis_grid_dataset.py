"""Operating-condition grid dataset for the dc_motor classifiers (docs/AI_TECHNICAL_AUDIT.md
AI-N07/AI-N12).

experiments/generate_diagnosis_dataset.py simulates ONE operating point (150 rad/s, calibrated
severity, no noise, nominal plant) and varies only the seed -- which changes only the vibration
channels (AUDIT_ANSWERS Q15). A model trained on it never sees another speed, so at 100 rad/s the
inner-race impulse rate (78.5 Hz) lands where the outer-race one was at 150 rad/s (73.3 Hz) and
every inner_race window is called outer_race.

This grid varies, per run:
  - speed reference: SPEEDS_RAD_S (7 levels, 80-220 rad/s);
  - fault severity: SEVERITY_SCALES x the calibrated electrical (8.0) and mechanical
    (CALIBRATED_MECHANICAL_SEVERITY) severities;
  - sensor noise: a NOISE_FREE_SHARE of the runs has none (the simulator's default, and what the
    GUI and the evaluation's nominal condition use); the others draw electrical and mechanical
    noise independently and uniformly in [0, MAX_NOISE_PCT] %. A first grid without noise-free runs
    and with at most 40 % noise misclassified a third of the noise-free healthy windows and every
    healthy window at 100 % noise (docs/AI_TECHNICAL_AUDIT.md AI-N07);
  - plant drift: psi_e x U(0.9, 1.1), j_rotor x U(0.8, 1.5);
  - seed.
Healthy runs get the same (speed, slot, seed) cells as the faulted ones, so every class has the
same number of runs at every speed. The draws come from one RNG seeded with --seed, so the dataset
is reproducible.

Held-out conditions for the generalization report (experiments/report_classifier_evaluation.py)
are deliberately NOT in this grid: 115 and 190 rad/s, severity x0.6, 250 rad/s (extrapolation).

Usage:
    python experiments/generate_diagnosis_grid_dataset.py --out data/diagnosis_grid_dataset.parquet
    # A2: five classes (ball and cage added; their amplitudes are SYNTHETIC-ONLY placeholders,
    # CALIBRATED_MECHANICAL_SEVERITY in datagen/scenario.py, not calibrated against real data)
    python experiments/generate_diagnosis_grid_dataset.py --faults outer_race,inner_race,ball,cage \
        --out data/diagnosis_grid_dataset_5class.parquet
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "diagnosis_grid_dataset.parquet"

SPEEDS_RAD_S = (80.0, 100.0, 125.0, 150.0, 175.0, 200.0, 220.0)
SEVERITY_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
FAULTS = (None, "outer_race", "inner_race")
MAX_NOISE_PCT = 100.0
NOISE_FREE_SHARE = 1 / 3
NOMINAL_PSI_E = 0.165
NOMINAL_J_ROTOR = 0.025


def build_jobs(n_seeds: int, duration_s: float, seed: int, faults: tuple = FAULTS) -> list:
    rng = np.random.default_rng(seed)
    jobs = []
    run_seed = 1000
    for speed in SPEEDS_RAD_S:
        for sev in SEVERITY_SCALES:
            for _ in range(n_seeds):
                for fault in faults:
                    run_seed += 1
                    noisy = rng.random() >= NOISE_FREE_SHARE
                    e_noise, m_noise = float(rng.uniform(0, MAX_NOISE_PCT)), float(rng.uniform(0, MAX_NOISE_PCT))
                    jobs.append({
                        "fault": fault, "speed": speed, "severity_scale": sev if fault else 0.0, "seed": run_seed, "duration_s": duration_s,
                        "electrical_noise_pct": e_noise if noisy else 0.0, "mechanical_noise_pct": m_noise if noisy else 0.0,
                        "psi_e": NOMINAL_PSI_E * float(rng.uniform(0.9, 1.1)), "j_rotor": NOMINAL_J_ROTOR * float(rng.uniform(0.8, 1.5)),
                    })
    return jobs


def _simulate(job: dict) -> pd.DataFrame:
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    fault = job["fault"]
    kw = {}
    if fault is not None:
        kw = {"electrical_severity": 8.0 * job["severity_scale"], "mechanical_severity": CALIBRATED_MECHANICAL_SEVERITY[fault] * job["severity_scale"]}
    sid = f"grid_{fault or 'normal'}_w{job['speed']:.0f}_s{job['severity_scale']:.2f}_seed{job['seed']}"
    sc = Scenario(
        scenario_id=sid, fault_type=fault, seed=job["seed"], duration_s=job["duration_s"], omega_ref_rad_s=job["speed"],
        electrical_noise_pct=job["electrical_noise_pct"], mechanical_noise_pct=job["mechanical_noise_pct"],
        motor_parameter_overrides={"psi_e": job["psi_e"], "j_rotor": job["j_rotor"]}, **kw,
    )
    df = records_to_dataframe(run_scenario(sc))
    for k in ("speed", "severity_scale", "electrical_noise_pct", "mechanical_noise_pct", "psi_e", "j_rotor"):
        df[f"grid_{k}"] = job[k]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, default=3, help="runs per (speed, severity, class) cell")
    ap.add_argument("--duration-s", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=7, help="seed of the noise / plant-drift draws")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--faults", default=",".join(f for f in FAULTS if f), help="comma-separated fault classes besides healthy (default: outer_race,inner_race)")
    args = ap.parse_args(argv)

    faults = (None,) + tuple(f.strip() for f in args.faults.split(",") if f.strip())
    jobs = build_jobs(args.seeds, args.duration_s, args.seed, faults)
    print(f"{len(jobs)} runs ({len(SPEEDS_RAD_S)} speeds x {len(SEVERITY_SCALES)} severities x {args.seeds} seeds x {len(faults)} classes), {args.duration_s} s each")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        frames = pool.map(_simulate, jobs)
    df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows to {args.out} in {time.time() - t0:.0f} s")
    print(df.groupby("label")["source_file"].nunique())


if __name__ == "__main__":
    main()
