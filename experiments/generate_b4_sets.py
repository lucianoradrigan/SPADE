"""B4 (docs/audit/b4/b4_preregistration.md, approved 2026-11-09): the sets that are not plain grids.

    python experiments/generate_b4_sets.py e4 --grid 3class|5class --out data/b4_e4_<grid>.parquet
    python experiments/generate_b4_sets.py offgrid --out data/b4_offgrid_healthy.parquet

e4      §3.4: held-out low speeds (configs/b4_plan.yaml e4_low_speed_heldout_rad_s = 50, 70 rad/s) x severities x1.0, x1.5
        x every class of the tier x 3 seeds; noise and plant drift drawn as in the grid (RNG seed 20261120); run seeds
        from 90001; 2.5 s. Never used for training.
offgrid §3.5 (ESP32 training only): the 20 healthy conditions of configs/b4_plan.yaml x 3 seeds, nominal plant and no
        noise unless the condition says so; run seeds from 85001; 2.5 s.
Plant: dc_perm_ex_v2 (converter dcm_exact). Output columns follow generate_diagnosis_grid_dataset (grid_* columns).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import generate_diagnosis_grid_dataset as grid  # noqa: E402

PLAN = yaml.safe_load((REPO / "configs" / "b4_plan.yaml").read_text())
E4_RNG_SEED, E4_FIRST_RUN_SEED, OFFGRID_FIRST_RUN_SEED = 20261120, 90001, 85001
CLASSES = {"3class": (None, "outer_race", "inner_race"), "5class": (None, "outer_race", "inner_race", "ball", "cage")}


def e4_jobs(grid_name: str) -> list:
    rng = np.random.default_rng(E4_RNG_SEED)
    jobs, run_seed = [], E4_FIRST_RUN_SEED - 1
    for speed in PLAN["e4_low_speed_heldout_rad_s"]:
        for sev in PLAN["e4_severity_scales"]:
            for _ in range(3):
                for fault in CLASSES[grid_name]:
                    run_seed += 1
                    noisy = rng.random() >= grid.NOISE_FREE_SHARE
                    e_noise, m_noise = float(rng.uniform(0, grid.MAX_NOISE_PCT)), float(rng.uniform(0, grid.MAX_NOISE_PCT))
                    jobs.append({"fault": fault, "speed": float(speed), "severity_scale": sev if fault else 0.0, "seed": run_seed, "duration_s": 2.5,
                                 "electrical_noise_pct": e_noise if noisy else 0.0, "mechanical_noise_pct": m_noise if noisy else 0.0,
                                 "psi_e": grid.NOMINAL_PSI_E * float(rng.uniform(0.9, 1.1)), "j_rotor": grid.NOMINAL_J_ROTOR * float(rng.uniform(0.8, 1.5)),
                                 "plant_config_id": PLAN["plant"]})
    return jobs


def _offgrid_run(job: dict) -> pd.DataFrame:
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    c = job["condition"]
    kw = {"omega_ref_rad_s": float(c["omega_ref_rad_s"])}
    for k in ("load_torque_nm", "electrical_noise_pct", "mechanical_noise_pct", "dc_supply_v"):
        if k in c:
            kw[k] = float(c[k])
    if "psi_e_scale" in c or "j_rotor_scale" in c:
        kw["motor_parameter_overrides"] = {"psi_e": grid.NOMINAL_PSI_E * c.get("psi_e_scale", 1.0), "j_rotor": grid.NOMINAL_J_ROTOR * c.get("j_rotor_scale", 1.0)}
    sid = f"b4_offgrid_{job['index']:02d}_seed{job['seed']}"
    df = records_to_dataframe(run_scenario(Scenario(scenario_id=sid, plant_config_id=PLAN["plant"], fault_type=None, seed=job["seed"], duration_s=2.5, **kw)))
    df["grid_speed"] = float(c["omega_ref_rad_s"])
    df["grid_severity_scale"] = 0.0
    df["b4_offgrid_condition"] = job["index"]
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=("e4", "offgrid"))
    ap.add_argument("--grid", choices=tuple(CLASSES), default="5class")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        if args.set == "e4":
            jobs = e4_jobs(args.grid)
            frames = pool.map(grid._simulate, jobs)
        else:
            n = PLAN["esp32_offgrid_healthy_seeds_per_condition"]
            jobs = [{"condition": c, "index": i, "seed": OFFGRID_FIRST_RUN_SEED + i * n + k} for i, c in enumerate(PLAN["esp32_offgrid_healthy"]) for k in range(n)]
            frames = pool.map(_offgrid_run, jobs)
    df = pd.concat(frames, ignore_index=True)
    df["b4_set"] = args.set if args.set == "offgrid" else f"e4_{args.grid}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"{args.set}: {len(jobs)} runs, {len(df)} rows -> {args.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
