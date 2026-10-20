"""Generates a PMSM dataset through the Scenario / run_scenario pipeline -- TODO_CLAUDE_CODE.md P1.

Both FOC policies of control/classical/pmsm_foc.py (MTPA and i_d = 0) at the same stator current
magnitudes, both torque signs, a few seeds: the same sweep as the PMSM page's MTPA-vs-naive cloud,
now as records with the documented schema (datagen/export_parquet.py SCHEMA_COLUMNS + PMSM_COLUMNS:
i_d, i_q, i_d_ref, i_q_ref, u_d, u_q, omega_rad_s; rpm and torque_nm from the motor state).
No fault model exists for the PMSM, so every record is labelled "normal"; this dataset is for
control-law analysis, not diagnosis. Datasets are not committed (data/ is git-ignored).

Usage:

    python experiments/generate_pmsm_dataset.py --out data/pmsm_dataset.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from driveflow.datagen import Scenario, run_scenarios
from driveflow.datagen.export_parquet import export_parquet

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENTS_A = (40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0, 360.0, 390.0)
POLICIES = ("FOC_MTPA", "FOC_naive")
SEEDS = (0, 1)
DURATION_S = 0.03


def scenarios() -> list:
    out = []
    for policy in POLICIES:
        for i_s in CURRENTS_A:
            for sign in (1.0, -1.0):
                for seed in SEEDS:
                    out.append(Scenario(scenario_id=f"pmsm_{policy}_{sign * i_s:+.0f}A_seed{seed}", controller_type=policy, plant_config_id="pmsm_v1",
                                        pmsm_current_ref_a=sign * i_s, seed=seed, duration_s=DURATION_S))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "pmsm_dataset.parquet")
    args = ap.parse_args(argv)
    sc = scenarios()
    df = export_parquet(run_scenarios(sc), args.out)
    print(f"{len(sc)} scenarios, {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
