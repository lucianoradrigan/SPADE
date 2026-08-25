"""Generates a real Fase C training dataset (INSTRUCTIONS.md Sec. 5) -- larger volume than A.5's
closing-criterion dataset, sized for actually training the sensor/gateway/envelope models, not
just proving fault frequencies show up.

Usage:
    python experiments/generate_diagnosis_dataset.py --out data/diagnosis_dataset.parquet
"""

import argparse
from pathlib import Path

from driveflow.datagen import export_parquet, healthy_and_faulted_grid, run_scenarios

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "diagnosis_dataset.parquet"

#: outer_race/inner_race only -- the two AUC-validated fault types (ball/cage reuse inner_race's
#: severity as a documented placeholder, see datagen/scenario.py -- not included here to keep
#: this comparison against paper_federative's reported metrics on validated ground, not a guess).
FAULT_TYPES = ("outer_race", "inner_race")
SEEDS = tuple(range(10))
DURATION_S = 5.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scenarios = healthy_and_faulted_grid(
        "c1_diag", fault_types=FAULT_TYPES, seeds=SEEDS, duration_s=DURATION_S, omega_ref_rad_s=150.0
    )
    print(f"{len(scenarios)} scenarios: {len(SEEDS)} seeds x {1 + len(FAULT_TYPES)} conditions, {DURATION_S}s each")

    runs = run_scenarios(scenarios)
    df = export_parquet(runs, args.out)
    print(f"\nwrote {len(df)} rows to {args.out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
