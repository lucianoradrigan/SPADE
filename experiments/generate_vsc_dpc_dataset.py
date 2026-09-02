"""Generates a real vsc_dpc (Fase B) dataset for training the config-driven LSTM forecaster
(configs/regressors/pc_full.yaml, docs/design_ai_layer_transversal.md Sec. 4.1/6.3/8 step 5) --
same idea as generate_diagnosis_dataset.py for the dc_motor domain, but there is no equivalent
generator for this domain yet (Fase B never persisted its rollouts as a reusable Parquet dataset).

On-distribution only (R/magnitude/omega at their training defaults, per-run diversity comes only
from seed -> reference start phase, runner.py) -- deliberately NOT sweeping into the documented
low-R divergence region (docs/patch9_correccion_divergencia_dpc.md) or other off-distribution
robustness-probe territory (tests/test_dpc_robustness_grid.py already covers that; this dataset is
for training a forecaster, not for re-probing robustness).

Usage:
    python experiments/generate_vsc_dpc_dataset.py --out data/vsc_dpc_dataset.parquet
"""

import argparse
from pathlib import Path

from driveflow.datagen import Scenario, export_parquet, run_scenarios

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "vsc_dpc_dataset.parquet"

#: Each run's DPC network inference dominates runtime (~15-20s per 0.3s-duration run on CPU) --
#: sized to finish in a few minutes, not to be a large-scale dataset.
SEEDS = tuple(range(15))
DURATION_S = 0.3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scenarios = [
        Scenario(scenario_id=f"b1_vsc_seed{seed}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=DURATION_S, seed=seed)
        for seed in SEEDS
    ]
    print(f"{len(scenarios)} scenarios, {DURATION_S}s each")

    runs = run_scenarios(scenarios)
    df = export_parquet(runs, args.out)
    print(f"\nwrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
