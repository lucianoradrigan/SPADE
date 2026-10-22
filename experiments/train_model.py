"""Single parametrized training entry point (docs/design_ai_layer_transversal.md Sec. 6.4, Sec. 8
step 5): `python experiments/train_model.py --config <path> --dataset <parquet path>` builds
whichever model architecture the config describes (a classifier or a forecaster, detected from the
config's own fields -- see _load_any_config), trains it, evaluates against a grouped-by-run test
split, and saves the three artifacts Sec. 6.4 requires together -- the exact config file, the
weights, and the resulting metrics -- under one timestamped run directory
(<config_dir>/<config_stem>/<date>_run<NN>/), so any checkpoint stays traceable to the exact
hyperparameters that produced it.

Does NOT replace Fase C's or Fase D.1's own specific entregables (INSTRUCTIONS.md Sec. 5/6): Fase
C's classifier is trained on the Fase A dataset with its own acceptance criteria (F1/MAE vs.
paper_federative), and D.1's dpc_tracking_forecaster explicitly adapts
regressors/envelope_forecaster.py, not the config-driven build_forecaster this script also drives
(see regressors/builder.py's module docstring for why the two forecaster families coexist). This
script is the shared plumbing Sec. 6.4 describes for the config-driven builder family
(models/*/schemas.py + builder.py, Sec. 8 step 4); running it against a real Fase A/B dataset for
Fase C/D.1 is a separate, larger decision than adding the script itself -- no such dataset exists
in this repo checkout yet (see docs/design_ai_layer_transversal.md's status note for the step this
script was added at).

Usage:
    python experiments/train_model.py --config configs/classifiers/pc_server.yaml --dataset path/to/dataset.parquet
    python experiments/train_model.py --config configs/regressors/pc_full.yaml --dataset path/to/dataset.parquet

    # Distillation (Sec. 4.1/4.2 "distillado del modelo de PC", Sec. 8 step 8) -- auto-resolves
    # the promoted PC-tier run for the SAME (domain, block) from the registry as teacher; requires
    # one to already be promoted (experiments/promote_run.py). --distill-alpha weighs the true/
    # direct target against the teacher's own prediction (1.0 = no distillation, 0.0 = pure
    # mimicry of the teacher) -- see _blend_classifier_targets/_blend_regressor_targets.
    python experiments/train_model.py --config configs/classifiers/rpi5_resnet1d_se.yaml --dataset path/to/dataset.parquet --distill
"""

import argparse
from pathlib import Path

# The training code lives in driveflow.models.training (importable by the GUI and the job worker);
# these names are re-exported so existing imports from this script keep working.
from driveflow.models.training import (  # noqa: F401
    _KIND_TO_REGISTRY_BLOCK,
    _PLANT_CONFIG_ID_BY_DOMAIN,
    _blend_classifier_targets,
    _blend_regressor_targets,
    _channels_for_domain,
    _load_any_config,
    _load_domain_dataframe,
    _next_run_dir,
    _save_artifacts,
    _train_classifier,
    _train_dab,
    _train_forecaster,
    train_from_config,
)
from driveflow.ai.registry import RegistryError
from driveflow.models.common import RPM_LEVEL_SCALING


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Path to a classifier or forecaster config YAML (Sec. 6.2/6.3).")
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet dataset exported via driveflow.datagen.export_parquet.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill", action="store_true", help="Blend targets with the promoted PC-tier model's own predictions (Sec. 8 step 8).")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight on the true/direct target vs. the teacher's prediction.")
    parser.add_argument("--keep-rpm-level", action="store_true", help="Classifiers: keep the rpm channel's absolute level (fixed scaling) instead of z-scoring it per window (AI-N07).")
    parser.add_argument("--cap-per-class", type=int, default=500, help="Classifiers: max windows per class.")
    parser.add_argument("--teacher-run", type=Path, default=None, help="With --distill: teacher run directory instead of the promoted PC-tier run.")
    args = parser.parse_args(argv)

    try:
        run_dir = train_from_config(
            args.config, args.dataset, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            distill=args.distill, distill_alpha=args.distill_alpha,
            level_scaling=RPM_LEVEL_SCALING if args.keep_rpm_level else None, cap_per_class=args.cap_per_class, teacher_run=args.teacher_run,
        )
    except RegistryError as exc:
        raise SystemExit(f"--distill requires a promoted PC-tier run first (experiments/promote_run.py): {exc}") from exc
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
