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
"""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, load_classifier_config
from driveflow.models.common import (
    VSC_DPC_CANDIDATE_CHANNELS,
    build_classification_windows,
    build_direct_forecast_windows,
    discover_channels,
    grouped_split,
    load_diagnosis_dataset,
    prepare_classification_splits,
)
from driveflow.models.regressors.builder import build_forecaster
from driveflow.models.regressors.schemas import ForecasterConfig, load_forecaster_config

#: domain -> plant_config_id, matching driveflow.datagen.scenario.Scenario._VALID_PAIRS. Only the
#: two domains that exist today -- extend when a third domain is added.
_PLANT_CONFIG_ID_BY_DOMAIN = {"dc_motor": "dc_perm_ex_v1", "vsc_dpc": "vsc_dpc_v1"}


def _load_any_config(config_path: Path):
    """Detects whether config_path is a classifier config ('blocks', Sec. 6.2) or a forecaster
    config ('recurrent_type', Sec. 6.3) and loads+validates it with the matching schema."""
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a mapping, got {raw!r}")
    if "blocks" in raw:
        return "classifier", load_classifier_config(config_path)
    if "recurrent_type" in raw:
        return "forecaster", load_forecaster_config(config_path)
    raise ValueError(f"{config_path}: cannot tell if this is a classifier config ('blocks' key) or a forecaster config ('recurrent_type' key)")


def _load_domain_dataframe(dataset_path: Path, domain: str) -> pd.DataFrame:
    if domain == "dc_motor":
        # The mandatory Fase C safeguard (INSTRUCTIONS.md Sec. 5 punto 3) -- excludes vsc_dpc_v1
        # rows, which have NaN in every dc_motor diagnosis column.
        return load_diagnosis_dataset(dataset_path)
    if domain == "vsc_dpc":
        df = pd.read_parquet(dataset_path)
        return df[df["plant_config_id"] == _PLANT_CONFIG_ID_BY_DOMAIN["vsc_dpc"]].reset_index(drop=True)
    raise ValueError(f"unknown domain {domain!r} -- expected one of {sorted(_PLANT_CONFIG_ID_BY_DOMAIN)}")


def _channels_for_domain(df: pd.DataFrame, domain: str, window_samples: int) -> list:
    if domain == "dc_motor":
        return discover_channels(df, window_samples)
    if domain == "vsc_dpc":
        return [c for c in VSC_DPC_CANDIDATE_CHANNELS if c in df.columns and df[c].notna().mean() > 0.5]
    raise ValueError(f"unknown domain {domain!r} -- expected one of {sorted(_PLANT_CONFIG_ID_BY_DOMAIN)}")


def _train_classifier(config: ClassifierConfig, df: pd.DataFrame, epochs: int, batch_size: int, seed: int) -> tuple:
    channels = _channels_for_domain(df, config.domain, config.input_window)
    if not channels:
        raise ValueError(f"no live channels found for domain {config.domain!r} at input_window={config.input_window}")
    classes = sorted(df["label"].unique())

    X, y_str, groups = build_classification_windows(df, classes, channels, config.input_window)
    if X is None:
        raise ValueError("no classification windows could be built -- dataset too short for input_window")

    X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_classification_splits(X, y_str, classes, groups, seed=seed)

    model = build_classifier(config, n_channels=len(channels))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    metrics = {
        "kind": "classifier",
        "domain": config.domain,
        "tier": config.tier,
        "channels": channels,
        "classes": [str(c) for c in le.classes_],
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "test_accuracy": float(np.mean(y_pred == y_test)) if len(y_test) else None,
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")) if len(y_test) else None,
    }
    return model, metrics


def _train_forecaster(config: ForecasterConfig, df: pd.DataFrame, epochs: int, batch_size: int, seed: int) -> tuple:
    channels = _channels_for_domain(df, config.domain, config.input_window)
    if not channels:
        raise ValueError(f"no channels found for domain {config.domain!r}")

    X, Y, groups = build_direct_forecast_windows(df, channels, config.input_window, config.horizon)
    if X is None:
        raise ValueError("no forecast windows could be built -- dataset too short for input_window+horizon")

    # No discrete label to stratify by -- grouped_split's stratified attempt fails on a
    # single-valued y_pool and falls back to a plain grouped split (see splits.py).
    dummy_labels = np.zeros(len(X), dtype=int)
    idx_train, idx_test = grouped_split(np.arange(len(X)), dummy_labels, groups, test_frac=0.2, seed=seed)

    model = build_forecaster(config, n_channels=len(channels))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X[idx_train], Y[idx_train], epochs=epochs, batch_size=batch_size, verbose=0)

    Y_pred = model.predict(X[idx_test], verbose=0) if len(idx_test) else np.empty((0, config.horizon, len(channels)))
    mse = float(np.mean((Y_pred - Y[idx_test]) ** 2)) if len(idx_test) else None
    metrics = {
        "kind": "forecaster",
        "domain": config.domain,
        "tier": config.tier,
        "channels": channels,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)) if mse is not None else None,
    }
    return model, metrics


def _next_run_dir(config_path: Path) -> Path:
    """<config_dir>/<config_stem>/<date>_run<NN>/, matching Sec. 6.4's own example layout."""
    run_root = config_path.parent / config_path.stem
    date_str = dt.date.today().isoformat()
    n = 1
    while True:
        candidate = run_root / f"{date_str}_run{n:02d}"
        if not candidate.exists():
            return candidate
        n += 1


def _save_artifacts(run_dir: Path, config_path: Path, model, metrics: dict) -> None:
    """Saves the three artifacts Sec. 6.4 requires together -- the exact config that produced
    this run (copied verbatim, not re-serialized -- keeps comments/formatting), the weights, and
    the resulting metrics."""
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    model.save_weights(run_dir / "model.weights.h5")
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def train_from_config(config_path, dataset_path, epochs: int = 20, batch_size: int = 32, seed: int = 42) -> Path:
    """Loads config_path (classifier or forecaster, auto-detected), trains it against
    dataset_path (a Parquet file exported via driveflow.datagen.export_parquet), and saves the
    run's artifacts. Returns the run directory."""
    config_path = Path(config_path)
    kind, config = _load_any_config(config_path)
    df = _load_domain_dataframe(Path(dataset_path), config.domain)

    if kind == "classifier":
        model, metrics = _train_classifier(config, df, epochs, batch_size, seed)
    else:
        model, metrics = _train_forecaster(config, df, epochs, batch_size, seed)

    run_dir = _next_run_dir(config_path)
    _save_artifacts(run_dir, config_path, model, metrics)
    return run_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Path to a classifier or forecaster config YAML (Sec. 6.2/6.3).")
    parser.add_argument("--dataset", required=True, type=Path, help="Parquet dataset exported via driveflow.datagen.export_parquet.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_dir = train_from_config(args.config, args.dataset, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
