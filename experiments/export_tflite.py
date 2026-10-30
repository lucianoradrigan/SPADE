"""Exports a promoted (domain, tier, block) run to TFLite -- float16 for rpi5, int8 (calibrated
on windows sampled from --dataset, the simulated training dataset) for esp32 -- docs/design_ai_layer_transversal.md
Sec. 4.1/4.2, Sec. 8 step 8. Saves alongside the run's existing artifacts as model.tflite.

Usage (run as a module -- the script imports experiments.train_model, so a direct
`python experiments/export_tflite.py` fails with "No module named 'experiments'"):
    python -m experiments.export_tflite --domain dc_motor --tier rpi5 --block classifier
    python -m experiments.export_tflite --domain vsc_dpc --tier esp32 --block regressor --dataset data/vsc_dpc_dataset.parquet
    python -m experiments.export_tflite --domain vsc_dpc --tier rpi5 --block regressor --run-dir configs/regressors/rpi5_tcn/<run>
"""

import argparse
from pathlib import Path

import numpy as np

from driveflow.ai.registry import load_promoted_model, load_run_model, resolve
from driveflow.ai.tflite_export import export_float16, export_int8
from driveflow.models.common import build_classification_windows, build_direct_forecast_windows
from experiments.train_model import _load_domain_dataframe


def _representative_windows(domain: str, block: str, dataset_path: Path, config, metrics: dict, n: int, noise_free: bool = False):
    df = _load_domain_dataframe(dataset_path, domain)
    if noise_free and {"grid_electrical_noise_pct", "grid_mechanical_noise_pct"} <= set(df.columns):
        # A2 follow-up: calibrate the int8 ranges on the grid's noise-free runs only, so the
        # activation ranges are not set by the up-to-100 % noise windows.
        df = df[(df["grid_electrical_noise_pct"] == 0) & (df["grid_mechanical_noise_pct"] == 0)]
    channels = metrics["channels"]
    if block == "classifier":
        X, _, _ = build_classification_windows(df, metrics["classes"], channels, config.input_window, cap_per_class=10_000, level_scaling=metrics.get("level_scaling"))
    else:
        X, _, _ = build_direct_forecast_windows(df, channels, config.input_window, config.horizon)
    if X is None:
        raise ValueError(f"could not build any representative windows from {dataset_path} for {domain}/{block}")
    # A seeded random subset: the builder returns windows grouped by class, so the first n would
    # calibrate the int8 ranges on one class only.
    return X[np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--tier", required=True, choices=["rpi5", "esp32"], help="pc has no TFLite export -- it runs the native keras.Model. The C2000 has no TFLite either: see experiments/export_c2000.py.")
    parser.add_argument("--block", required=True, choices=["classifier", "regressor"])
    parser.add_argument("--dataset", type=Path, help="Required for --tier esp32 (int8 calibration data); ignored for rpi5.")
    parser.add_argument("--n-calibration-samples", type=int, default=1000)
    parser.add_argument("--calibration-noise-free", action="store_true", help="esp32: calibrate int8 ranges on the dataset's noise-free grid runs only")
    parser.add_argument("--run-dir", type=Path, default=None, help="export this run instead of the promoted one (e.g. a candidate awaiting promotion)")
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        run_dir = args.run_dir
        model, config, metrics = load_run_model(run_dir, args.block)
    else:
        model, config, metrics = load_promoted_model(args.domain, args.tier, args.block)
        run_dir = resolve(args.domain, args.tier, args.block)
    out_path = run_dir / "model.tflite"

    if args.tier == "rpi5":
        export_float16(model, out_path)
        print(f"Exported float16 TFLite: {out_path}")
    else:
        if args.dataset is None:
            raise SystemExit("--dataset is required for --tier esp32 (int8 calibration needs windows sampled from the training dataset).")
        windows = _representative_windows(args.domain, args.block, args.dataset, config, metrics, args.n_calibration_samples, noise_free=args.calibration_noise_free)
        export_int8(model, windows, out_path, n_calibration_samples=len(windows))
        print(f"Exported int8 TFLite: {out_path}")


if __name__ == "__main__":
    main()
