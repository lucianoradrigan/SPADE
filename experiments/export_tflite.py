"""Exports a promoted (domain, tier, block) run to TFLite -- float16 for rpi5, int8 (calibrated
against real windows rebuilt from --dataset) for esp32 -- docs/design_ai_layer_transversal.md
Sec. 4.1/4.2, Sec. 8 step 8. Saves alongside the run's existing artifacts as model.tflite.

Usage (run as a module -- the script imports experiments.train_model, so a direct
`python experiments/export_tflite.py` fails with "No module named 'experiments'"):
    python -m experiments.export_tflite --domain dc_motor --tier rpi5 --block classifier
    python -m experiments.export_tflite --domain vsc_dpc --tier esp32 --block regressor --dataset data/vsc_dpc_dataset.parquet
"""

import argparse
from pathlib import Path

from driveflow.ai.registry import load_promoted_model, resolve
from driveflow.ai.tflite_export import export_float16, export_int8
from driveflow.models.common import build_classification_windows, build_direct_forecast_windows
from experiments.train_model import _load_domain_dataframe


def _representative_windows(domain: str, block: str, dataset_path: Path, config, metrics: dict, n: int):
    df = _load_domain_dataframe(dataset_path, domain)
    channels = metrics["channels"]
    if block == "classifier":
        X, _, _ = build_classification_windows(df, metrics["classes"], channels, config.input_window)
    else:
        X, _, _ = build_direct_forecast_windows(df, channels, config.input_window, config.horizon)
    if X is None:
        raise ValueError(f"could not build any representative windows from {dataset_path} for {domain}/{block}")
    return X[:n]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--tier", required=True, choices=["rpi5", "esp32"], help="pc has no TFLite export -- it runs the native keras.Model.")
    parser.add_argument("--block", required=True, choices=["classifier", "regressor"])
    parser.add_argument("--dataset", type=Path, help="Required for --tier esp32 (int8 calibration data); ignored for rpi5.")
    parser.add_argument("--n-calibration-samples", type=int, default=100)
    args = parser.parse_args(argv)

    model, config, metrics = load_promoted_model(args.domain, args.tier, args.block)
    run_dir = resolve(args.domain, args.tier, args.block)
    out_path = run_dir / "model.tflite"

    if args.tier == "rpi5":
        export_float16(model, out_path)
        print(f"Exported float16 TFLite: {out_path}")
    else:
        if args.dataset is None:
            raise SystemExit("--dataset is required for --tier esp32 (int8 calibration needs real windows).")
        windows = _representative_windows(args.domain, args.block, args.dataset, config, metrics, args.n_calibration_samples)
        export_int8(model, windows, out_path)
        print(f"Exported int8 TFLite: {out_path}")


if __name__ == "__main__":
    main()
