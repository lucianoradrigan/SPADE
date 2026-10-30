"""Exports a run to a C kit for the TI C2000 LAUNCHXL-F28379D, written to <run>/c2000/ (see
driveflow/ai/c2000_export.py). The C2000 is an export target, not a training tier: the source is
the promoted run of --tier, or --run-dir.

Usage (run as a module, same reason as experiments/export_tflite.py):
    python -m experiments.export_c2000 --domain dc_motor --tier esp32 --block classifier
    python -m experiments.export_c2000 --domain vsc_dpc --tier esp32 --block regressor
    python -m experiments.export_c2000 --domain vsc_dpc --block regressor --run-dir configs/regressors/rpi5_tcn/<run>
    python -m experiments.export_c2000 --domain dc_motor --tier rpi5 --block classifier --check

--check only reports whether the model fits; it writes nothing.
"""

import argparse
from pathlib import Path

from driveflow.ai.c2000_export import EXPORT_DIRNAME, C2000ExportError, check_fits, export_c2000, host_parity, plan_model, simulated_raw_windows
from driveflow.ai.registry import load_promoted_model, load_run_model, resolve


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--tier", choices=["esp32", "rpi5", "pc"], default="esp32", help="whose promoted run to export")
    parser.add_argument("--block", required=True, choices=["classifier", "regressor"])
    parser.add_argument("--run-dir", type=Path, default=None, help="export this run instead of the promoted one")
    parser.add_argument("--reference-windows", type=int, default=6)
    parser.add_argument("--check", action="store_true", help="report the memory verdict and exit")
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        run_dir = args.run_dir
        model, _, metrics = load_run_model(run_dir, args.block)
    else:
        model, _, metrics = load_promoted_model(args.domain, args.tier, args.block)
        run_dir = resolve(args.domain, args.tier, args.block)

    try:
        plan = plan_model(model)
        if args.check:
            print(f"{run_dir}: {check_fits(plan)}")
            return
        raw = simulated_raw_windows(args.domain, metrics["channels"], plan.input_shape[0], args.reference_windows)
        manifest = export_c2000(model, metrics, raw, run_dir / EXPORT_DIRNAME, source_run=f"{Path(run_dir).parent.name}/{Path(run_dir).name}")
    except C2000ExportError as exc:
        raise SystemExit(f"{run_dir}: {exc}") from exc
    print(f"Exported C kit: {run_dir / EXPORT_DIRNAME}  memory {manifest['memory']}")
    print(f"Host parity (gcc, on this PC, not the board): {host_parity(run_dir / EXPORT_DIRNAME)}")


if __name__ == "__main__":
    main()
