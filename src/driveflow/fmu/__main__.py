"""python -m driveflow.fmu fetch [--release TAG] [--model NAME] [--allow-mismatch] (see driveflow.fmu)."""

from __future__ import annotations

import argparse
import sys

from driveflow.fmu import FetchError, fetch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m driveflow.fmu", description="Prebuilt OpenModelica FMUs of SPADE.")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download prebuilt FMUs from a GitHub Release, verifying SHA-256")
    f.add_argument("--release", help="Release tag, e.g. fmu-v1 (default: the latest published release)")
    f.add_argument("--model", help="model name, e.g. BuckAveraged (default: every FMU of the release for this platform)")
    f.add_argument("--allow-mismatch", action="store_true", help="install even if the release was built from another package.mo (recorded)")
    f.add_argument("--dest", help="target folder (default: data/fmu or $SPADE_FMU_DIR)")
    args = ap.parse_args(argv)
    try:
        recs = fetch(args.release, args.model, args.allow_mismatch, args.dest)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if all(r["status"] == "installed" for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
