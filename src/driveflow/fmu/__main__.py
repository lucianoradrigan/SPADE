"""python -m driveflow.fmu fetch [--release TAG] [--model NAME] -- disabled (see driveflow.fmu)."""

from __future__ import annotations

import argparse
import sys

from driveflow.fmu import FETCH_DISABLED


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m driveflow.fmu", description="OpenModelica FMUs of SPADE.")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download prebuilt FMUs (disabled: licence review pending)")
    f.add_argument("--release", help="Release tag, e.g. fmu-v1")
    f.add_argument("--model", help="model name, e.g. BuckAveraged")
    ap.parse_args(argv)
    print(FETCH_DISABLED, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
