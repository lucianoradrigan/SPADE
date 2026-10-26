"""Downloads the Modelica Association's Reference FMUs (BSD-2-Clause), pinned by SHA-256, and extracts
the FMUs SPADE's tests use into data/fmu/reference/{2.0,3.0}/ (task P1b: FMI 3.0 with real binaries;
OpenModelica 1.27.1 does not export FMI 3.0). Not committed (data/ is git-ignored).

Usage: python scripts/fmu/fetch_reference_fmus.py [--dest data/fmu/reference]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.0.42"
URL = f"https://github.com/modelica/Reference-FMUs/releases/download/v{VERSION}/Reference-FMUs.zip"
SHA256 = "b6b4a195b9c0f6bdd3459cd949cbb35fc21ee2049d6dab4cb7a3ba7da63b1e97"
MEMBERS = ("LICENSE.txt", "3.0/Dahlquist.fmu", "3.0/Stair.fmu", "3.0/VanDerPol.fmu", "3.0/BouncingBall.fmu", "3.0/Feedthrough.fmu",
           "2.0/Dahlquist.fmu", "2.0/Stair.fmu")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "fmu" / "reference")
    args = ap.parse_args(argv)
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        print(f"SHA-256 mismatch: expected {SHA256}, got {got}; nothing extracted", file=sys.stderr)
        return 2
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in MEMBERS:
            out = args.dest / m
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(m))
    (args.dest / "SOURCE.txt").write_text(f"Modelica Association Reference FMUs v{VERSION}\n{URL}\nSHA-256 {SHA256}\nLicence: BSD-2-Clause (LICENSE.txt)\n")
    print(f"extracted {len(MEMBERS)} files to {args.dest} (Reference FMUs v{VERSION}, SHA-256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
