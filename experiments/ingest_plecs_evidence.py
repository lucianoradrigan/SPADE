"""Ingest a PLECS researcher-kit evidence package (task A2).

    python experiments/ingest_plecs_evidence.py results/plecs_evidence_<UTC>.zip

Checks every file's SHA-256 against hashes.json and evidence.json's schema, then copies the evidence to
docs/audit/plecs/<package>/ and the sweep dataset to data/plecs/<profile>/<run_id>/. Refuses (and
copies nothing) on any mismatch or if the package was already ingested.
"""

import argparse
import json
import sys
from pathlib import Path

from driveflow.integrations.plecs.kit import KitError, ingest

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--docs-root", type=Path, default=ROOT / "docs" / "audit" / "plecs")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "plecs")
    args = ap.parse_args(argv)
    try:
        res = ingest(args.package, args.docs_root, args.data_root)
    except KitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
