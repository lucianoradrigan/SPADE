"""PLECS researcher kit -- the single command (task A2; guide: docs/guides/PLECS_RESEARCHER_KIT.md).

    python experiments/plecs_researcher_run.py --circuit my_buck.plecs \
        --manifest configs/plecs/manifest_templates/buck_example.yaml --out results/

Needs PLECS Standalone running on THIS computer with its RPC server enabled
(Preferences → General → RPC interface, port 1080). Writes results/plecs_evidence_<UTC>.zip, also when
a step fails (the cause is inside). Send that zip back; nothing else is needed.
"""

import argparse
import sys

from driveflow.integrations.plecs.kit import run_kit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", required=True, help="the .plecs file")
    ap.add_argument("--manifest", required=True, help="the kit template (YAML with 'manifest' and 'kit' blocks)")
    ap.add_argument("--out", required=True, help="folder for the evidence package")
    ap.add_argument("--rpc-url", default="http://localhost:1080", help="PLECS RPC endpoint on this computer (default http://localhost:1080)")
    args = ap.parse_args(argv)
    cmd = " ".join(["python experiments/plecs_researcher_run.py", *sys.argv[1:]]) if argv is None else " ".join(["python experiments/plecs_researcher_run.py", *argv])
    z = run_kit(args.circuit, args.manifest, args.out, args.rpc_url, command=cmd)
    print(f"\nEvidence package: {z}\nSend this file back. If a step failed, the cause is in evidence.json and log.txt inside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
