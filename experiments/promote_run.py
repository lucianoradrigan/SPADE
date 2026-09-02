"""Promotes an experiments/train_model.py run directory to production for its (domain, tier,
block) -- docs/design_ai_layer_transversal.md Sec. 7, Sec. 8 step 6. A deliberate, separate step
from training itself (see driveflow.ai.registry's module docstring).

Usage:
    python experiments/promote_run.py configs/classifiers/pc_server/2026-09-01_run01
"""

import argparse
from pathlib import Path

from driveflow.ai.registry import promote


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    entry = promote(args.run_dir)
    print(f"Promoted {entry.run_dir} -> {entry.domain}/{entry.tier}/{entry.block}")


if __name__ == "__main__":
    main()
