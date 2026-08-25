"""Continues fine-tuning configs/dpc_trained_v2_closed_loop.weights.h5 for more epochs, on the
SAME augmented dataset finetune_dpc_closed_loop.py built (loss hadn't flattened yet at 300 epochs
-- see docs/macro_fase_B2_dpc_deployment.md). Rebuilds the identical 18000-row set deterministically
(closed-loop rollout collection from the v1 controller has no randomness) rather than saving/
reloading it, to avoid a second data artifact that has to stay in sync with the original.

Usage:
    python experiments/continue_finetune_dpc.py \\
        --data-path /path/to/DPC4PowerElectronics/Data4train.mat \\
        --epochs 1500
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # running this file directly puts experiments/ (not the repo
# root) on sys.path, so "experiments.finetune_dpc_closed_loop" wouldn't otherwise be importable.

import numpy as np
import tensorflow as tf

from driveflow.control.dpc import build_dpc_network, load_training_data, train_on_array
from driveflow.control.dpc.network import N_INPUTS
from experiments.finetune_dpc_closed_loop import collect_closed_loop_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--v1-weights", type=Path, default=REPO_ROOT / "configs" / "dpc_trained.weights.h5", help="used only to rebuild the identical augmented dataset")
    parser.add_argument("--current-weights", type=Path, default=REPO_ROOT / "configs" / "dpc_trained_v2_closed_loop.weights.h5")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "configs" / "dpc_trained_v3_closed_loop.weights.h5")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=1)  # different seed -> different minibatch shuffling than the v1->v2 run
    args = parser.parse_args()

    print("Rebuilding the same augmented dataset (deterministic from v1's rollouts)...")
    closed_loop_rows = collect_closed_loop_rows(args.v1_weights, n_rollouts=8, steps_per_rollout=1500)
    x_train_original, _ = load_training_data(args.data_path, num_train=6000)
    x_augmented = np.concatenate([x_train_original, closed_loop_rows], axis=0)
    print(f"  augmented set: {x_augmented.shape[0]} rows\n")

    model = build_dpc_network()
    model(tf.zeros((1, N_INPUTS)))
    model.load_weights(args.current_weights)

    print(f"Continuing fine-tuning from {args.current_weights} for {args.epochs} more epochs...")
    model, history = train_on_array(
        x_augmented,
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print(f"\nfirst-epoch loss: {history[0]:.3f}")
    print(f"last-epoch loss:  {history[-1]:.3f}")
    print(f"min loss:         {min(history):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output)
    print(f"\nsaved weights to {args.output}")


if __name__ == "__main__":
    main()
