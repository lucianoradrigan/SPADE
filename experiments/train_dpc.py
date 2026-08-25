"""Trains the Keras port of DPC4PowerElectronics (Macro-fase B.1).

Usage:
    python experiments/train_dpc.py --data-path /path/to/DPC4PowerElectronics/Data4train.mat \\
        --epochs 200

What this validates, and why (there is no MATLAB in this environment, so this can't be a
byte-for-byte weight comparison against the original toolbox -- see tests/test_dpc_loss.py's
docstring for the translation-fidelity checks that stand in for that):

1. Training loss should decrease and roughly plateau within `--epochs` (a basic "the port isn't
   broken" check -- a mis-wired loss/gradient would plateau immediately near its initial value or
   diverge, not converge).
2. On the held-out rows of Data4train.mat (rows num_train:10000, never seen during training),
   report the same loss decomposition (voltage tracking vs. current consistency) to check the
   network generalizes rather than memorizing the training rows.
3. Saves the trained weights to configs/dpc_trained.weights.h5 for reuse by
   datagen/runner.py's future controller_type="DPC" path (Macro-fase B.2, not yet wired).
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from driveflow.control.dpc import dpc_loss, load_training_data, train_dpc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "dpc_trained.weights.h5"


def evaluate_holdout(model, mat_path: Path, num_train: int) -> float:
    _, x_holdout = load_training_data(mat_path, num_train=num_train)
    x_holdout = tf.constant(x_holdout)
    v_o = model(x_holdout, training=False)
    return float(dpc_loss(x_holdout, v_o))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True, help="Path to DPC4PowerElectronics/Data4train.mat")
    parser.add_argument("--epochs", type=int, default=200, help="Main.txt uses 10000; see train.py's docstring")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-train", type=int, default=6000, help="Main.txt's num=6000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model, history = train_dpc(
        args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train=args.num_train,
        seed=args.seed,
    )

    print()
    print(f"first-epoch loss:  {history[0]:12.3f}")
    print(f"last-epoch loss:   {history[-1]:12.3f}")
    print(f"min loss:          {min(history):12.3f}")

    holdout_loss = evaluate_holdout(model, args.data_path, args.num_train)
    print(f"holdout loss:      {holdout_loss:12.3f}  (rows {args.num_train}:10000, never trained on)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output)
    print(f"\nsaved weights to {args.output}")

    if history[-1] >= history[0]:
        print("\nWARNING: loss did not decrease -- see train.py/loss.py before trusting this checkpoint.")


if __name__ == "__main__":
    main()
