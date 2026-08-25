"""Fine-tunes the DPC network on an AUGMENTED dataset: Data4train.mat's original i.i.d. snapshots
PLUS states actually visited by the current controller in real closed-loop rollouts.

Why (see docs/macro_fase_B2_dpc_deployment.md for the full finding): the network trained purely on
Data4train.mat scores near-perfectly on that data's own holdout (R^2~1.000) but shows a
~17-degree steady-state phase lag and ~33% RMSE when deployed in an actual closed loop against
VscSystem. Data4train.mat's rows are i.i.d. random (state, reference) snapshots -- the network
never saw, during training, the states its OWN actions actually lead to over a real trajectory.
This is a standard train/deploy distribution-shift problem for offline-trained control policies;
the standard fix (DAgger-style data aggregation, adapted here to DPC's self-supervised loss
instead of imitation-learning labels -- no expert labels are needed either way, since the loss is
model-based, not a regression against a target action) is: roll the CURRENT policy out in the
real closed loop, record the states it actually visits, add those as new training rows (with their
correct, analytically-known reference), and continue training (fine-tune, not from-scratch) on the
combined set.

Usage:
    python experiments/finetune_dpc_closed_loop.py \\
        --data-path /path/to/DPC4PowerElectronics/Data4train.mat \\
        --base-weights configs/dpc_trained.weights.h5 \\
        --output configs/dpc_trained_v2_closed_loop.weights.h5
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from driveflow.control.dpc import build_dpc_network, load_training_data, train_on_array
from driveflow.control.dpc.controller import DpcController
from driveflow.control.dpc.network import N_INPUTS
from driveflow.control.dpc.reference import RotatingReference
from driveflow.sim.vsc_system import VscSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "dpc_trained_v2_closed_loop.weights.h5"
R_OHM = 8.0064  # matches Data4train.mat's holdout -- see docs/macro_fase_B1_dpc.md


def collect_closed_loop_rows(weights_path: Path, n_rollouts: int = 8, steps_per_rollout: int = 1500, tau: float = 1e-4) -> np.ndarray:
    """Runs the CURRENT controller in closed loop from several different reference starting
    phases (0, pi/4, pi/2, ... -- the only source of rollout-to-rollout diversity available,
    since R is fixed at the one value we have real data for), and records every visited
    (state, reference) pair in Data4train.mat's exact 15-column row format -- the reference at
    each future horizon step is computed analytically (RotatingReference is exogenous/known, not
    forecast), so no extra measurement is needed to build a valid training row from a rollout.
    """
    controller = DpcController(weights_path, r_ohm=R_OHM)
    rows = []
    for i in range(n_rollouts):
        phase0 = 2 * np.pi * i / n_rollouts
        system = VscSystem(load_resistance_ohm=R_OHM, tau=tau)
        reference = RotatingReference(tau=tau, phase0_rad=phase0)
        state = system.reset()
        controller.reset()
        for k in range(steps_per_rollout):
            v_o = controller.control(state, reference, k)
            horizon_refs = reference.horizon(k, horizon=5)
            row = [state.i_f_real, state.i_f_imag, state.vc_real, state.vc_imag, horizon_refs[0][0], horizon_refs[0][1], R_OHM]
            for vr, vi in horizon_refs[1:]:
                row += [vr, vi]
            rows.append(row)
            state = system.simulate(*v_o)
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, default=REPO_ROOT / "configs" / "dpc_trained.weights.h5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--steps-per-rollout", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="lower than train_dpc.py's default -- fine-tuning, not training from scratch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Collecting closed-loop rollout data from {args.n_rollouts} rollouts x {args.steps_per_rollout} steps...")
    closed_loop_rows = collect_closed_loop_rows(args.base_weights, n_rollouts=args.n_rollouts, steps_per_rollout=args.steps_per_rollout)
    print(f"  collected {closed_loop_rows.shape[0]} closed-loop-visited rows")

    x_train_original, _ = load_training_data(args.data_path, num_train=6000)
    x_augmented = np.concatenate([x_train_original, closed_loop_rows], axis=0)
    print(f"  original Data4train.mat training rows: {x_train_original.shape[0]}")
    print(f"  augmented training set: {x_augmented.shape[0]} rows\n")

    model = build_dpc_network()
    model(tf.zeros((1, N_INPUTS)))
    model.load_weights(args.base_weights)

    print("Fine-tuning on the augmented dataset...")
    model, history = train_on_array(
        x_augmented,
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print(f"\nfirst-epoch loss: {history[0]:.3f}")
    print(f"last-epoch loss:  {history[-1]:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output)
    print(f"\nsaved fine-tuned weights to {args.output}")


if __name__ == "__main__":
    main()
