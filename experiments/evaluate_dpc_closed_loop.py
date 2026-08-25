"""Closed-loop validation of the trained DPC controller (Macro-fase B.2), on the actual VscSystem
plant -- NOT the static per-sample holdout metrics in evaluate_dpc.py.

Why this is a separate, necessary check: Data4train.mat's rows are i.i.d. random (state,
reference) snapshots -- the network never saw its own actions shape the next state during
training. Good holdout metrics on that data say the network can predict a good v_o *for a given
snapshot*; they say nothing about whether repeatedly applying only its first predicted step, over
hundreds of real consecutive steps, is a *stable* closed loop (errors could compound, or the
receding-horizon deployment could behave differently from the training distribution it was fit
against). This script runs that real loop and reports whether tracking converges and stays
converged.

Usage:
    python experiments/evaluate_dpc_closed_loop.py --weights configs/dpc_trained.weights.h5
"""

import argparse
from pathlib import Path

import numpy as np

from driveflow.control.dpc.controller import DpcController
from driveflow.control.dpc.reference import RotatingReference
from driveflow.sim.vsc_system import VscSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "configs" / "dpc_trained.weights.h5"


def run_closed_loop(weights_path: Path, r_ohm: float = 8.0064, n_steps: int = 2000, tau: float = 1e-4):
    system = VscSystem(load_resistance_ohm=r_ohm, tau=tau)
    controller = DpcController(weights_path, r_ohm=r_ohm)
    reference = RotatingReference(tau=tau)

    state = system.reset()
    controller.reset()

    records = []
    for k in range(n_steps):
        v_o_real, v_o_imag = controller.control(state, reference, k)
        state = system.simulate(v_o_real, v_o_imag)
        vref_real, vref_imag = reference.at_step(k)
        records.append(
            {
                "t": k * tau,
                "vc_real": state.vc_real,
                "vc_imag": state.vc_imag,
                "vref_real": vref_real,
                "vref_imag": vref_imag,
                "i_f_real": state.i_f_real,
                "i_f_imag": state.i_f_imag,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--r-ohm", type=float, default=8.0064)
    args = parser.parse_args()

    records = run_closed_loop(args.weights, r_ohm=args.r_ohm, n_steps=args.n_steps)

    vc_real = np.array([r["vc_real"] for r in records])
    vc_imag = np.array([r["vc_imag"] for r in records])
    vref_real = np.array([r["vref_real"] for r in records])
    vref_imag = np.array([r["vref_imag"] for r in records])
    err = np.sqrt((vc_real - vref_real) ** 2 + (vc_imag - vref_imag) ** 2)

    n = len(records)
    warmup = min(50, n // 4)  # first ~5ms: startup transient from state=0, excluded from "settled" stats

    print(f"Closed loop: {n} steps ({n * 1e-4 * 1000:.1f} ms simulated), R={args.r_ohm} ohm")
    print(f"\nTransient (first {warmup} steps):")
    print(f"  max |error|:  {err[:warmup].max():.2f} V")
    print(f"  final error of transient window: {err[warmup - 1]:.2f} V")
    print(f"\nSettled (steps {warmup}:{n}):")
    print(f"  RMSE:  {np.sqrt(np.mean(err[warmup:] ** 2)):.3f} V")
    print(f"  MAE:   {np.mean(np.abs(err[warmup:])):.3f} V")
    print(f"  max:   {err[warmup:].max():.3f} V")
    print(f"  (reference magnitude: 50.0 V -- see reference.py)")

    diverging = err[-1] > err[warmup] * 2 and err[-1] > 5.0
    if diverging:
        print("\nWARNING: error grew rather than settling -- closed loop looks unstable, do not trust this checkpoint for B.2.")
    else:
        print("\nNo divergence: error stays bounded/settles across the full run.")


if __name__ == "__main__":
    main()
