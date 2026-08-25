"""Evaluates the trained DPC network (configs/dpc_trained.weights.h5) on the holdout split of
Data4train.mat, with regression/control metrics -- NOT classification metrics.

Why not F1-score / precision / recall: those are for a model that outputs discrete class labels.
This network outputs a continuous converter voltage command (v_o, 10 real numbers); "correctness"
is how close the resulting simulated voltage/current land to their continuous targets, not a
predicted-class-vs-true-class table. The regression analogues used here (RMSE, MAE, R^2, and a
tolerance-band "success rate" as the closest thing to a single interpretable percentage) are what
actually describe this model's performance -- see the printed rationale for each metric below.

Usage:
    python experiments/evaluate_dpc.py \\
        --data-path /path/to/DPC4PowerElectronics/Data4train.mat \\
        --weights configs/dpc_trained.weights.h5
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from driveflow.control.dpc import HORIZON, build_dpc_network, load_training_data, simulate_horizon

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "configs" / "dpc_trained.weights.h5"
#: Tolerance for the "tracking success" rate: 5% of Data4train.mat's actual reference voltage
#: magnitude. Checked directly on the holdout split: |v_ref| = sqrt(vref_alpha^2+vref_beta^2) is
#: 50.00 V for every single sample (std ~1e-6, i.e. a rotating vector of fixed magnitude, not the
#: wider uniform range an earlier abandoned draft (DPC4VSC.py) sampled from -- see
#: docs/macro_fase_B1_dpc.md). 325V would have been the wrong scale for THIS dataset.
VOLTAGE_MAGNITUDE_V = 50.0
VOLTAGE_TOLERANCE_V = 0.05 * VOLTAGE_MAGNITUDE_V


def _metrics(pred: np.ndarray, ref: np.ndarray, scale: float) -> dict:
    """pred/ref: pooled per-*component* arrays (real-axis values concatenated with imag-axis
    values), NOT vector magnitudes. Data4train.mat's |v_ref| is a constant 50.0V for every
    sample (a fixed-magnitude rotating reference vector -- checked directly, see
    VOLTAGE_MAGNITUDE_V's comment), so computing R2/NRMSE against the *magnitude* divides by a
    near-zero variance and blows up to nonsense. The individual alpha/beta *components* swing
    across the full +-50V range (std ~35V) as the vector rotates, so R2/NRMSE computed per
    component is well-posed.
    """
    error = pred - ref
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    nrmse_pct = 100.0 * rmse / scale
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "nrmse_pct": nrmse_pct, "r2": r2}


def evaluate(data_path: Path, weights_path: Path, num_train: int = 6000):
    _, x_holdout = load_training_data(data_path, num_train=num_train)
    x_holdout_tf = tf.constant(x_holdout)

    model = build_dpc_network()
    model(tf.zeros((1, x_holdout.shape[1])))  # build before loading weights
    model.load_weights(weights_path)

    v_o = model(x_holdout_tf, training=False)
    sim = simulate_horizon(x_holdout_tf, v_o)

    v_o_zero = tf.zeros_like(v_o)
    sim_baseline = simulate_horizon(x_holdout_tf, v_o_zero)

    v_ref_real = sim["v_ref_real"].numpy()
    v_ref_imag = sim["v_ref_imag"].numpy()

    print(f"Holdout size: {x_holdout.shape[0]} samples (rows {num_train}:10000 of Data4train.mat, never trained on)\n")

    print("=" * 78)
    print("VOLTAGE TRACKING  (predicted capacitor voltage vs. reference, per horizon step)")
    print("=" * 78)
    print(f"{'step':>5} {'RMSE (V)':>10} {'MAE (V)':>10} {'NRMSE %':>9} {'R2':>7}   {'baseline RMSE (V)':>18}")
    for k in range(HORIZON):
        pred = np.concatenate([sim["vc_real"][k].numpy(), sim["vc_imag"][k].numpy()])
        ref = np.concatenate([v_ref_real[k], v_ref_imag[k]])
        m = _metrics(pred, ref, scale=VOLTAGE_MAGNITUDE_V)
        pred_base = np.concatenate([sim_baseline["vc_real"][k].numpy(), sim_baseline["vc_imag"][k].numpy()])
        rmse_base = float(np.sqrt(np.mean((pred_base - ref) ** 2)))
        print(f"{k + 1:>5} {m['rmse']:>10.2f} {m['mae']:>10.2f} {m['nrmse_pct']:>8.1f}% {m['r2']:>7.3f}   {rmse_base:>18.2f}")

    all_err = np.sqrt((v_ref_real - sim["vc_real"].numpy()) ** 2 + (v_ref_imag - sim["vc_imag"].numpy()) ** 2)
    success_rate = 100.0 * float(np.mean(all_err < VOLTAGE_TOLERANCE_V))
    print(
        f"\nTracking 'success rate' (|error vector| < {VOLTAGE_TOLERANCE_V:.2f} V, i.e. 5% of the "
        f"reference's {VOLTAGE_MAGNITUDE_V:.0f}V magnitude) across all steps/samples: {success_rate:.1f}%"
    )

    print()
    print("=" * 78)
    print("CURRENT CONSISTENCY  (predicted filter current vs. load + capacitive KCL term)")
    print("=" * 78)
    print(f"{'step':>5} {'RMSE (A)':>10} {'MAE (A)':>10} {'NRMSE %':>9} {'R2':>7}   {'baseline RMSE (A)':>18}")
    for k in range(HORIZON):
        c_wref_vrefimag = 15e-6 * 2 * np.pi * 50.0 * v_ref_imag[k]
        c_wref_vrefreal = 15e-6 * 2 * np.pi * 50.0 * v_ref_real[k]
        target_real = sim["i_load_real"].numpy() - c_wref_vrefimag
        target_imag = sim["i_load_imag"].numpy() + c_wref_vrefreal
        pred = np.concatenate([sim["ift_real"][k].numpy(), sim["ift_imag"][k].numpy()])
        target = np.concatenate([target_real, target_imag])
        # Scale computed from the data itself (not hardcoded like VOLTAGE_MAGNITUDE_V): i_load's
        # magnitude here happens to be near-constant too (R is fixed at ~8.0 ohm across this
        # holdout), but deriving the scale from target's own RMS keeps this correct even if a
        # future holdout has varying R.
        scale = float(np.sqrt(np.mean(target**2))) or 1.0
        m = _metrics(pred, target, scale=scale)
        pred_base = np.concatenate([sim_baseline["ift_real"][k].numpy(), sim_baseline["ift_imag"][k].numpy()])
        rmse_base = float(np.sqrt(np.mean((pred_base - target) ** 2)))
        print(f"{k + 1:>5} {m['rmse']:>10.2f} {m['mae']:>10.2f} {m['nrmse_pct']:>8.1f}% {m['r2']:>7.3f}   {rmse_base:>18.2f}")

    print()
    print("Note: F1-score/precision/recall don't apply -- this is a continuous control-action")
    print("regression, not a classifier. RMSE/MAE/R2/NRMSE + a tolerance-band success rate are")
    print("the analogous metrics for 'how good is this model' here.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--num-train", type=int, default=6000)
    args = parser.parse_args()
    evaluate(args.data_path, args.weights, num_train=args.num_train)


if __name__ == "__main__":
    main()
