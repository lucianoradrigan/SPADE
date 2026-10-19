"""Fine-tunes the DPC network over a RANGE of load resistances and reference magnitudes
(docs/SIMULATION_STABILITY_REPORT.md S-05).

v3 (configs/dpc_trained_v3_closed_loop.weights.h5) was fine-tuned on closed-loop rollouts at ONE
load, R = 8.0064 ohm, and Data4train.mat holds only R = 8.0064 and 40 ohm. The network takes R as an
input, but it tracks well only near 8 ohm: settled RMSE 18 V at 4 ohm, 15 V at 20 ohm, 111 V at
1 kohm, against 1.18 V at 8 ohm.

Same DAgger-style recipe as experiments/finetune_dpc_closed_loop.py, generalized: each round rolls
the CURRENT controller out in closed loop at every (R, |v_ref|, start phase) of the grid below,
records the visited (state, reference) rows -- the reference horizon is known analytically, and the
loss is model-based, so no labels are needed -- adds them to the training set and continues
training. Data4train.mat's rows stay in the set so the original operating points are not forgotten.

R_TRAIN starts at 3.5 ohm, just above the plant's own open-loop stability boundary
R* = 3.3707 ohm (below it the plant diverges with any controller, S-08). The reference frequency
stays at the grid's 50 Hz: dpc_loss models the capacitor current with that fixed frequency.

Held-out evaluation loads (R_EVAL) include values between and beyond the training ones.

Usage:
    python experiments/finetune_dpc_r_range.py --data-path ../DPC4PowerElectronics/Data4train.mat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from driveflow.ai.provenance import file_sha256, run_metadata  # noqa: E402
from driveflow.control.dpc import build_dpc_network, load_training_data, train_on_array  # noqa: E402
from driveflow.control.dpc.network import N_INPUTS  # noqa: E402
from driveflow.control.dpc.reference import RotatingReference  # noqa: E402
from driveflow.sim.vsc_system import VscSystem  # noqa: E402

R_TRAIN = (3.5, 4.0, 5.0, 6.5, 8.0064, 11.0, 15.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0)
MAGNITUDES_TRAIN = (25.0, 50.0, 75.0, 100.0)
R_EVAL = (3.5, 4.0, 4.5, 6.0, 8.0064, 9.0, 13.0, 20.0, 25.0, 70.0, 200.0, 1000.0, 5000.0)
MAGNITUDES_EVAL = (50.0, 35.0, 120.0, 10.0, 150.0)
N_PHASES = 2
STEPS_PER_ROLLOUT = 1200
EVAL_STEPS = 2000
WARMUP = 50
TAU = 1e-4
#: A rollout state beyond this magnitude is not a useful training row (the controller has lost
#: the plant); rows are dropped from that point on.
STATE_LIMIT = 1e3


def _row(state, refs, r_ohm: float) -> list:
    row = [state.i_f_real, state.i_f_imag, state.vc_real, state.vc_imag, refs[0][0], refs[0][1], r_ohm]
    for vr, vi in refs[1:]:
        row += [vr, vi]
    return row


def batched_closed_loop(model, cases: list, n_steps: int) -> tuple:
    """Runs every (r_ohm, magnitude_v, phase0) case in closed loop at once: one network call per
    step for the whole batch (the same receding-horizon rule as DpcController.control: apply only
    the first predicted action). Returns (rows (n_cases, n_steps, N_INPUTS), error (n_cases, n_steps))
    with error[k] = |v_c - v_ref(k)| after step k, as experiments/evaluate_dpc_closed_loop.py
    measures it. A case whose state leaves STATE_LIMIT is frozen (rows/errors NaN from there)."""
    systems = [VscSystem(load_resistance_ohm=r, tau=TAU) for r, _m, _p in cases]
    refs = [RotatingReference(tau=TAU, phase0_rad=p, magnitude_v=m) for _r, m, p in cases]
    states = [s.reset() for s in systems]
    alive = np.ones(len(cases), dtype=bool)
    rows = np.full((len(cases), n_steps, N_INPUTS), np.nan, dtype=np.float32)
    err = np.full((len(cases), n_steps), np.nan)
    for k in range(n_steps):
        x = np.array([_row(states[i], refs[i].horizon(k, horizon=5), cases[i][0]) for i in range(len(cases))], dtype=np.float32)
        v_o = model(tf.constant(x), training=False).numpy()
        for i in np.flatnonzero(alive):
            st = (states[i].i_f_real, states[i].i_f_imag, states[i].vc_real, states[i].vc_imag)
            if not np.all(np.isfinite(st)) or max(abs(v) for v in st) > STATE_LIMIT:
                alive[i] = False
                continue
            rows[i, k] = x[i]
            states[i] = systems[i].simulate(float(v_o[i, 0]), float(v_o[i, 1]))
            vr, vi = refs[i].at_step(k)
            err[i, k] = np.hypot(states[i].vc_real - vr, states[i].vc_imag - vi)
    return rows, err


def collect(model, magnitudes=MAGNITUDES_TRAIN) -> np.ndarray:
    cases = [(r, m, 2 * np.pi * i / N_PHASES) for r in R_TRAIN for m in magnitudes for i in range(N_PHASES)]
    rows, _ = batched_closed_loop(model, cases, STEPS_PER_ROLLOUT)
    rows = rows.reshape(-1, N_INPUTS)
    return rows[np.all(np.isfinite(rows), axis=1)]


def rmse_table(model) -> dict:
    cases = [(r, m, 0.0) for m in MAGNITUDES_EVAL for r in R_EVAL]
    _, err = batched_closed_loop(model, cases, EVAL_STEPS)
    out = {}
    for (r, m, _p), e in zip(cases, err):
        out.setdefault(f"{m:g}V", {})[f"{r:g}"] = float(np.sqrt(np.mean(e[WARMUP:] ** 2))) if np.all(np.isfinite(e)) else None
    return out


def _load(weights_path: Path):
    model = build_dpc_network()
    model(tf.zeros((1, N_INPUTS)))
    model.load_weights(weights_path)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", type=Path, required=True)
    ap.add_argument("--base-weights", type=Path, default=REPO_ROOT / "configs" / "dpc_trained_v3_closed_loop.weights.h5")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "configs" / "dpc_trained_v4_r_range.weights.h5")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150, help="epochs per round")
    ap.add_argument("--learning-rate", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=list(MAGNITUDES_TRAIN), help="reference magnitudes of the rollouts (V)")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs" / "audit" / "stability" / "dpc_r_range_finetune.json")
    args = ap.parse_args(argv)
    args.base_weights, args.output, args.report = args.base_weights.resolve(), args.output.resolve(), args.report.resolve()

    t0 = time.time()
    x_orig, _ = load_training_data(args.data_path, num_train=6000)
    report = {"command": "python experiments/finetune_dpc_r_range.py " + " ".join(argv or sys.argv[1:]), "r_train": R_TRAIN, "magnitudes_train": tuple(args.magnitudes),
              "r_eval": R_EVAL, "magnitudes_eval": MAGNITUDES_EVAL, "base_weights": str(args.base_weights.relative_to(REPO_ROOT)), "rounds": []}
    print("baseline RMSE table ...", flush=True)
    model = _load(args.base_weights)
    report["baseline_rmse_V"] = rmse_table(model)
    print(json.dumps(report["baseline_rmse_V"]["50V"]), flush=True)
    aggregated = [x_orig]
    for rnd in range(args.rounds):
        rows = collect(model, tuple(args.magnitudes))
        aggregated.append(rows)
        x = np.concatenate(aggregated, axis=0)
        print(f"round {rnd + 1}: +{len(rows)} rollout rows, training on {len(x)} rows", flush=True)
        model, history = train_on_array(x, model=model, epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed + rnd, verbose=False)
        table = rmse_table(model)
        report["rounds"].append({"round": rnd + 1, "n_rows": int(len(x)), "loss_first": history[0], "loss_last": history[-1], "rmse_V": table})
        print(json.dumps(table["50V"]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output)
    report.update({"output": str(args.output.relative_to(REPO_ROOT)), "output_sha256": file_sha256(args.output), "wall_time_s": round(time.time() - t0, 1), "metadata": run_metadata()})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {args.output}; report {args.report}")


if __name__ == "__main__":
    main()
