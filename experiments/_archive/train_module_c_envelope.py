"""HISTORICAL / NOT RUNNABLE: residual_model.py (Module C) was deleted after this experiment's own
result (see below) motivated retiring it -- see docs/patch2_retiro_modulo_C.md. Kept only as the
documented record of the investigation that led to that decision; do not try to run this.

Moved out of the active experiments/ tree into experiments/_archive/ in Patch 11
(docs/patch11_archivado_modulo_c.md) -- its "Module C" name was repeatedly confused with the
unrelated "Fase C" (diagnosis classifiers, see INSTRUCTIONS.md Sec. 5); living outside the
active tree makes that mix-up harder for a casual `ls experiments/` to reproduce.

Original docstring follows, describing what it did when it still ran:

Module C, take 2: a windowed-RMS target + a sequence model (GRU), instead of the pointwise MLP
predicting an instantaneous raw residual (train_module_c.py), which failed to learn anything
(holdout RMSE indistinguishable from Module B alone -- see that script's report).

Two changes, both motivated by the same finding:

1. Target is the RMS of the real residual over a short window (5 ms, matching
   paper_federative's own windowing convention -- see docs/propuesta_consolidacion.pdf Sec 2.4's
   envelope_forecaster), not the instantaneous raw value. A pointwise MLP cannot predict the
   phase of an oscillation from slowly-varying omega/torque; predicting the *local energy level*
   of the residual is a much more tractable regression target and is also what the downstream
   classifiers actually consume (windowed features), not raw waveform.
2. The predictor is a small GRU over the window's [omega, torque, vib_b] sequence, not a
   pointwise MLP -- it can use the window's temporal structure (e.g. how much vib_b itself
   fluctuates) as a proxy for the local residual energy, which a single-instant feature vector
   cannot.

Output: a multiplicative RMS-correction factor per window, not an additive per-sample residual --
different contract from residual_model.ResidualCorrector. This script only trains and evaluates
the idea; it does not wire anything into sim/vibration/ yet (see its printed report for whether
that's warranted).
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from driveflow.sim.vibration import calibration as cal
from driveflow.sim.vibration import force_synthesis
from driveflow.sim.vibration.modal_model import Mode, ModalAxis

TRAIN_BEARINGS = ["K002", "K004", "K005", "K006"]
HOLDOUT_BEARINGS = ["K001", "K003"]
CONDITION = "N09_M07_F10"
N_RUNS_PER_BEARING = 6

WINDOW_SAMPLES = 320  # 5 ms at 64 kHz, matches paper_federative's bin size
STRIDE_SAMPLES = 320  # non-overlapping windows


def module_b_prediction(modes, recording):
    axis = ModalAxis([Mode(m.natural_freq_hz, m.damping_ratio, m.gain) for m in modes])
    t_mech = np.arange(len(recording.speed_rpm)) / cal.MECH_FS_HZ
    t_vib = np.arange(len(recording.vibration)) / cal.VIBRATION_FS_HZ
    omega = np.interp(t_vib, t_mech, recording.speed_rpm * 2 * np.pi / 60.0)
    torque = np.interp(t_vib, t_mech, recording.torque)

    force = force_synthesis.torque_ripple_force(torque, gain=1.0)
    dt_sub = 1.0 / cal.VIBRATION_FS_HZ
    vib_b = np.zeros(len(recording.vibration))
    for k in range(len(force) - 1):
        vib_b[k] = axis.step(force[k : k + 2], dt_sub)
    vib_b[-1] = vib_b[-2]
    return omega, torque, vib_b


def windows(arr, window, stride):
    n = (len(arr) - window) // stride + 1
    return np.stack([arr[i * stride : i * stride + window] for i in range(n)])


def build_dataset(modes, bearing_codes, dataset_root):
    seq_features, rms_targets = [], []
    for code in bearing_codes:
        paths = [p for p in cal.list_recordings(dataset_root, code) if p.name.startswith(CONDITION)][:N_RUNS_PER_BEARING]
        for path in paths:
            rec = cal.load_recording(path)
            omega, torque, vib_b = module_b_prediction(modes, rec)
            residual_x = rec.vibration - vib_b

            omega_w = windows(omega, WINDOW_SAMPLES, STRIDE_SAMPLES)
            torque_w = windows(torque, WINDOW_SAMPLES, STRIDE_SAMPLES)
            vib_b_w = windows(vib_b, WINDOW_SAMPLES, STRIDE_SAMPLES)
            residual_w = windows(residual_x, WINDOW_SAMPLES, STRIDE_SAMPLES)

            # z-score omega/torque per-recording (their absolute scale carries little
            # window-to-window information here since speed/load are ~constant within a run;
            # what matters for the GRU is the shape of vib_b within the window) and leave vib_b
            # in its native scale (its magnitude is informative).
            def z(x):
                std = x.std()
                return (x - x.mean()) / std if std > 1e-12 else x - x.mean()

            seq = np.stack([z(omega_w), z(torque_w), vib_b_w], axis=-1)  # (n_windows, WINDOW, 3)
            rms = np.sqrt(np.mean(residual_w**2, axis=1))  # (n_windows,)

            seq_features.append(seq)
            rms_targets.append(rms)
        print(f"  {code}: {len(paths)} recordings loaded")
    return np.concatenate(seq_features), np.concatenate(rms_targets)


def build_gru_model(window_size):
    import keras

    inputs = keras.Input(shape=(window_size, 3), name="window")
    x = keras.layers.GRU(16)(inputs)
    x = keras.layers.Dense(8, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="softplus", name="rms_residual")(x)  # RMS >= 0
    model = keras.Model(inputs, outputs, name="module_c_envelope_gru")
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    with open(repo_root / "configs" / "vibration_module_b.yaml") as f:
        modes_data = yaml.safe_load(f)
    modes = [Mode(**m) for m in modes_data["modes"]]

    print(f"Building training set ({TRAIN_BEARINGS})...")
    x_train, y_train = build_dataset(modes, TRAIN_BEARINGS, args.dataset_root)
    print(f"train windows: X={x_train.shape} y={y_train.shape}")

    print(f"Building holdout set ({HOLDOUT_BEARINGS})...")
    x_holdout, y_holdout = build_dataset(modes, HOLDOUT_BEARINGS, args.dataset_root)
    print(f"holdout windows: X={x_holdout.shape} y={y_holdout.shape}")

    baseline_rmse = float(np.sqrt(np.mean(y_holdout**2)))  # "Module B alone" = predict 0 residual RMS
    naive_mean_rmse = float(np.sqrt(np.mean((y_holdout - y_train.mean()) ** 2)))  # predict train-set mean RMS
    print(f"\nBaseline (predict 0 residual RMS, i.e. Module B alone) holdout RMSE: {baseline_rmse:.4f}")
    print(f"Naive (predict training-set mean RMS) holdout RMSE:                 {naive_mean_rmse:.4f}")

    model = build_gru_model(WINDOW_SAMPLES)
    model.summary()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=128, validation_split=0.1, verbose=2)

    pred_holdout = model.predict(x_holdout, verbose=0).flatten()
    model_rmse = float(np.sqrt(np.mean((y_holdout - pred_holdout) ** 2)))
    improvement_vs_zero = (1 - model_rmse / baseline_rmse) * 100
    improvement_vs_naive = (1 - model_rmse / naive_mean_rmse) * 100
    print(f"\nGRU envelope model holdout RMSE: {model_rmse:.4f}")
    print(f"  vs. Module B alone (0):        {improvement_vs_zero:+.1f}%")
    print(f"  vs. naive (train-set mean):    {improvement_vs_naive:+.1f}%")

    out_path = repo_root / "configs" / "module_c_envelope_gru.keras"
    model.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
