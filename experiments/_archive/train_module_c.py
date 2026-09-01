"""HISTORICAL / NOT RUNNABLE: residual_model.py (Module C) was deleted after this experiment's own
result (see below) motivated retiring it -- see docs/patch2_retiro_modulo_C.md. Kept only as the
documented record of the investigation that led to that decision; do not try to run this.

Moved out of the active experiments/ tree into experiments/_archive/ in Patch 11
(docs/patch11_archivado_modulo_c.md) -- its "Module C" name was repeatedly confused with the
unrelated "Fase C" (diagnosis classifiers, see INSTRUCTIONS.md Sec. 5); living outside the
active tree makes that mix-up harder for a casual `ls experiments/` to reproduce.

Original docstring follows, describing what it did when it still ran:

Trains Module C (residual_model.ResidualCorrector) against the REAL residual observed in
KAt-DataCenter: measured_vibration(t) - Module B's synthetic prediction(t), for the same
recording's real omega(t)/torque(t) trajectory. This is the step the A.3 report flagged as
missing: Module B alone was shown (via calibrate_module_b.py's leave-bearing-out CV) to not
generalize between individual physical bearing samples -- Module C exists specifically to absorb
that per-sample residual (docs/addendum_vibracion_v1.md's whole rationale for a B+C split).

Needs a local extraction of the KAt-DataCenter dataset (see experiments/calibrate_module_b.py's
docstring for how) and configs/vibration_module_b.yaml (Module B's calibration, from that same
script) to already exist.

Known limitation carried into this training set (documented, not silently assumed): only ONE
vibration axis is measured, so the y/z targets used here are the real x-residual scaled by the
SAME per-axis gain ratios datagen/runner.py already uses for Module B's y/z modes (_AXIS_PROFILE)
-- not independently measured. Also, Paderborn's real motor is an induction machine with two AC
phase currents, which have no principled dq-equivalent for driveflow's own DC motor; i_d/i_q are
therefore fixed at 0 in both this training set and datagen/runner.py's inference call, so Module C
does not (yet) use current information in v1 -- consistent with Module B's own i_d/i_q being
unused (see modal_model.py's step() docstring).

Usage:
    python experiments/train_module_c.py --dataset-root /path/to/BearingDataCenter/extracted
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from driveflow.sim.vibration import calibration as cal
from driveflow.sim.vibration import force_synthesis
from driveflow.sim.vibration.modal_model import Mode, ModalAxis
from driveflow.sim.vibration.residual_model import ResidualCorrector

# Same split as calibrate_module_b.py's cross-validation, kept disjoint by bearing (not run) so
# holdout evaluation never leaks the same physical bearing sample into training.
TRAIN_BEARINGS = ["K002", "K004", "K005", "K006"]
HOLDOUT_BEARINGS = ["K001", "K003"]
CONDITION = "N09_M07_F10"
N_RUNS_PER_BEARING = 6
DOWNSAMPLE = 32  # keep every 32nd vibration sample (2 kHz effective) -- density vs. dataset size

# Must match datagen/runner.py's _AXIS_PROFILE gain_scale (y=0.80, z=0.35): the only axis with
# real data is x, so y/z training targets reuse the real x-residual scaled the same way Module
# B's own y/z modes are scaled from the calibrated x modes.
AXIS_GAIN = {"x": 1.00, "y": 0.80, "z": 0.35}


def module_b_prediction(modes, recording):
    """Runs Module B (single axis) forward through the recording's REAL omega(t)/torque(t) at the
    vibration signal's native 64 kHz. Returns (omega_at_vib_rate, torque_at_vib_rate, vib_b)."""
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


def build_dataset(modes, bearing_codes, dataset_root):
    features, targets = [], []
    for code in bearing_codes:
        paths = [p for p in cal.list_recordings(dataset_root, code) if p.name.startswith(CONDITION)][:N_RUNS_PER_BEARING]
        for path in paths:
            rec = cal.load_recording(path)
            omega, torque, vib_b = module_b_prediction(modes, rec)
            residual_x = rec.vibration - vib_b

            idx = np.arange(0, len(residual_x) - 1, DOWNSAMPLE)
            zeros = np.zeros(len(idx))
            feats = np.stack([omega[idx], zeros, zeros, torque[idx], vib_b[idx], vib_b[idx], vib_b[idx]], axis=1)
            tgts = np.stack(
                [residual_x[idx] * AXIS_GAIN["x"], residual_x[idx] * AXIS_GAIN["y"], residual_x[idx] * AXIS_GAIN["z"]],
                axis=1,
            )
            features.append(feats)
            targets.append(tgts)
        print(f"  {code}: {len(paths)} recordings loaded")
    return np.concatenate(features), np.concatenate(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", default=str(repo_root / "configs" / "module_c_residual.keras"))
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    with open(repo_root / "configs" / "vibration_module_b.yaml") as f:
        modes_data = yaml.safe_load(f)
    modes = [Mode(**m) for m in modes_data["modes"]]

    print(f"Building training set ({TRAIN_BEARINGS})...")
    x_train, y_train = build_dataset(modes, TRAIN_BEARINGS, args.dataset_root)
    print(f"train: X={x_train.shape} y={y_train.shape}")

    print(f"Building holdout set ({HOLDOUT_BEARINGS})...")
    x_holdout, y_holdout = build_dataset(modes, HOLDOUT_BEARINGS, args.dataset_root)
    print(f"holdout: X={x_holdout.shape} y={y_holdout.shape}")

    baseline_rmse = float(np.sqrt(np.mean(y_holdout**2)))
    print(f"\nModule B alone (Module C predicts zero residual) holdout RMSE: {baseline_rmse:.4f}")

    corrector = ResidualCorrector(hidden_units=(16, 16))
    corrector.model.compile(optimizer="adam", loss="mse")
    corrector.fit(x_train, y_train, epochs=args.epochs, batch_size=256, validation_split=0.1, verbose=2)

    pred_holdout = corrector.model.predict(x_holdout, verbose=0)
    combined_rmse = float(np.sqrt(np.mean((y_holdout - pred_holdout) ** 2)))
    improvement = (1 - combined_rmse / baseline_rmse) * 100
    print(f"\nModule B+C holdout RMSE: {combined_rmse:.4f} (baseline {baseline_rmse:.4f}, {improvement:+.1f}% change)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    corrector.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
