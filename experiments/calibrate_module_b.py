"""Reproduces the Module B calibration in configs/vibration_module_b.yaml (Macro-fase A.3).

Needs a local extraction of the KAt-DataCenter (Paderborn) .rar archives (CC BY-NC, not part of
this repo -- see docs/addendum_vibracion_v1.md Sec. 9). Point --dataset-root at the folder holding
one subfolder per bearing code (K001/, K002/, ..., each with its .mat runs).

Usage:
    python experiments/calibrate_module_b.py --dataset-root /path/to/BearingDataCenter/extracted

What this does, and why the result looks the way it does:

1. Pools every healthy bearing (K001-K006) at matched operating conditions (900 rpm / 0.7 Nm /
   1000 N, the "N09_M07_F10" code) and fits a 4-mode bank to the averaged vibration PSD.
2. Runs a leave-bearing-out cross-validation (2 folds x 3 holdout bearings = 6 evaluations) to
   check whether a mode bank fit on some bearings transfers to an unseen one.

Finding (see cross_validation_note in the saved YAML): it does NOT transfer well -- holdout RMSE
in log10(PSD) is worse than a trivial flat-PSD baseline in 5/6 folds. This means individual
physical bearing samples have PSD idiosyncrasies a shared few-mode structural model cannot
capture, even under matched operating conditions and even though the fit-set RMSE is good
(~0.06). Module B is a coarse physical approximation of the *generic* structural resonance --
docs/addendum_vibracion_v1.md originally paired this with Module C (a data-driven residual
corrector) to absorb per-bearing-sample idiosyncrasies, but Module C was retired after failing to
find any exploitable signal (see docs/patch2_retiro_modulo_C.md, docs/patch3_mejora_modulo_B.md).
The pooled fit here is still the most defensible "generic" Module B calibration available without
per-bearing recalibration, and is what gets saved -- along with a calibrated background-noise
gain (Patch 3's replacement for the retired torque-coupled excitation, see
sim/vibration/background_noise.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from driveflow.sim.vibration import calibration as cal

HEALTHY_BEARING_CODES = ["K001", "K002", "K003", "K004", "K005", "K006"]
OPERATING_CONDITION = "N09_M07_F10"  # 900 rpm / 0.7 Nm / 1000 N, matched across all six
N_MODES = 4
FREQ_BOUNDS_HZ = (100.0, 10000.0)
DAMPING_BOUNDS = (1e-3, 0.3)
N_RUNS_PER_BEARING = 15


def load_bearing(dataset_root, code, n=N_RUNS_PER_BEARING, condition=OPERATING_CONDITION):
    paths = [p for p in cal.list_recordings(dataset_root, code) if p.name.startswith(condition)][:n]
    return [cal.load_recording(p) for p in paths]


def cross_validate(dataset_root, recs_by_code):
    """Leave-bearing-out CV: 2 folds of 3 fit / 3 holdout bearings each."""
    folds = [
        (["K002", "K003", "K004"], ["K001", "K005", "K006"]),
        (["K001", "K005", "K006"], ["K002", "K003", "K004"]),
    ]
    results = []
    for fit_codes, holdout_codes in folds:
        fit_recs = sum((recs_by_code[c] for c in fit_codes), [])
        modes, _, _ = cal.calibrate_module_b(fit_recs, n_modes=N_MODES, freq_bounds=FREQ_BOUNDS_HZ, damping_bounds=DAMPING_BOUNDS)
        for holdout_code in holdout_codes:
            freqs_h, psd_h = cal.averaged_psd(recs_by_code[holdout_code])
            model_rmse = cal.psd_log_rmse(freqs_h, psd_h, modes, freq_bounds=FREQ_BOUNDS_HZ)
            mask = (freqs_h >= FREQ_BOUNDS_HZ[0]) & (freqs_h <= FREQ_BOUNDS_HZ[1])
            null_rmse = float(
                np.sqrt(np.mean((np.log10(np.mean(psd_h[mask])) - np.log10(np.maximum(psd_h[mask], 1e-30))) ** 2))
            )
            results.append(
                dict(fit_codes=fit_codes, holdout_code=holdout_code, model_rmse=model_rmse, null_rmse=null_rmse)
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted KAt-DataCenter archives")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "configs" / "vibration_module_b.yaml")
    )
    args = parser.parse_args()

    recs_by_code = {code: load_bearing(args.dataset_root, code) for code in HEALTHY_BEARING_CODES}
    n_total = sum(len(v) for v in recs_by_code.values())
    print(f"Loaded {n_total} recordings across {HEALTHY_BEARING_CODES}")

    cv_results = cross_validate(args.dataset_root, recs_by_code)
    for r in cv_results:
        print(f"  fit={r['fit_codes']} holdout={r['holdout_code']}: model_rmse={r['model_rmse']:.3f} null_rmse={r['null_rmse']:.3f}")

    all_recs = sum(recs_by_code.values(), [])
    modes, freqs, psd = cal.calibrate_module_b(all_recs, n_modes=N_MODES, freq_bounds=FREQ_BOUNDS_HZ, damping_bounds=DAMPING_BOUNDS)
    pooled_rmse = cal.psd_log_rmse(freqs, psd, modes, freq_bounds=FREQ_BOUNDS_HZ)
    print(f"Pooled fit-set RMSE: {pooled_rmse:.4f}")

    background_gain = cal.fit_background_noise_gain(all_recs, modes)
    print(f"Background-noise gain (RMS-matched, healthy pool): {background_gain:.6f}")

    worse_than_null = sum(1 for r in cv_results if r["model_rmse"] > r["null_rmse"])
    out = {
        "geometry": {
            "source": "K001.pdf..K006.pdf (KAt-DataCenter 'Profile of rolling bearing damage' datasheets)",
            "bearing_type": "6203",
            "n_elements": 8,
            "element_diameter_m": 6.75e-3,
            "pitch_diameter_m": 29.05e-3,
            "contact_angle_rad": 0.0,
        },
        "operating_condition_fit_to": f"{OPERATING_CONDITION} (900 rpm nominal, matched across {HEALTHY_BEARING_CODES})",
        "fit_bearing_codes": HEALTHY_BEARING_CODES,
        "n_recordings_used": n_total,
        "freq_bounds_hz": list(FREQ_BOUNDS_HZ),
        "damping_bounds": list(DAMPING_BOUNDS),
        "modes": [
            {"natural_freq_hz": float(m.natural_freq_hz), "damping_ratio": float(m.damping_ratio), "gain": float(m.gain)}
            for m in modes
        ],
        "pooled_fit_rmse_log10_psd": pooled_rmse,
        "background_noise_gain": background_gain,
        "cross_validation": cv_results,
        "cross_validation_note": (
            f"Leave-bearing-out CV ({len(cv_results)} holdout evals) shows this generic fit does NOT "
            f"transfer well to individual unseen bearing samples: model RMSE was worse than a flat-PSD "
            f"null model in {worse_than_null}/{len(cv_results)} folds. Module B is a coarse physical "
            "approximation of the *generic* structural resonance, not a per-bearing-exact model -- "
            "Module C (a data-driven residual corrector meant to absorb per-sample idiosyncrasies) was "
            "tried and retired, see docs/patch2_retiro_modulo_C.md."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
