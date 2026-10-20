"""Step 0 of D.2: is a degraded VSC filter capacitor separable from a healthy one at all?
-- TODO_CLAUDE_CODE.md A4, docs/INSTRUCTIONS.md Sec. 6 D.2.

PRE-REGISTERED (fixed in this docstring before any result was computed):
- Primary metric: out-of-fold ROC AUC of a logistic regression (standardized features,
  GroupKFold(5) grouped by operating point (R, |v_ref|, phase), so the healthy and the degraded run
  of one operating point are always in the same fold) separating HEALTHY windows from windows with
  the capacitance at 80 % of nominal (ESR unchanged).
- Cut rule: if that AUC < 0.70, stop D.2 and document the limitation; do not build a classifier.
- Secondary (reported, no decision attached): the same AUC for C = 90 % and 70 %, for ESR = 0.05
  and 0.2 ohm at nominal C, and the best single-feature AUC per level.

Setup:
- plant: sim/vsc_faults.DegradedVscSystem (exact ZOH of the L-C model with c_scale / esr_ohm);
- controller: the default DPC v4 (configs/dpc_trained_v4_r_range.weights.h5). It was trained, and
  its loss written, for the NOMINAL capacitor, so closed-loop control may hide the degradation;
- operating points: R = 8.0064, 20, 100 ohm x |v_ref| = 35, 50, 75 V x 4 reference phases;
- 2000 steps (0.2 s) per run; the first 300 steps (start-up) are dropped; non-overlapping windows
  of 256 steps (25.6 ms, more than one 50 Hz cycle) -> 6 windows per run;
- features per window, from (v_c, v_ref, i_f) and the control effort v_o: RMS tracking error,
  mean |i_f|, RMS |v_o|, mean active and reactive filter current relative to v_c, std of |v_c| and
  of |i_f|, RMS|v_o| / RMS|v_c|, plus the known operating point (R, |v_ref|).

PARAMETER-UNCERTAINTY CHECK (pre-registered 2026-10-20, before running it; --param-uncertainty-pct
U): every run's REAL plant draws L, R_f and the load R independently from U(1 - U/100, 1 + U/100)
x nominal, while the DPC controller and the features only know the NOMINAL R (and the nominal L,
R_f, C built into the controller). Healthy and degraded runs draw independently. Run with U = 5 and
10, both with 5 % measurement noise. Same metric and same cut as above: if the AUC at C = 80 %
falls below 0.70, a capacitor classifier is not viable with these signals and the follow-up stops.

POST-HOC robustness check (added after the primary result, declared as such; it does not change
the pre-registered verdict): --noise-pct adds complex Gaussian measurement noise to v_c, i_f and
v_o before the features are computed, with a standard deviation of that percentage of each
signal's RMS over the run. The controller still sees noise-free states.

EPISTEMIC WARNING: no real reference dataset exists for capacitor degradation; any separability
measured here holds for this synthetic fault definition only.

Usage:

    python experiments/verify_vsc_capacitor_separability.py --out-dir docs/audit/ai
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEVELS = {
    "healthy": (1.0, 0.0),
    "C_90pct": (0.9, 0.0),
    "C_80pct": (0.8, 0.0),
    "C_70pct": (0.7, 0.0),
    "ESR_0.05ohm": (1.0, 0.05),
    "ESR_0.2ohm": (1.0, 0.2),
}
PRIMARY = "C_80pct"
CUT_AUC = 0.70
R_OHM = (8.0064, 20.0, 100.0)
MAGNITUDES = (35.0, 50.0, 75.0)
PHASES = (0.0, 0.7, 1.9, 3.3)
N_STEPS, WARMUP, WINDOW = 2000, 300, 256
FEATURES = ["rms_tracking_error", "mean_abs_i_f", "rms_v_o", "active_current", "reactive_current", "std_abs_v_c", "std_abs_i_f", "v_o_over_v_c", "r_ohm", "v_ref_magnitude"]
WEIGHTS = ROOT / "configs" / "dpc_trained_v4_r_range.weights.h5"


NOISE_PCT = 0.0


def _run(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.control.dpc.controller import DpcController
    from driveflow.control.dpc.reference import RotatingReference
    from driveflow.sim.vsc_faults import DegradedVscSystem

    level, c_scale, esr, r, mag, phase, noise_pct, unc_pct = job
    l_scale = rf_scale = r_scale = 1.0
    if unc_pct > 0:
        urng = np.random.default_rng(zlib.crc32(repr(("params", level, r, mag, phase)).encode()))
        l_scale, rf_scale, r_scale = urng.uniform(1 - unc_pct / 100, 1 + unc_pct / 100, size=3)
    plant = DegradedVscSystem(r * r_scale, c_scale=c_scale, esr_ohm=esr, l_scale=l_scale, rf_scale=rf_scale)  # controller keeps nominal r
    ctrl = DpcController(WEIGHTS, r_ohm=r)
    ref = RotatingReference(magnitude_v=mag, phase0_rad=phase)
    s = plant.reset()
    vc, vr, i_f, vo = (np.empty(N_STEPS, complex) for _ in range(4))
    for k in range(N_STEPS):
        u = ctrl.control(s, ref, k)
        s = plant.simulate(*u)
        vo[k] = complex(*u)
        vc[k], i_f[k] = complex(s.vc_real, s.vc_imag), complex(s.i_f_real, s.i_f_imag)
        vr[k] = complex(*ref.at_step(k))
    if noise_pct > 0:
        rng = np.random.default_rng(zlib.crc32(repr((level, r, mag, phase)).encode()))  # reproducible across processes
        for sig in (vc, i_f, vo):
            sd = noise_pct / 100 * np.sqrt(np.mean(np.abs(sig) ** 2)) / np.sqrt(2)
            sig += rng.normal(0, sd, N_STEPS) + 1j * rng.normal(0, sd, N_STEPS)
    rows = []
    for start in range(WARMUP, N_STEPS - WINDOW + 1, WINDOW):
        w = slice(start, start + WINDOW)
        v, i, o = vc[w], i_f[w], vo[w]
        unit = v / np.maximum(np.abs(v), 1e-9)
        rows.append([
            float(np.sqrt(np.mean(np.abs(v - vr[w]) ** 2))), float(np.mean(np.abs(i))), float(np.sqrt(np.mean(np.abs(o) ** 2))),
            float(np.mean((i * np.conj(unit)).real)), float(np.mean((i * np.conj(unit)).imag)),
            float(np.std(np.abs(v))), float(np.std(np.abs(i))),
            float(np.sqrt(np.mean(np.abs(o) ** 2)) / max(np.sqrt(np.mean(np.abs(v) ** 2)), 1e-9)), r, mag,
        ])
    return {"level": level, "run": f"{level}|R{r:g}|V{mag:g}|p{phase:g}", "op": f"R{r:g}|V{mag:g}|p{phase:g}", "X": rows, "finite": bool(np.all(np.isfinite(rows)))}


def _auc_cv(X, y, groups) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main(argv=None):
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "audit" / "ai")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--noise-pct", type=float, default=0.0, help="post-hoc robustness check: measurement noise, %% of each signal's RMS")
    ap.add_argument("--param-uncertainty-pct", type=float, default=0.0, help="real plant's L, R_f and R drawn within +-this %% of nominal (pre-registered check)")
    args = ap.parse_args(argv)

    jobs = [(lvl, c, e, r, m, p, args.noise_pct, args.param_uncertainty_pct) for lvl, (c, e) in LEVELS.items() for r in R_OHM for m in MAGNITUDES for p in PHASES]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(_run, jobs)

    def stack(level):
        sel = [r for r in runs if r["level"] == level]
        X = np.array([row for r in sel for row in r["X"]])
        g = np.array([r["op"] for r in sel for _ in r["X"]])
        return X, g

    Xh, gh = stack("healthy")
    res = {"measurement_noise_pct": args.noise_pct, "param_uncertainty_pct": args.param_uncertainty_pct, "pre_registered": {"primary_level": PRIMARY, "cut_auc": CUT_AUC, "metric": "out-of-fold logistic-regression AUC, GroupKFold(5) by operating point"},
           "features": FEATURES, "n_runs_per_level": len(R_OHM) * len(MAGNITUDES) * len(PHASES), "windows_per_run": len(runs[0]["X"]),
           "all_finite": all(r["finite"] for r in runs), "levels": {}}
    for level in LEVELS:
        if level == "healthy":
            continue
        Xd, gd = stack(level)
        X = np.vstack([Xh, Xd])
        y = np.r_[np.zeros(len(Xh)), np.ones(len(Xd))]
        # Group by operating point, not by run: a healthy run and a degraded run at the same
        # (R, |v_ref|, phase) share the controller's response to that point, so keeping them in the
        # same fold stops the classifier from matching operating points instead of the fault.
        groups = np.r_[gh, gd]
        single = {f: float(max(roc_auc_score(y, X[:, j]), 1 - roc_auc_score(y, X[:, j]))) for j, f in enumerate(FEATURES) if f not in ("r_ohm", "v_ref_magnitude")}
        best = max(single, key=single.get)
        res["levels"][level] = {"c_scale": LEVELS[level][0], "esr_ohm": LEVELS[level][1], "auc_logistic_grouped_cv": _auc_cv(X, y, groups),
                                "best_single_feature": best, "best_single_feature_auc": single[best], "single_feature_auc": single}
        print(level, {k: v for k, v in res["levels"][level].items() if k != "single_feature_auc"}, flush=True)

    primary = res["levels"][PRIMARY]["auc_logistic_grouped_cv"]
    res["verdict"] = "PROCEED: separable above the pre-registered cut" if primary >= CUT_AUC else "STOP D.2: below the pre-registered cut"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "vsc_capacitor_separability" + (f"_noise{args.noise_pct:g}pct" if args.noise_pct else "") + (f"_params{args.param_uncertainty_pct:g}pct" if args.param_uncertainty_pct else "") + ".json"
    (args.out_dir / name).write_text(json.dumps(res, indent=2) + "\n")
    print(res["verdict"], f"(primary AUC {primary:.3f}, cut {CUT_AUC})")


if __name__ == "__main__":
    main()
