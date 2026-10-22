"""E14 last attempt (round 2026-10-22, decision D2): noise-aware residual threshold, implemented
exactly as pre-registered in docs/audit/ai/e14_noise_aware_preregistration.md (commits 8615d31 and
0c76c85, both made before anything below was computed).

Steps (section numbers of the pre-registration):
  §2 sigma_hat per window: MAD of current_r high-passed at 1 kHz (8th-order Butterworth, zero-phase),
     rescaled to full-band white noise; ripple-inflation check on healthy windows + synthetic ripple.
  §3 calibration set: the 21 healthy validation runs of run02 + 21 new healthy grid-recipe runs at
     50/75/100 % noise (seeds 5001-5021), gated windows only, 60 per run.
  §4 tau(sigma_hat): 8 equal-count bins, 99th percentile per bin, cumulative max; above the largest
     calibration sigma_hat -> "out of calibrated noise range".
  §5 single evaluation on the E14 test conditions with r = score / tau(sigma_hat).
  §6 cut rule.

    python experiments/report_e14_noise_aware.py
Writes docs/audit/ai/e14_noise_aware_results.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import roc_auc_score

from driveflow.ai import residual
from driveflow.ai.registry import load_run_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "ai"
RUN = ROOT / "configs" / "regressors" / "dc_motor_residual_pc_novv" / "2026-10-21_run02"
FS_HZ, FC_HZ, ORDER = 10_000.0, 1000.0, 8
HARMONIC_WEIGHTS = (1.0, 0.35, 0.15, 0.07)  # datagen/fault_injection._HARMONIC_WEIGHTS
RIPPLE_A = 2 * 8.0 / 0.165  # largest grid torque ripple (severity x2 of 8 N·m) / psi_e, in A
FAULT_ORDERS = {"outer_race": 3.07, "inner_race": 4.93}
INFLATION_LIMIT = 0.10
N_BINS, QUANTILE, CAP = 8, 0.99, 60
CAL_SPEEDS = (80.0, 100.0, 125.0, 150.0, 175.0, 200.0, 220.0)
CAL_NOISE = (50.0, 75.0, 100.0)
HELD_OUT = ("unseen_speed_115_rad_s", "unseen_speed_190_rad_s", "extrapolated_speed_250_rad_s", "noise_100pct", "plant_drift_psi-10pct_J+50pct")
OOR_GUARD = 0.10
_SOS = butter(ORDER, FC_HZ, btype="highpass", fs=FS_HZ, output="sos")


def sigma_hat(x: np.ndarray) -> np.ndarray:
    """§2: per-row robust noise std of x (n, samples), full-band white-noise equivalent."""
    hp = sosfiltfilt(_SOS, np.asarray(x, dtype=float), axis=-1)
    mad = np.median(np.abs(hp - np.median(hp, axis=-1, keepdims=True)), axis=-1)
    return 1.4826 * mad / np.sqrt(1.0 - FC_HZ / (FS_HZ / 2))


def _sim_cal(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    sc = Scenario(scenario_id=f"e14cal_w{job['speed']:.0f}_n{job['noise']:.0f}_seed{job['seed']}", fault_type=None, seed=job["seed"], duration_s=2.5,
                  omega_ref_rad_s=job["speed"], electrical_noise_pct=job["noise"], mechanical_noise_pct=job["noise"],
                  motor_parameter_overrides={"psi_e": job["psi_e"], "j_rotor": job["j_rotor"]})
    df = records_to_dataframe(run_scenario(sc))
    df["grid_electrical_noise_pct"] = job["noise"]
    return df


def calibration_runs(cache: Path, workers: int) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    rng = np.random.default_rng(22)
    jobs, seed = [], 5000
    for speed in CAL_SPEEDS:
        for noise in CAL_NOISE:
            seed += 1
            jobs.append({"speed": speed, "noise": noise, "seed": seed, "psi_e": 0.165 * float(rng.uniform(0.9, 1.1)), "j_rotor": 0.025 * float(rng.uniform(0.8, 1.5))})
    with mp.get_context("spawn").Pool(workers) as pool:
        df = pd.concat(pool.map(_sim_cal, jobs), ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _segments(df, metrics, gate_s):
    return residual.raw_windows(df, list(metrics["channels"]), int(metrics["input_window"]), int(metrics["horizon"]), cap_per_group=CAP, start_s=gate_s)


def _auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "diagnosis_grid_dataset.parquet")
    ap.add_argument("--test-cache", type=Path, default=ROOT / "data" / "eval_cache_5class.parquet")
    ap.add_argument("--cal-cache", type=Path, default=ROOT / "data" / "e14_noise_aware_calibration.parquet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    model, _cfg, metrics = load_run_model(RUN, "regressor")
    det = metrics["residual_detector"]
    gate_s = float(det["gate"]["gate_s"])
    ch = list(metrics["channels"])
    j_cur = ch.index("current_r")
    rep = {"command": "python experiments/report_e14_noise_aware.py", "preregistration": "docs/audit/ai/e14_noise_aware_preregistration.md (8615d31, 0c76c85)",
           "run": str(RUN.relative_to(ROOT)), "gate_s": gate_s, "estimator": {"fc_hz": FC_HZ, "butterworth_order": ORDER, "zero_phase": True, "channel": "current_r"},
           "environment": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"}

    # §3 calibration set
    grid = pd.read_parquet(args.dataset)
    old = grid[grid["source_file"].isin(metrics["val_groups"])]
    assert set(old["label"]) == {"normal"}
    new = calibration_runs(args.cal_cache, args.workers)
    cal_df = pd.concat([old, new], ignore_index=True)
    seg, groups, labels, _ = _segments(cal_df, metrics, gate_s)
    noise_by_run = cal_df.groupby("source_file")["grid_electrical_noise_pct"].first()
    win_noise = noise_by_run.reindex(groups).to_numpy()
    s_cal = sigma_hat(seg[:, :, j_cur])
    rep["calibration"] = {"n_runs_recorded_val": int(old["source_file"].nunique()), "n_runs_new": int(new["source_file"].nunique()), "n_windows": int(len(seg)),
                          "noise_pct_range": [float(np.min(win_noise)), float(np.max(win_noise))], "sigma_hat_range_a": [float(s_cal.min()), float(s_cal.max())],
                          "sigma_hat_rpm_median": float(np.median(sigma_hat(seg[:, :, ch.index("rpm")])))}

    # §2 ripple-inflation check (healthy windows + synthetic ripple)
    rng = np.random.default_rng(0)
    mask = win_noise >= 10.0
    t = np.arange(seg.shape[1]) / FS_HZ
    mean_rpm = seg[:, :, ch.index("rpm")].mean(axis=1)
    infl = {}
    for fault, order in FAULT_ORDERS.items():
        f_f = order * mean_rpm[mask] / 60.0
        phi = rng.uniform(0, 2 * np.pi, size=mask.sum())
        ripple = RIPPLE_A * sum(w * np.cos(2 * np.pi * (k + 1) * f_f[:, None] * t[None, :] + phi[:, None]) for k, w in enumerate(HARMONIC_WEIGHTS))
        with_r = sigma_hat(seg[mask, :, j_cur] + ripple)
        infl[fault] = float(np.median(with_r / s_cal[mask] - 1.0))
    rep["inflation_check"] = {"n_windows": int(mask.sum()), "ripple_amplitude_a": RIPPLE_A, "median_relative_inflation": infl, "limit": INFLATION_LIMIT,
                              "pass": all(v <= INFLATION_LIMIT for v in infl.values())}
    print("inflation", rep["inflation_check"], flush=True)
    if not rep["inflation_check"]["pass"]:
        rep["verdict"] = "NEGATIVE: the noise estimator is inflated by fault ripple (§2); no calibration or test evaluation (§6)"
        return _write(rep)

    # §4 tau(sigma_hat)
    score_cal = residual.score(model, seg, metrics)
    edges = np.unique(np.quantile(s_cal, np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.searchsorted(edges, s_cal, side="right") - 1, 0, len(edges) - 2)
    taus = []
    for b in range(len(edges) - 1):
        sb = score_cal[idx == b]
        taus.append(float(np.quantile(sb, QUANTILE)) if len(sb) else np.nan)
    taus = np.maximum.accumulate(np.nan_to_num(np.asarray(taus), nan=-np.inf))
    s_max = float(s_cal.max())
    rep["threshold"] = {"bin_edges_a": edges.tolist(), "tau_per_bin": taus.tolist(), "n_per_bin": np.bincount(idx, minlength=len(edges) - 1).tolist(),
                        "sigma_hat_max_calibrated_a": s_max, "run02_constant_threshold": det["threshold"]}

    def r_of(seg_t):
        if len(seg_t) == 0:
            return np.array([]), np.array([], bool)
        s = sigma_hat(seg_t[:, :, j_cur])
        oor = s > s_max
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
        return residual.score(model, seg_t, metrics) / taus[b], oor

    # §5 single evaluation
    test = pd.read_parquet(args.test_cache)
    per, pooled_neg, pooled_pos = {}, [], {f: [] for f in FAULT_ORDERS}
    for cond in HELD_OUT:
        sub = test[test["condition"] == cond]
        r_h, oor_h = r_of(_segments(sub[sub["label"] == "normal"], metrics, gate_s)[0])
        row = {"healthy_n": int(len(r_h)), "healthy_out_of_range": float(oor_h.mean()) if len(r_h) else None,
               "false_alarm_rate_in_range": float(np.mean(r_h[~oor_h] > 1)) if (~oor_h).any() else None}
        pooled_neg.append(r_h[~oor_h])
        for f in FAULT_ORDERS:
            r_f, oor_f = r_of(_segments(sub[sub["label"] == f], metrics, gate_s)[0])
            pooled_pos[f].append(r_f[~oor_f])
            row[f] = {"n": int(len(r_f)), "out_of_range": float(oor_f.mean()) if len(r_f) else None, "auroc": _auroc(r_f[~oor_f], r_h[~oor_h]),
                      "detection_rate_in_range": float(np.mean(r_f[~oor_f] > 1)) if (~oor_f).any() else None}
        far, oor = row["false_alarm_rate_in_range"], row["healthy_out_of_range"]
        row["criterion_3_pass"] = far is not None and far <= 0.01 and oor is not None and oor <= OOR_GUARD
        per[cond] = row
    neg = np.concatenate(pooled_neg)
    c2 = {f: _auroc(np.concatenate(pooled_pos[f]), neg) for f in FAULT_ORDERS}
    sev06 = test[(test["condition"] == "unseen_severity_x0.6")]
    det06 = {}
    for f in FAULT_ORDERS:
        r_f, oor_f = r_of(_segments(sev06[sev06["label"] == f], metrics, gate_s)[0])
        det06[f] = float(np.mean(r_f[~oor_f] > 1)) if (~oor_f).any() else None
    rep["evaluation"] = {"per_condition": per, "criterion_2_pooled_auroc": c2, "criterion_2_pass": all(v is not None and v >= 0.9 for v in c2.values()),
                         "criterion_3_pass": all(r["criterion_3_pass"] for r in per.values()), "detection_rate_severity_x0.6_in_range": det06}
    ok = rep["evaluation"]["criterion_2_pass"] and rep["evaluation"]["criterion_3_pass"]
    rep["verdict"] = ("PASS: both criteria met. Not promoted, not integrated; waiting for the user's decision (§6)" if ok
                      else "NEGATIVE: criteria not met; E14 closes as a negative result, no further attempts (§6)")
    _write(rep)


def _f(x, spec=".3f"):
    return "—" if x is None else format(x, spec)


def _write(rep: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e14_noise_aware_results.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    L = ["# E14 — noise-aware residual threshold: results (round 2026-10-22, T6)", "",
         f"Pre-registration: [`e14_noise_aware_preregistration.md`](e14_noise_aware_preregistration.md), commits 8615d31 and 0c76c85, both made before this run. "
         f"Command: `{rep['command']}` ({rep['environment']}). Raw values: [`e14_noise_aware_results.json`](e14_noise_aware_results.json). "
         f"Run: `{rep['run']}`; gate {rep['gate_s'] * 1000:.0f} ms. Simulated windows only.", "",
         f"**Verdict: {rep['verdict']}**", ""]
    c = rep.get("calibration")
    if c:
        L += ["## Calibration set (§3)", "",
              f"- Runs: {c['n_runs_recorded_val']} recorded validation runs and {c['n_runs_new']} new runs. Windows: {c['n_windows']} gated.",
              f"- Electrical noise: {c['noise_pct_range'][0]:.0f}–{c['noise_pct_range'][1]:.0f} %.",
              f"- σ̂(current_r): {c['sigma_hat_range_a'][0]:.3g}–{c['sigma_hat_range_a'][1]:.3g} A. Median σ̂(rpm), not used: {c['sigma_hat_rpm_median']:.3g}.", ""]
    i = rep["inflation_check"]
    L += ["## Ripple-inflation check (§2)", "",
          f"- Data: {i['n_windows']} healthy windows with noise ≥ 10 %, plus synthetic ripple of {i['ripple_amplitude_a']:.1f} A.",
          "- Median relative inflation of σ̂: " + ", ".join(f"{k} {v:+.2%}" for k, v in i["median_relative_inflation"].items()) + f" (limit {i['limit']:.0%}).",
          f"- Result: **{'PASS' if i['pass'] else 'FAIL'}**.", ""]
    t = rep.get("threshold")
    if t:
        L += ["## Threshold τ(σ̂) (§4)", "", "| Bin | σ̂ range (A) | Windows | τ (q99, monotone) |", "|---|---|---|---|"]
        for b, tau in enumerate(t["tau_per_bin"]):
            L.append(f"| {b + 1} | {t['bin_edges_a'][b]:.3g}–{t['bin_edges_a'][b + 1]:.3g} | {t['n_per_bin'][b]} | {tau:.3f} |")
        L += ["", f"Above σ̂ = {t['sigma_hat_max_calibrated_a']:.3g} A the answer is \"out of calibrated noise range\". The run02 constant threshold was {t['run02_constant_threshold']:.3f}.", ""]
    e = rep.get("evaluation")
    if e:
        L += ["## Single evaluation on the E14 test (§5)", "",
              "| Condition | Healthy windows | Out of range (healthy) | False alarms (in range) | AUROC outer / inner | Detection outer / inner (in range) | Criterion 3 |",
              "|---|---|---|---|---|---|---|"]
        for cond, r in e["per_condition"].items():
            L.append(f"| {cond} | {r['healthy_n']} | {_f(r['healthy_out_of_range'], '.1%')} | {_f(r['false_alarm_rate_in_range'], '.1%')} | "
                     f"{_f(r['outer_race']['auroc'])} / {_f(r['inner_race']['auroc'])} | {_f(r['outer_race']['detection_rate_in_range'], '.0%')} / "
                     f"{_f(r['inner_race']['detection_rate_in_range'], '.0%')} | {'PASS' if r['criterion_3_pass'] else 'FAIL'} |")
        L += ["", f"- **Criterion 2** (pooled AUROC ≥ 0.9, severity ×1): outer {_f(e['criterion_2_pooled_auroc']['outer_race'])}, inner {_f(e['criterion_2_pooled_auroc']['inner_race'])}. "
              f"**{'PASS' if e['criterion_2_pass'] else 'FAIL'}**.",
              f"- **Criterion 3** (≤ 1 % false alarms per condition, ≤ 10 % of healthy windows out of range): **{'PASS' if e['criterion_3_pass'] else 'FAIL'}**.",
              f"- Detection at severity ×0.6, in range, reported: outer {_f(e['detection_rate_severity_x0.6_in_range']['outer_race'], '.0%')}, "
              f"inner {_f(e['detection_rate_severity_x0.6_in_range']['inner_race'], '.0%')}.", ""]
    (OUT / "e14_noise_aware_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
