"""R2 (round PLECS/OpenModelica, 2026-10-25): sensitivity of fault diagnosis to the native DC
plant's discretization (finding S-16).

The native plant ``dc_perm_ex_v1`` integrates one forward-Euler step per control period
TAU = 100 us; its armature time constant is ~1.19 ms. Every diagnosis dataset was generated with
it. This script asks whether its discretization error changes what the promoted classifiers see
and decide. Two stages, run separately:

``convergence`` (R2.1, decision G7): picks the study's reference integration. For three
operating scenarios (start-up, steady state, load-torque step), healthy and faulted, it compares
Euler with 1, 10, 100 and 1000 substeps per TAU and RK45 (rtol = atol = 1e-9) on i_a, omega and
torque. The reference is the coarsest Euler whose NRMS against the finest solution is < 1e-4 on
all three signals in every case. The controller acts every TAU in all of them; only the plant
integration changes (Scenario.dc_plant_solver / dc_plant_substeps).

``sensitivity`` (R2.2/R2.3): the pre-registered study (docs/audit/plecs_om/r2_preregistration.md).
NAT = the production plant, REF = the converged reference; same seeds, same configuration, paired
windows. Measures signal NRMS, fault-band spectral amplitudes, and top-1 agreement / accuracy /
macro-F1 / calibrated confidence / abstention of the three promoted dc_motor classifiers.

Usage:
  python experiments/report_s16_discretization_sensitivity.py convergence [--out docs/audit/plecs_om]
  python experiments/report_s16_discretization_sensitivity.py sensitivity [--speeds 100,150,200]
      [--severities 1.0,0.5] [--seeds 2] [--workers 8] [--cache sims.parquet] [--out docs/audit/plecs_om]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO_ROOT / "docs" / "audit" / "plecs_om"

# ----------------------------------------------------------------------------- convergence (R2.1)
#: integration name -> Scenario overrides
INTEGRATIONS = {
    "euler_x1": {},
    "euler_x10": {"dc_plant_substeps": 10},
    "euler_x100": {"dc_plant_substeps": 100},
    "euler_x1000": {"dc_plant_substeps": 1000},
    "rk45": {"dc_plant_solver": "rk45"},
}
EULER_ORDER = ("euler_x1", "euler_x10", "euler_x100", "euler_x1000")
FINEST = "rk45"
CONVERGENCE_TOL = 1e-4
SIGNALS = ("current_r", "rpm", "torque_nm")
SIGNAL_NAMES = {"current_r": "i_a", "rpm": "omega", "torque_nm": "torque"}
#: Load-torque step of the third scenario: 4 N·m (25 % of the motor's nominal 16 N·m) at 1.0 s.
LOAD_STEP_NM, LOAD_STEP_T_S = 4.0, 1.0
#: scenario -> (duration_s, extra load step?, time segment [t0, t1) the metric is taken over)
CONV_SCENARIOS = {
    "startup": (1.5, False, (0.0, 0.5)),
    "steady_state": (1.5, False, (0.5, 1.5)),
    "load_step": (1.5, True, (0.9, 1.5)),
}
CONV_SPEED = 150.0
#: Decision of the user (2026-10-25, R2 option a): the reference is RK45 (rtol = atol = 1e-9). Its
#: convergence (G7) is shown against the same RK45 at 1e-11 in every scenario and fault case.
RK45_TIGHT, RK45_TIGHT_TOL = "rk45_tol1e-11", 1e-11
CONV_FAULTS = (None, "outer_race")


def _conv_job(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen import runner

    scen, fault, integ = job
    duration, load_step, _seg = CONV_SCENARIOS[scen]
    if integ == RK45_TIGHT:  # same solver, 100x tighter tolerance (this process only)
        runner.DC_RK45_TOL = RK45_TIGHT_TOL
    original_load = runner.PolynomialStaticLoad
    if load_step:
        # The production plant has no load-step input. The validation benches' LoadTorqueStep is
        # PolynomialStaticLoad plus an external step; swapping it in (this process only) leaves
        # build_plant and everything else as run_scenario uses them.
        from driveflow.validation.benches import LoadTorqueStep

        runner.PolynomialStaticLoad = lambda load_parameter: LoadTorqueStep(load_parameter=load_parameter, step_nm=LOAD_STEP_NM, t_step_s=LOAD_STEP_T_S)
    sc = Scenario(scenario_id=f"r2conv_{scen}_{fault}_{integ}", fault_type=fault, seed=0, duration_s=duration, omega_ref_rad_s=CONV_SPEED,
                  **(INTEGRATIONS["rk45"] if integ == RK45_TIGHT else INTEGRATIONS[integ]))
    t0 = time.time()
    try:
        recs = run_scenario(sc)
    finally:
        runner.PolynomialStaticLoad = original_load
    arr = {s: np.array([r[s] for r in recs]) for s in SIGNALS}
    arr["t"] = np.array([r["timestamp_s"] for r in recs])
    return job, arr, time.time() - t0


def _nrms(a, b, scale):
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale) if scale > 0 else float("nan")


def convergence(args) -> dict:
    jobs = [(s, f, i) for s in CONV_SCENARIOS for f in CONV_FAULTS for i in INTEGRATIONS]
    # start-up and steady state share one simulation; only the segment differs
    jobs = [j for j in jobs if j[0] != "steady_state"]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        results = pool.map(_conv_job, jobs, chunksize=1)
    sims = {(s, f, i): arr for (s, f, i), arr, _w in results}
    wall = {f"{s}|{f}|{i}": round(w, 2) for (s, f, i), _a, w in results}
    cases = {}
    for scen, (_d, load_step, (t0, t1)) in CONV_SCENARIOS.items():
        src = "startup" if scen == "steady_state" else scen
        for fault in CONV_FAULTS:
            ref = sims[(src, fault, FINEST)]
            m = (ref["t"] > t0) & (ref["t"] <= t1)
            row = {}
            for integ in INTEGRATIONS:
                if integ == FINEST:
                    continue
                x = sims[(src, fault, integ)]
                row[integ] = {
                    # primary: normalised by the signal's range over the whole run of the finest
                    # solution (as B3 criterion 4); secondary: by the segment's own RMS
                    "nrms_full_range": {SIGNAL_NAMES[s]: _nrms(x[s][m], ref[s][m], float(np.ptp(ref[s]))) for s in SIGNALS},
                    "nrms_segment_rms": {SIGNAL_NAMES[s]: _nrms(x[s][m], ref[s][m], float(np.sqrt(np.mean(ref[s][m] ** 2)))) for s in SIGNALS},
                }
            cases[f"{scen}|{fault or 'healthy'}"] = row
    passing = {}
    for integ in EULER_ORDER:
        worst = max(v for c in cases.values() for v in c[integ]["nrms_full_range"].values())
        passing[integ] = {"worst_nrms_vs_finest": worst, "passes": worst < CONVERGENCE_TOL}
    # demonstrated convergence of the finest itself: Euler x1000 vs RK45 must be well below the tolerance
    chosen = next((i for i in EULER_ORDER if passing[i]["passes"]), None)
    return {
        "stage": "convergence", "finest": FINEST, "rk45_tol": 1e-9, "tolerance": CONVERGENCE_TOL,
        "rule": "reference = the coarsest Euler substep count whose NRMS vs the finest solution is < 1e-4 on i_a, omega and torque, in every scenario and fault case",
        "normalisation": "primary: RMS(diff over the segment) / range of the finest solution over the whole run; secondary (reported only): / RMS of the finest solution over the segment",
        "scenarios": {k: {"duration_s": v[0], "load_step": {"nm": LOAD_STEP_NM, "t_s": LOAD_STEP_T_S} if v[1] else None, "segment_s": v[2]} for k, v in CONV_SCENARIOS.items()},
        "speed_rad_s": CONV_SPEED, "faults": [f or "healthy" for f in CONV_FAULTS],
        "cases": cases, "per_integration": passing, "reference": chosen, "wall_s": wall,
    }


def _conv_md(r: dict) -> str:
    lines = ["## R2.1 Convergence of the reference (decision G7)", "",
             f"Finest solution: RK45, rtol = atol = {r['rk45_tol']:g}. Rule: {r['rule']}.",
             f"Normalisation: {r['normalisation']}.", "",
             f"Speed reference {r['speed_rad_s']:g} rad/s, PI controller, seed 0. Load step: {LOAD_STEP_NM:g} N·m at {LOAD_STEP_T_S:g} s (validation benches' `LoadTorqueStep`).", "",
             "| Scenario | Fault | Integration | i_a | omega | torque | i_a (segment-RMS norm.) |", "|---|---|---|---|---|---|---|"]
    for case, row in r["cases"].items():
        scen, fault = case.split("|")
        for integ, v in row.items():
            a = v["nrms_full_range"]
            lines.append(f"| {scen} | {fault} | {integ} | {a['i_a']:.2e} | {a['omega']:.2e} | {a['torque']:.2e} | {v['nrms_segment_rms']['i_a']:.2e} |")
    lines += ["", "| Integration | Worst NRMS vs RK45 | < 1e-4 |", "|---|---|---|"]
    for integ, v in r["per_integration"].items():
        lines.append(f"| {integ} | {v['worst_nrms_vs_finest']:.2e} | {'yes' if v['passes'] else 'no'} |")
    lines += ["", f"**Reference chosen: `{r['reference']}`.**" if r["reference"] else "**No Euler substep count reaches the tolerance: R2 stops here (R2.1).**", ""]
    return "\n".join(lines)


def rk45_convergence(args) -> dict:
    jobs = [(s, f, i) for s in CONV_SCENARIOS if s != "steady_state" for f in CONV_FAULTS for i in ("rk45", RK45_TIGHT, "euler_x1000")]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        results = pool.map(_conv_job, jobs, chunksize=1)
    sims = {(s, f, i): arr for (s, f, i), arr, _w in results}
    cases = {}
    for scen, (_d, _ls, (t0, t1)) in CONV_SCENARIOS.items():
        src = "startup" if scen == "steady_state" else scen
        for fault in CONV_FAULTS:
            fine, ref = sims[(src, fault, RK45_TIGHT)], sims[(src, fault, "rk45")]
            m = (fine["t"] > t0) & (fine["t"] <= t1)
            cases[f"{scen}|{fault or 'healthy'}"] = {
                "rk45_1e-9_vs_1e-11": {SIGNAL_NAMES[s]: _nrms(ref[s][m], fine[s][m], float(np.ptp(fine[s]))) for s in SIGNALS},
                "euler_x1000_vs_rk45_1e-11": {SIGNAL_NAMES[s]: _nrms(sims[(src, fault, "euler_x1000")][s][m], fine[s][m], float(np.ptp(fine[s]))) for s in SIGNALS}}
    worst = max(v for c in cases.values() for v in c["rk45_1e-9_vs_1e-11"].values())
    return {"stage": "rk45_convergence", "decision": "user, 2026-10-25: R2 reference = RK45 (option a)", "reference": "rk45, rtol = atol = 1e-9",
            "finer": "rk45, rtol = atol = 1e-11", "tolerance": CONVERGENCE_TOL, "worst_nrms_rk45_vs_finer": worst, "converged": worst < CONVERGENCE_TOL,
            "normalisation": "RMS(diff over the segment) / range of the 1e-11 solution over the whole run", "cases": cases}


def _rk45_md(r: dict) -> str:
    L = ["## R2.1 (continued) Convergence of the RK45 reference — decision of 2026-10-25 (option a)", "",
         f"Reference: {r['reference']}. Shown converged against {r['finer']} (G7); Euler × 1000 is listed for comparison.", "",
         "| Scenario | Fault | RK45 1e-9 vs 1e-11: i_a / omega / torque | Euler × 1000 vs RK45 1e-11: i_a / omega / torque |", "|---|---|---|---|"]
    for case, c in r["cases"].items():
        a, b = c["rk45_1e-9_vs_1e-11"], c["euler_x1000_vs_rk45_1e-11"]
        L.append(f"| {case.split('|')[0]} | {case.split('|')[1]} | {a['i_a']:.1e} / {a['omega']:.1e} / {a['torque']:.1e} | {b['i_a']:.1e} / {b['omega']:.1e} / {b['torque']:.1e} |")
    L += ["", f"Worst RK45 (1e-9) vs RK45 (1e-11): {r['worst_nrms_rk45_vs_finer']:.1e} (< {r['tolerance']:g}: {'yes' if r['converged'] else 'NO'}). "
          + ("**The RK45 reference is converged; R2 continues with it.**" if r["converged"] else "**Not converged: R2 stops.**"), ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- sensitivity (R2.2/R2.3)
CLASSES5 = ("normal", "outer_race", "inner_race", "ball", "cage")
SETTLE_S = 0.5
SENS_DURATION_S = 2.5
NOISE_RNG_SEED = 25
FIRST_SCENARIO_SEED = 5000
VERSIONS = {"NAT": {}, "REF": {"dc_plant_solver": "rk45"}}
SIG_CHANNELS = ("current_r", "rpm", "torque_nm", "acc_x", "acc_y", "acc_z")
MODELS = {"PC": ("configs/classifiers/pc_server_5class_w2048/2026-10-20_run01", "keras"),
          "RPi5": ("configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01", "tflite"),
          "ESP32": ("configs/classifiers/esp32_dscnn/2026-10-18_run01", "tflite")}
RULE = {"agreement_min": 0.98, "abs_delta_accuracy_max": 0.01, "spectral_rel_diff_max": 0.05}
BAND_HZ, N_HARMONICS = 2.0, 3
FAULT_KEY = {"outer_race": "outer_race", "inner_race": "inner_race", "ball": "ball", "cage": "cage"}


def sens_cells(speeds, severities, n_seeds) -> list:
    rng = np.random.default_rng(NOISE_RNG_SEED)
    cells, idx = [], 0
    for speed in speeds:
        for sev in severities:
            for cls in CLASSES5:
                for k in range(n_seeds):
                    noisy = rng.random() >= 1 / 3
                    e, m = float(rng.uniform(0, 100)), float(rng.uniform(0, 100))
                    cells.append({"cell": idx, "speed": speed, "severity_scale": sev, "fault": None if cls == "normal" else cls, "label": cls, "seed_index": k,
                                  "seed": FIRST_SCENARIO_SEED + idx, "electrical_noise_pct": e if noisy else 0.0, "mechanical_noise_pct": m if noisy else 0.0})
                    idx += 1
    return cells


def _sens_job(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    c, version = job
    kw = {}
    if c["fault"] is not None:
        kw = {"electrical_severity": 8.0 * c["severity_scale"], "mechanical_severity": CALIBRATED_MECHANICAL_SEVERITY[c["fault"]] * c["severity_scale"]}
    sc = Scenario(scenario_id=f"r2_{c['label']}_w{c['speed']:.0f}_s{c['severity_scale']:.2f}_cell{c['cell']}", fault_type=c["fault"], seed=c["seed"],
                  duration_s=SENS_DURATION_S, omega_ref_rad_s=c["speed"], electrical_noise_pct=c["electrical_noise_pct"],
                  mechanical_noise_pct=c["mechanical_noise_pct"], **kw, **VERSIONS[version])
    t0 = time.time()
    df = records_to_dataframe(run_scenario(sc))
    df = df[["timestamp_s", "label", "source_file", "plant_config_id", *SIG_CHANNELS]].copy()
    df["version"], df["cell"] = version, c["cell"]
    return df, time.time() - t0


def _simulate_sens(args, cells):
    import pandas as pd

    if args.cache is not None and args.cache.exists():
        return pd.read_parquet(args.cache), None
    jobs = [(c, v) for c in cells for v in VERSIONS]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        out = pool.map(_sens_job, jobs, chunksize=1)
    data = pd.concat([d for d, _w in out], ignore_index=True)
    if args.cache is not None:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(args.cache)
    return data, {"runs": len(jobs), "median_wall_s_NAT": float(np.median([w for (c, v), (_d, w) in zip(jobs, out) if v == "NAT"])),
                  "median_wall_s_REF": float(np.median([w for (c, v), (_d, w) in zip(jobs, out) if v == "REF"]))}


def _signal_nrms(data, cells):
    rows = {seg: {ch: [] for ch in SIG_CHANNELS} for seg in ("startup", "steady_state")}
    for c in cells:
        nat = data[(data["cell"] == c["cell"]) & (data["version"] == "NAT")].sort_values("timestamp_s")
        ref = data[(data["cell"] == c["cell"]) & (data["version"] == "REF")].sort_values("timestamp_s")
        t = ref["timestamp_s"].to_numpy()
        for seg, m in (("startup", t < SETTLE_S), ("steady_state", t >= SETTLE_S)):
            for ch in SIG_CHANNELS:
                a, b = nat[ch].to_numpy(), ref[ch].to_numpy()
                rows[seg][ch].append(_nrms(a[m], b[m], float(np.ptp(b))))
    return {seg: {ch: {"median": float(np.median(v)), "max": float(np.max(v))} for ch, v in d.items()} for seg, d in rows.items()}


def _spectrum(x, fs):
    x = np.asarray(x, float) - np.mean(x)
    w = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / fs), 2 * np.abs(np.fft.rfft(x * w)) / np.sum(w)


def _spectral(data, cells):
    from driveflow.sim.vibration import KAT_DATACENTER_6203_GEOMETRY
    from driveflow.sim.vibration import bearing_frequencies as bf

    per_run, decision = [], []
    for c in cells:
        sel = {v: data[(data["cell"] == c["cell"]) & (data["version"] == v) & (data["timestamp_s"] >= SETTLE_S)].sort_values("timestamp_s") for v in VERSIONS}
        f, a_nat = _spectrum(sel["NAT"]["current_r"], 1 / 1e-4)
        _, a_ref = _spectrum(sel["REF"]["current_r"], 1 / 1e-4)
        shaft_hz = float(sel["REF"]["rpm"].mean()) / 60.0
        freqs = bf.fault_frequencies_hz(shaft_hz, KAT_DATACENTER_6203_GEOMETRY)
        bands = {}
        for name, f0 in freqs.items():
            for k in range(1, N_HARMONICS + 1):
                m = np.abs(f - k * f0) <= BAND_HZ
                an, ar = float(a_nat[m].max()), float(a_ref[m].max())
                bands[f"{name}_x{k}"] = {"f_hz": k * f0, "A_NAT": an, "A_REF": ar, "rel_diff": abs(an - ar) / ar if ar > 0 else float("inf")}
        own = [v["rel_diff"] for kk, v in bands.items() if c["fault"] and kk.startswith(FAULT_KEY[c["fault"]] + "_x")]
        per_run.append({"cell": c["cell"], "label": c["label"], "speed": c["speed"], "severity_scale": c["severity_scale"], "noisy": c["electrical_noise_pct"] > 0,
                        "own_band_max_rel_diff": max(own) if own else None, "all_bands_max_rel_diff": max(v["rel_diff"] for v in bands.values()), "bands": bands})
        decision += own
    return {"per_run": per_run, "decision_bands_max_rel_diff": float(max(decision)), "decision_bands_median_rel_diff": float(np.median(decision)),
            "all_bands_median_rel_diff": float(np.median([r["all_bands_max_rel_diff"] for r in per_run]))}


def _predict(model_key, X):
    from driveflow.ai.calibration import apply_calibration

    run, kind = MODELS[model_key]
    d = REPO_ROOT / run
    metrics = json.loads((d / "metrics.json").read_text())
    if kind == "keras":
        from driveflow.models.classifiers.builder import build_classifier
        from driveflow.models.classifiers.schemas import load_classifier_config

        m = build_classifier(load_classifier_config(d / "config.yaml"), n_channels=len(metrics["channels"]))
        m.load_weights(d / "model.weights.h5")
        probs = m.predict(X, verbose=0)
    else:
        sys.path.insert(0, str(REPO_ROOT / "experiments"))
        from report_tflite_parity import _tflite_predict

        probs, _ms = _tflite_predict(d / "model.tflite", X)
    return apply_calibration(probs, metrics), metrics


def _windows_paired(data, cells, metrics, window, seg):
    from driveflow.models.common import build_classification_windows

    out = {v: [] for v in VERSIONS}
    labels, cell_of = [], []
    for c in cells:
        per = {}
        for v in VERSIONS:
            df = data[(data["cell"] == c["cell"]) & (data["version"] == v)].sort_values("timestamp_s")
            df = df[df["timestamp_s"] >= SETTLE_S] if seg == "steady_state" else df[df["timestamp_s"] < SETTLE_S]
            X, y, _g = build_classification_windows(df, list(CLASSES5), metrics["channels"], window, cap_per_class=100_000, level_scaling=metrics.get("level_scaling"))
            per[v] = (X, y)
        n = 0 if per["NAT"][0] is None else len(per["NAT"][0])
        assert n == (0 if per["REF"][0] is None else len(per["REF"][0])), "windows not paired"
        if n == 0:
            continue
        for v in VERSIONS:
            out[v].append(per[v][0])
        labels += list(per["NAT"][1])
        cell_of += [c["cell"]] * n
    return {v: np.concatenate(x) for v, x in out.items()}, np.array(labels), np.array(cell_of)


def _classifier_results(data, cells):
    from sklearn.metrics import f1_score

    res = {}
    for key, (run, _kind) in MODELS.items():
        metrics = json.loads((REPO_ROOT / run / "metrics.json").read_text())
        import yaml

        window = int(yaml.safe_load((REPO_ROOT / run / "config.yaml").read_text())["input_window"])
        classes, thr = metrics["classes"], metrics["calibration"]["abstain_below"]
        res[key] = {"run": run, "classes": classes, "window": window}
        for seg in ("startup", "steady_state"):
            X, labels, _cells = _windows_paired(data, cells, metrics, window, seg)
            p = {v: _predict(key, X[v])[0] for v in VERSIONS}
            pred = {v: p[v].argmax(1) for v in VERSIONS}
            known = np.isin(labels, classes)
            y = np.array([classes.index(l) if l in classes else -1 for l in labels])
            r = {"n_windows": int(len(labels)), "n_windows_known_label": int(known.sum()), "agreement": float(np.mean(pred["NAT"] == pred["REF"])),
                 "mean_abs_delta_confidence": float(np.mean(np.abs(p["NAT"].max(1) - p["REF"].max(1))))}
            for v in VERSIONS:
                r[f"accuracy_{v}"] = float(np.mean(pred[v][known] == y[known]))
                r[f"macro_f1_{v}"] = float(f1_score(y[known], pred[v][known], labels=list(range(len(classes))), average="macro", zero_division=0))
                r[f"abstention_{v}"] = float(np.mean(p[v].max(1) < thr))
            r["delta_accuracy"] = r["accuracy_REF"] - r["accuracy_NAT"]
            res[key][seg] = r
    return res


def _svg_lines(path, title, x_label, y_label, series, vlines=(), x_max=None):
    """Minimal SVG line plot (no plotting dependency): series = [(label, x, y, colour, dash)]."""
    W, H, L, R, T, B = 900, 360, 70, 20, 40, 50
    xs = np.concatenate([np.asarray(s[1]) for s in series])
    ys = np.concatenate([np.asarray(s[2]) for s in series])
    x0, x1 = float(xs.min()), float(x_max if x_max is not None else xs.max())
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    y1 = y1 if y1 > y0 else y0 + 1

    def px(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(y):
        return H - B - (y - y0) / (y1 - y0) * (H - T - B)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="white"/>', f'<text x="{L}" y="22" font-size="14">{title}</text>',
           f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#444"/>', f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#444"/>',
           f'<text x="{(W + L) / 2}" y="{H - 12}" text-anchor="middle">{x_label}</text>',
           f'<text x="16" y="{(H) / 2}" transform="rotate(-90 16 {H / 2})" text-anchor="middle">{y_label}</text>']
    for k in range(5):
        xv, yv = x0 + k * (x1 - x0) / 4, y0 + k * (y1 - y0) / 4
        out += [f'<text x="{px(xv)}" y="{H - B + 16}" text-anchor="middle">{xv:.3g}</text>', f'<text x="{L - 6}" y="{py(yv) + 4}" text-anchor="end">{yv:.3g}</text>']
    for label, f_hz, colour in vlines:
        if x0 <= f_hz <= x1:
            out.append(f'<line x1="{px(f_hz):.1f}" y1="{T}" x2="{px(f_hz):.1f}" y2="{H - B}" stroke="{colour}" stroke-dasharray="2,3"/>')
            out.append(f'<text x="{px(f_hz) + 2:.1f}" y="{T + 10}" fill="{colour}" font-size="10">{label}</text>')
    for i, (label, x, y, colour, dash) in enumerate(series):
        x, y = np.asarray(x), np.asarray(y)
        keep = x <= x1
        step = max(1, int(keep.sum() // 2000))
        pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x[keep][::step], y[keep][::step]))
        out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.3" {"stroke-dasharray=" + chr(34) + dash + chr(34) if dash else ""} points="{pts}"/>')
        out.append(f'<text x="{W - R - 160}" y="{T + 14 + 16 * i}" fill="{colour}">{label}</text>')
    out.append("</svg>")
    Path(path).write_text("\n".join(out) + "\n")


def _figures(data, cells, out_dir):
    from driveflow.sim.vibration import KAT_DATACENTER_6203_GEOMETRY
    from driveflow.sim.vibration import bearing_frequencies as bf

    figs = out_dir / "r2_figures"
    figs.mkdir(parents=True, exist_ok=True)
    c = next(c for c in cells if c["label"] == "outer_race" and c["speed"] == 150.0 and c["severity_scale"] == 1.0 and c["electrical_noise_pct"] == 0.0) \
        if any(c["label"] == "outer_race" and c["speed"] == 150.0 and c["severity_scale"] == 1.0 and c["electrical_noise_pct"] == 0.0 for c in cells) \
        else next(c for c in cells if c["label"] == "outer_race")
    d = {v: data[(data["cell"] == c["cell"]) & (data["version"] == v)].sort_values("timestamp_s") for v in VERSIONS}
    for name, (t0, t1) in (("ia_startup", (0.0, 0.005)), ("ia_steady_state", (1.0, 1.05))):
        series = []
        for v, colour, dash in (("REF", "#1f77b4", ""), ("NAT", "#d62728", "5,3")):
            m = (d[v]["timestamp_s"] >= t0) & (d[v]["timestamp_s"] <= t1)
            series.append((f"{v} ({'RK45' if v == 'REF' else 'Euler x1'})", d[v]["timestamp_s"][m], d[v]["current_r"][m], colour, dash))
        _svg_lines(figs / f"{name}.svg", f"i_a, NAT vs REF, {c['label']} {c['speed']:g} rad/s, cell {c['cell']} ({name.split('_', 1)[1].replace('_', ' ')})",
                   "time (s)", "current_r (A)", series)
    spec = []
    for v, colour, dash in (("REF", "#1f77b4", ""), ("NAT", "#d62728", "5,3")):
        dd = d[v][d[v]["timestamp_s"] >= SETTLE_S]
        f, a = _spectrum(dd["current_r"], 1e4)
        spec.append((v, f, a, colour, dash))
    shaft = float(d["REF"][d["REF"]["timestamp_s"] >= SETTLE_S]["rpm"].mean()) / 60
    col = {"outer_race": "#2ca02c", "inner_race": "#9467bd", "ball": "#8c564b", "cage": "#e377c2"}
    vl = [(f"{n[:5]}x{k}", k * f0, col[n]) for n, f0 in bf.fault_frequencies_hz(shaft, KAT_DATACENTER_6203_GEOMETRY).items() for k in (1, 2, 3)]
    _svg_lines(figs / "current_spectrum.svg", f"current_r amplitude spectrum, steady state, cell {c['cell']} ({c['label']}, {c['speed']:g} rad/s)",
               "frequency (Hz)", "amplitude (A)", spec, vlines=vl, x_max=400.0)
    return {"cell": c["cell"], "files": sorted(p.name for p in figs.glob("*.svg"))}


def _verdict(spectral, clf):
    spec_ok = spectral["decision_bands_max_rel_diff"] < RULE["spectral_rel_diff_max"]
    per = {}
    for key, r in clf.items():
        for seg in ("startup", "steady_state"):
            x = r[seg]
            fails = []
            if x["agreement"] < RULE["agreement_min"]:
                fails.append(f"agreement {x['agreement']:.3f} < 0.98")
            if abs(x["delta_accuracy"]) > RULE["abs_delta_accuracy_max"]:
                fails.append(f"|Δ accuracy| {abs(x['delta_accuracy']):.3f} > 0.01")
            if not spec_ok:
                fails.append(f"spectral difference {spectral['decision_bands_max_rel_diff']:.3f} ≥ 0.05")
            per[f"{key}|{seg}"] = {"harmless": not fails, "fails": fails}
    overall = all(v["harmless"] for v in per.values())
    return {"rule": RULE, "spectral_ok": spec_ok, "per_classifier_regime": per,
            "overall": "S-16 harmless for diagnosis" if overall else "S-16 affects diagnosis"}


def sensitivity(args) -> dict:
    speeds = [float(x) for x in args.speeds.split(",")]
    sevs = [float(x) for x in args.severities.split(",")]
    cells = sens_cells(speeds, sevs, args.seeds)
    data, sim = _simulate_sens(args, cells)
    out = {"stage": "sensitivity", "preregistration": "docs/audit/plecs_om/r2_preregistration.md", "versions": {"NAT": "Euler x1 (production)", "REF": "RK45 rtol=atol=1e-9"},
           "grid": {"speeds": speeds, "severities": sevs, "seeds_per_cell": args.seeds, "classes": list(CLASSES5), "n_runs_per_version": len(cells),
                    "duration_s": SENS_DURATION_S, "noise_rng_seed": NOISE_RNG_SEED, "first_scenario_seed": FIRST_SCENARIO_SEED},
           "cells": cells, "simulation": sim}
    out["signal_nrms"] = _signal_nrms(data, cells)
    out["spectral"] = _spectral(data, cells)
    out["classifiers"] = _classifier_results(data, cells)
    out["figures"] = _figures(data, cells, args.out)
    out["verdict"] = _verdict(out["spectral"], out["classifiers"])
    return out


def _sens_md(r: dict) -> str:
    v = r["verdict"]
    L = ["# R2 — Sensitivity of fault diagnosis to the DC plant's discretization (S-16)", "",
         f"Pre-registration: `{r['preregistration']}` (committed before any diagnosis metric). Command: `{r['command']}`. Host: {r['host']}, Python {r['python']}.", "",
         f"NAT = {r['versions']['NAT']}; REF = {r['versions']['REF']}. Grid: speeds {r['grid']['speeds']} rad/s × severities {r['grid']['severities']} × 5 classes × "
         f"{r['grid']['seeds_per_cell']} seeds = {r['grid']['n_runs_per_version']} runs per version, {r['grid']['duration_s']} s each.", "",
         "## Convergence of the reference (R2.1)", "", "See `r2_convergence.md`: Euler substeps do not converge in a load-step scenario (S-17); by the user's decision (option a) "
         "the reference is RK45 (1e-9), within 6.4e-6 NRMS of RK45 at 1e-11 in every scenario (`r2_rk45_convergence.json`).", "",
         f"## Verdict: **{v['overall']}**", "", "| Classifier | Regime | Harmless? | Failed conditions |", "|---|---|---|---|"]
    for k, x in v["per_classifier_regime"].items():
        c, seg = k.split("|")
        L.append(f"| {c} | {seg} | {'yes' if x['harmless'] else 'NO'} | {'; '.join(x['fails']) or '—'} |")
    L += ["", "## 1. Signal NRMS, NAT vs REF (RMS of the difference / REF range over the run)", "", "| Channel | Start-up median / max | Steady state median / max |", "|---|---|---|"]
    for ch in SIG_CHANNELS:
        a, b = r["signal_nrms"]["startup"][ch], r["signal_nrms"]["steady_state"][ch]
        L.append(f"| {ch} | {a['median']:.2e} / {a['max']:.2e} | {b['median']:.2e} / {b['max']:.2e} |")
    s = r["spectral"]
    L += ["", "## 2. current_r spectrum at the fault bands (steady state, ±2 Hz, fundamental and 2 harmonics)", "",
          f"Decision bands (each faulted run's own fault): worst relative difference **{s['decision_bands_max_rel_diff']:.3%}**, median {s['decision_bands_median_rel_diff']:.3%} "
          f"(limit 5 %). All bands of all runs, median of the per-run worst: {s['all_bands_median_rel_diff']:.3%} (reported, not decided on).", "",
          "| Fault | Speed | Severity | Noisy | Own-band worst rel. diff |", "|---|---|---|---|---|"]
    for pr in s["per_run"]:
        if pr["own_band_max_rel_diff"] is not None:
            L.append(f"| {pr['label']} | {pr['speed']:g} | ×{pr['severity_scale']:g} | {'yes' if pr['noisy'] else 'no'} | {pr['own_band_max_rel_diff']:.3%} |")
    L += ["", "## 3–5. Classifiers (paired windows)", "", "| Classifier | Regime | Windows | Agreement | Acc. NAT → REF | Δ acc. | Macro-F1 NAT → REF | Mean |Δ conf.| | Abstention NAT → REF |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, c in r["classifiers"].items():
        for seg in ("startup", "steady_state"):
            x = c[seg]
            L.append(f"| {key} | {seg} | {x['n_windows']} ({x['n_windows_known_label']} with a known label) | {x['agreement']:.3f} | {x['accuracy_NAT']:.3f} → {x['accuracy_REF']:.3f} | "
                     f"{x['delta_accuracy']:+.3f} | {x['macro_f1_NAT']:.3f} → {x['macro_f1_REF']:.3f} | {x['mean_abs_delta_confidence']:.4f} | {x['abstention_NAT']:.3f} → {x['abstention_REF']:.3f} |")
    L += ["", "ESP32: 3 classes (no ball/cage); accuracy and macro-F1 over its own classes, agreement over all windows.", "",
          f"Figures (`r2_figures/`, cell {r['figures']['cell']}): " + ", ".join(f"`{f}`" for f in r["figures"]["files"]) + ".", "",
          "Nothing was regenerated or retrained and no default changed (round §6).", ""]
    return "\n".join(L)


def _offset_job(job):
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen import runner
    from driveflow.validation.benches import LoadTorqueStep

    t_step, overrides = job
    runner.PolynomialStaticLoad = lambda load_parameter: LoadTorqueStep(load_parameter=load_parameter, step_nm=LOAD_STEP_NM, t_step_s=t_step)
    recs = run_scenario(Scenario(scenario_id="r2_offset", seed=0, duration_s=1.1, omega_ref_rad_s=CONV_SPEED, **overrides))
    return np.array([r["current_r"] for r in recs])


def load_step_offset(args) -> dict:
    """Diagnostic of the load-step floor (finding S-17): Euler x1000 with the step at 1.0 s against
    RK45 with the step at 1.0 s, half a period earlier and one period earlier. Metric over
    t > 0.9 s, normalised by the RK45 range of the run."""
    shifts = (0.0, 0.5, 1.0)
    jobs = [(1.0, {"dc_plant_substeps": 1000})] + [(1.0 - k * 1e-4, {"dc_plant_solver": "rk45"}) for k in shifts]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        euler, *rk = pool.map(_offset_job, jobs, chunksize=1)
    m = slice(9000, None)
    return {"stage": "load_step_offset", "euler_x1000_step_at_s": 1.0,
            "nrms_euler_x1000_vs_rk45": {f"rk45 step {k:g} period(s) earlier": _nrms(euler[m], r[m], float(np.ptp(r))) for k, r in zip(shifts, rk)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("convergence", "rk45_convergence", "load_step_offset", "sensitivity"))
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 2) - 1)))
    ap.add_argument("--speeds", default="100,150,200")
    ap.add_argument("--severities", default="1.0,0.5")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"command": "python experiments/report_s16_discretization_sensitivity.py " + " ".join(argv if argv is not None else sys.argv[1:]),
            "host": platform.platform(), "python": platform.python_version()}
    if args.stage == "convergence":
        r = {**meta, **convergence(args)}
        r["total_wall_s"] = round(time.time() - t0, 1)
        (args.out / "r2_convergence.json").write_text(json.dumps(r, indent=2) + "\n")
        (args.out / "r2_convergence.md").write_text(_conv_md(r))
        print(_conv_md(r))
    elif args.stage == "rk45_convergence":
        r = {**meta, **rk45_convergence(args)}
        r["total_wall_s"] = round(time.time() - t0, 1)
        (args.out / "r2_rk45_convergence.json").write_text(json.dumps(r, indent=2) + "\n")
        print(_rk45_md(r))
    elif args.stage == "load_step_offset":
        r = {**meta, **load_step_offset(args)}
        (args.out / "r2_load_step_offset.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))
    else:
        r = {**meta, **sensitivity(args)}
        r["total_wall_s"] = round(time.time() - t0, 1)
        (args.out / "r2_s16_sensitivity.json").write_text(json.dumps(r, indent=2, default=float) + "\n")
        (args.out / "r2_s16_sensitivity.md").write_text(_sens_md(r))
        print(_sens_md(r))


if __name__ == "__main__":
    main()
