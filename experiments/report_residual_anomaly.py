"""Evaluation of the dc_motor forecast-residual anomaly detector -- expansion task E14
(DECISIONES_EXPANSION_2026-10-21.md §5). Writes docs/audit/ai/dc_motor_residual_anomaly.{md,json}.

Both PC variants (with and without voltage_v as an input) are evaluated against the fixed criteria:
1. forecast RMSE vs. persistence, per channel, on healthy windows of held-out conditions: <= 0.8x;
2. AUROC for outer_race and inner_race at severity >= x1 (held-out conditions): >= 0.9 (< 0.7 = negative);
3. false alarms on healthy windows with 100 % noise, plant drift and unseen speeds: <= 1 % per condition;
4. ball, cage and the 1.3x-order fault: AUROC and detection rate at the fixed threshold, no target;
5. start-up and reference steps: false-alarm rates reported separately;
6. edge tiers (if --edge-runs): TFLite vs. Keras parity, host cost, Flex ops.
The threshold is the one stored in each run (99th percentile of healthy validation scores). Nothing
here refits it.

Data:
- held-out healthy and faulted runs come from the classifier evaluation cache
  (experiments/report_classifier_evaluation.py --cache, 1.5 s runs, 4 seeds). Its conditions are not
  in the training grid: 115, 190, 250 rad/s, 100 % noise, plant drift. Caveat (AUDIT_ANSWERS Q15):
  the seed changes only the vibration channels, so in the noise-free conditions the four seeds give
  identical current/rpm. Window counts are not independent samples.
- the 1.3x-order fault comes from the custom-order OOD cache (report_ood_detector.py --orders);
- reference steps are simulated here (PI loop of datagen.runner, reference changed mid-run).
Steady state = windows starting after SETTLE_S; start-up = windows ending before it.

    python experiments/report_residual_anomaly.py --runs <pc run dir> <pc_novv run dir> [--edge-runs <rpi5 dir> <esp32 dir>] [--agent]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from driveflow.ai import residual  # noqa: E402
from driveflow.ai.registry import load_run_model  # noqa: E402
from driveflow.models.common import normalize_window_like_training  # noqa: E402

OUT = ROOT / "docs" / "audit" / "ai"
SETTLE_S = 0.5
HELD_OUT = ("unseen_speed_115_rad_s", "unseen_speed_190_rad_s", "extrapolated_speed_250_rad_s", "noise_100pct", "plant_drift_psi-10pct_J+50pct")
FA_CONDITIONS = HELD_OUT  # criterion 3: noise 100 %, drift, unseen speeds
KNOWN_FAULTS = ("outer_race", "inner_race")
STEPS = ((150.0, 175.0), (150.0, 125.0), (100.0, 200.0))
STEP_T_S, STEP_DURATION_S = 0.75, 1.5


def _segments(df, metrics, steady=True):
    window, horizon = int(metrics["input_window"]), int(metrics["horizon"])
    if steady:
        return residual.raw_windows(df, list(metrics["channels"]), window, horizon, start_s=SETTLE_S)
    return residual.raw_windows(df, list(metrics["channels"]), window, horizon, end_s=SETTLE_S)


def _persistence_ratio(model, seg, metrics) -> dict:
    """Physical-unit forecast RMSE / persistence RMSE per target channel (criterion 1)."""
    window, ch = int(metrics["input_window"]), list(metrics["channels"])
    pred = residual.forecast(model, seg, ch, window, metrics.get("level_scaling"))
    out = {}
    for c in metrics["residual_detector"]["target_channels"]:
        j = ch.index(c)
        fut = seg[:, window:, j]
        model_rmse = float(np.sqrt(np.mean((pred[:, :, j] - fut) ** 2)))
        pers_rmse = float(np.sqrt(np.mean((seg[:, window - 1:window, j] - fut) ** 2)))
        out[c] = {"model_rmse": model_rmse, "persistence_rmse": pers_rmse, "ratio": model_rmse / pers_rmse if pers_rmse > 0 else None}
    return out


def _auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def _step_runs() -> pd.DataFrame:
    """Healthy runs with a speed-reference step at STEP_T_S (PI loop as in datagen.runner)."""
    from driveflow.control.classical import PICascadeController
    from driveflow.datagen.runner import TAU, build_plant
    from driveflow.datagen.scenario import Scenario

    frames = []
    for a, b in STEPS:
        sc = Scenario(scenario_id=f"step_{a:g}_{b:g}", omega_ref_rad_s=a, duration_s=STEP_DURATION_S)
        system = build_plant(sc)
        ctrl = PICascadeController(system)
        system.seed(np.random.SeedSequence(0))
        state = system.reset()
        ctrl.reset()
        rows = []
        for k in range(int(round(STEP_DURATION_S / TAU))):
            ref = a if (k + 1) * TAU < STEP_T_S else b
            state = system.simulate(ctrl.control(state, ref))
            ph = state * system.limits
            rows.append({"timestamp_s": (k + 1) * TAU, "current_r": float(ph[system.CURRENTS_IDX[0]]), "rpm": float(ph[system.OMEGA_IDX]) * 60 / (2 * np.pi),
                         "voltage_v": float(ph[system.VOLTAGES_IDX[0]]), "label": "normal", "source_file": sc.scenario_id})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def evaluate_variant(run_dir: Path, cache: pd.DataFrame, custom: pd.DataFrame, steps: pd.DataFrame) -> dict:
    model, _config, metrics = load_run_model(run_dir, "regressor")
    det = metrics["residual_detector"]
    thr = det["threshold"]
    res = {"run": str(run_dir.relative_to(ROOT)), "channels": metrics["channels"], "threshold": thr, "quantile": det["quantile"],
           "n_val_windows": det["fitted_on"]["n"], "train_time_test_rmse_per_channel": metrics.get("test_rmse_per_channel"),
           "train_time_persistence_rmse_per_channel": metrics.get("persistence_test_rmse_per_channel")}

    def scored(df, steady=True):
        seg, groups, labels, t0 = _segments(df, metrics, steady)
        return seg, labels, (residual.score(model, seg, metrics) if len(seg) else np.array([]))

    # criterion 1 + 3: healthy windows of held-out conditions
    healthy_scores, per_cond, healthy_segs = [], {}, []
    for cond in HELD_OUT:
        sub = cache[(cache.condition == cond) & (cache.label == "normal")]
        seg, _, s = scored(sub)
        healthy_scores.append(s)
        healthy_segs.append(seg)
        per_cond[cond] = {"n_windows": int(len(s)), "false_alarm_rate": float(np.mean(s > thr)) if len(s) else None,
                          "persistence": _persistence_ratio(model, seg, metrics) if len(seg) else None}
    healthy = np.concatenate(healthy_scores)
    pooled = _persistence_ratio(model, np.concatenate(healthy_segs), metrics)
    # Steady-state healthy current/rpm are CONSTANT in the noise-free conditions (rpm in all of them:
    # the noise is added to current and vibration only), so persistence is exact there and the
    # ratio is undefined. Two evaluable readings are added (declared deviation, see the report):
    # whole healthy runs of the same held-out conditions (start-up included), and train_model's own
    # test split (held-out grid runs, normalized units).
    full = np.concatenate([residual.raw_windows(cache[(cache.condition == c) & (cache.label == "normal")], list(metrics["channels"]), int(metrics["input_window"]),
                                                int(metrics["horizon"]))[0] for c in HELD_OUT])
    whole = _persistence_ratio(model, full, metrics)
    tt, tp = metrics.get("test_rmse_per_channel") or {}, metrics.get("persistence_test_rmse_per_channel") or {}
    internal = {c: {"model_rmse": tt.get(c), "persistence_rmse": tp.get(c), "ratio": (tt[c] / tp[c]) if tp.get(c) else None} for c in det["target_channels"]}
    readings = {"steady_state_held_out": pooled, "whole_runs_held_out": whole, "train_time_test_split_normalized": internal}
    evaluable = [v["ratio"] for r in readings.values() for v in r.values() if v["ratio"] is not None]
    res["criterion_1"] = {"pooled": pooled, "readings": readings, "per_condition": {k: v["persistence"] for k, v in per_cond.items()},
                          "steady_state_undefined_channels": [c for c, v in pooled.items() if v["ratio"] is None],
                          "pass": bool(evaluable) and all(r <= 0.8 for r in evaluable)}
    res["criterion_3"] = {"per_condition": {k: {"n_windows": v["n_windows"], "false_alarm_rate": v["false_alarm_rate"]} for k, v in per_cond.items()},
                          "pass": all(v["false_alarm_rate"] is not None and v["false_alarm_rate"] <= 0.01 for v in per_cond.values())}

    # criterion 2: known faults at severity x1 in the same held-out conditions (+ severity x2, in-grid speed)
    c2 = {}
    for fault in KNOWN_FAULTS:
        pos = np.concatenate([scored(cache[(cache.condition == c) & (cache.label == fault)])[2] for c in HELD_OUT])
        per = {c: _auroc(scored(cache[(cache.condition == c) & (cache.label == fault)])[2], per_cond_scores) for c, per_cond_scores in zip(HELD_OUT, healthy_scores)}
        x2 = scored(cache[(cache.condition == "severity_x2") & (cache.label == fault)])[2]
        x2_neg = scored(cache[(cache.condition == "severity_x2") & (cache.label == "normal")])[2]
        x06 = scored(cache[(cache.condition == "unseen_severity_x0.6") & (cache.label == fault)])[2]
        c2[fault] = {"auroc_held_out_x1": _auroc(pos, healthy), "detection_rate_x1": float(np.mean(pos > thr)) if len(pos) else None, "n_pos": int(len(pos)),
                     "auroc_per_condition": per, "auroc_severity_x2_150rad_s": _auroc(x2, x2_neg), "detection_rate_severity_x2": float(np.mean(x2 > thr)) if len(x2) else None,
                     "detection_rate_unseen_severity_x0.6": float(np.mean(x06 > thr)) if len(x06) else None}
    aucs = [c2[f]["auroc_held_out_x1"] for f in KNOWN_FAULTS]
    res["criterion_2"] = {"per_fault": c2, "min_auroc": min(aucs), "pass": min(aucs) >= 0.9, "negative_result": min(aucs) < 0.7}

    # criterion 4: unseen fault types, no target
    c4 = {}
    for fault in ("ball", "cage"):
        pos = np.concatenate([scored(cache[(cache.condition == c) & (cache.label == fault)])[2] for c in HELD_OUT])
        c4[fault] = {"auroc": _auroc(pos, healthy), "detection_rate": float(np.mean(pos > thr)) if len(pos) else None, "n": int(len(pos))}
    pos = scored(custom[custom.label == "unknown_1p3x"])[2]
    c4["order_1.3x"] = {"auroc": _auroc(pos, healthy), "detection_rate": float(np.mean(pos > thr)) if len(pos) else None, "n": int(len(pos)),
                        "note": "runs at 80/125/175/220 rad/s (grid speeds), severity x0.5/1/2; negatives = held-out healthy windows"}
    res["criterion_4"] = c4

    # criterion 5: start-up and reference steps (healthy)
    start = np.concatenate([scored(cache[(cache.condition == c) & (cache.label == "normal")], steady=False)[2] for c in HELD_OUT + ("nominal_steady_state",)])
    seg_all, groups, labels, t0s = residual.raw_windows(steps, list(metrics["channels"]), int(metrics["input_window"]), int(metrics["horizon"]), start_s=SETTLE_S)
    step_scores = residual.score(model, seg_all, metrics) if len(seg_all) else np.array([])
    win_s = (int(metrics["input_window"]) + int(metrics["horizon"])) * 1e-4
    near = (t0s + win_s >= STEP_T_S) & (t0s < STEP_T_S + 0.3)  # overlaps the step, or starts within 0.3 s after it
    res["criterion_5"] = {"startup_false_alarm_rate": float(np.mean(start > thr)) if len(start) else None, "startup_n": int(len(start)),
                          "reference_step_false_alarm_rate": float(np.mean(step_scores[near] > thr)) if near.any() else None, "reference_step_n": int(near.sum()),
                          "steps": [f"{a:g} -> {b:g} rad/s at {STEP_T_S} s" for a, b in STEPS], "window_rule": "windows overlapping the step or starting within 0.3 s after it"}
    return res


def edge_parity(run_dir: Path, cache: pd.DataFrame) -> dict:
    """TFLite vs. Keras on held-out windows (healthy + outer/inner race, severity x1): forecast
    difference, the detector's decision agreement, host latency, Flex ops."""
    from report_tflite_parity import _op_names, _tflite_predict

    model, _config, metrics = load_run_model(run_dir, "regressor")
    tfl = run_dir / "model.tflite"
    if not tfl.exists():
        return {"run": str(run_dir.relative_to(ROOT)), "exported": False}
    ch, window = list(metrics["channels"]), int(metrics["input_window"])
    sub = cache[(cache.condition.isin(HELD_OUT)) & (cache.label.isin(("normal",) + KNOWN_FAULTS))]
    seg, _, labels, _ = residual.raw_windows(sub, ch, window, int(metrics["horizon"]), cap_per_group=8, start_s=SETTLE_S)
    norm = [normalize_window_like_training(x[:window], ch) for x in seg]
    ctx = np.stack([n[0] for n in norm])
    mus, sigmas = np.stack([n[1] for n in norm]), np.stack([n[2] for n in norm])
    keras_pred = np.asarray(model(ctx, training=False))
    tfl_pred, host_ms = _tflite_predict(tfl, ctx)
    det = metrics["residual_detector"]

    def _scores(pred_norm):
        phys = pred_norm * sigmas[:, None, :] + mus[:, None, :]
        return residual.scores_from_rmse(residual.channel_rmse(seg, phys, ch, det["target_channels"], window), det)

    ks, ts = _scores(keras_pred), _scores(tfl_pred)
    ops = _op_names(tfl)
    return {"run": str(run_dir.relative_to(ROOT)), "exported": True, "tier": metrics.get("tier"), "n_windows": int(len(seg)),
            "max_abs_diff_normalized": float(np.max(np.abs(keras_pred - tfl_pred))), "rmse_diff_normalized": float(np.sqrt(np.mean((keras_pred - tfl_pred) ** 2))),
            "decision_agreement": float(np.mean((ks > det["threshold"]) == (ts > det["threshold"]))), "host_ms_per_window_median": host_ms,
            "flex_ops": [o for o in ops if o.startswith("Flex")], "size_kb": tfl.stat().st_size / 1024,
            "keras_detection_rate_faulty": float(np.mean(ks[labels != "normal"] > det["threshold"])), "tflite_detection_rate_faulty": float(np.mean(ts[labels != "normal"] > det["threshold"]))}


def agent_comparison(cache: pd.DataFrame, variants: dict) -> dict:
    """Run-level: the simulation-based agent's score vs. the residual detector's mean window score,
    on one run per (held-out condition, healthy/outer/inner)."""
    from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector

    speeds = {"unseen_speed_115_rad_s": 115.0, "unseen_speed_190_rad_s": 190.0, "extrapolated_speed_250_rad_s": 250.0, "noise_100pct": 150.0, "plant_drift_psi-10pct_J+50pct": 150.0}
    agent = DCMotorAnomalyDetector()
    rows = []
    for cond in HELD_OUT:
        for lab in ("normal",) + KNOWN_FAULTS:
            sub = cache[(cache.condition == cond) & (cache.label == lab)]
            if sub.empty:
                continue
            run = sub[sub.source_file == sub.source_file.iloc[0]]
            t0 = time.perf_counter()
            tele = {f: run[f].to_numpy(dtype=float) for f in agent.monitored_features if f in run.columns}
            a_score, closest = agent.detect_anomaly(tele, operating_point={"omega_ref_rad_s": speeds[cond]})
            row = {"condition": cond, "label": lab, "agent_score": a_score, "agent_closest": closest, "agent_s": round(time.perf_counter() - t0, 2)}
            for name, (model, metrics) in variants.items():
                seg, _, _, _ = residual.raw_windows(run, list(metrics["channels"]), int(metrics["input_window"]), int(metrics["horizon"]), start_s=SETTLE_S)
                s = residual.score(model, seg, metrics)
                row[f"{name}_mean_score"] = float(np.mean(s))
                row[f"{name}_flagged_share"] = float(np.mean(s > metrics["residual_detector"]["threshold"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    out = {"runs": rows}
    y = (df.label != "normal").to_numpy()
    out["run_level_auroc"] = {"agent": float(roc_auc_score(y, df.agent_score))}
    for name in variants:
        out["run_level_auroc"][name] = float(roc_auc_score(y, df[f"{name}_mean_score"]))
    out["agent_alert_threshold"] = 0.7
    out["agent_alert_rate_healthy"] = float(np.mean(df[~y].agent_score >= 0.7))
    out["agent_alert_rate_faulty"] = float(np.mean(df[y].agent_score >= 0.7))
    return out


def _fmt(x, spec=".3f"):
    return "—" if x is None else format(x, spec)


def render_md(rep: dict) -> str:
    L = ["# dc_motor forecast-residual anomaly detector (E14)", "",
         f"Generated by `{rep['meta']['command']}` on {rep['meta']['date']} ({rep['meta']['machine']}). Raw values: [`dc_motor_residual_anomaly.json`](dc_motor_residual_anomaly.json).", "",
         "Simulated windows only. Every number below is measured on simulated windows of conditions excluded from training. "
         "The seed only changes the vibration channels, so the four seeds of a noise-free condition give identical current/rpm windows (AUDIT_ANSWERS Q15).", "",
         "## Design (fixed before looking at faulted windows)", "",
         f"- Highest fault frequency evaluated: **{rep['design']['highest_fault_hz']:.1f} Hz** (inner race at 250 rad/s); half period **{rep['design']['half_period_samples']:.1f} samples** at 10 kHz. "
         f"Horizon **{rep['design']['horizon']} samples** ({rep['design']['horizon'] / 10:.1f} ms). No decimation (the 4th ripple harmonic, {4 * rep['design']['highest_fault_hz']:.0f} Hz, is far below the 5 kHz Nyquist).",
         f"- Per-speed fault frequencies (Hz): see the JSON (`design.fault_frequencies_hz`).",
         "- Score: mean over current_r and rpm of RMSE(physical residual over the horizon) / median healthy-validation RMSE. Threshold: 99th percentile of healthy validation scores.", ""]
    for name, v in rep["variants"].items():
        c1, c2, c3, c4, c5 = (v[f"criterion_{i}"] for i in range(1, 6))
        L += [f"## Variant `{name}` — inputs {', '.join(v['channels'])}", "", f"Run `{v['run']}`; threshold {v['threshold']:.3f} (q{v['quantile']:.2f} of {v['n_val_windows']} healthy validation windows).", "",
              "| # | Criterion | Target | Result | Status |", "|---|---|---|---|---|",
              "| 1 | Forecast RMSE / persistence RMSE per channel: steady-state held-out · whole held-out runs · train-time test (normalized) | ≤ 0.8 per channel | "
              + " · ".join(", ".join(f"{k} {_fmt(x['ratio'])}" for k, x in r.items()) for r in c1["readings"].values()) + f" | {'PASS*' if c1['pass'] else 'FAIL'} |",
              "| 2 | AUROC outer / inner race, severity ×1, held-out conditions | ≥ 0.9 (< 0.7 negative) | " + " / ".join(_fmt(c2["per_fault"][f]["auroc_held_out_x1"]) for f in KNOWN_FAULTS)
              + f" | {'PASS' if c2['pass'] else ('NEGATIVE' if c2['negative_result'] else 'FAIL')} |",
              "| 3 | False alarms on healthy windows | ≤ 1 % per condition | " + ", ".join(f"{k}: {_fmt(x['false_alarm_rate'], '.1%')}" for k, x in c3["per_condition"].items()) + f" | {'PASS' if c3['pass'] else 'FAIL'} |",
              "| 4 | Ball / cage / 1.3× order (AUROC; detection rate) | none | " + "; ".join(f"{k}: {_fmt(x['auroc'])}, {_fmt(x['detection_rate'], '.1%')}" for k, x in c4.items()) + " | reported |",
              f"| 5 | False alarms: start-up / reference steps | report | {_fmt(c5['startup_false_alarm_rate'], '.1%')} (n={c5['startup_n']}) / {_fmt(c5['reference_step_false_alarm_rate'], '.1%')} (n={c5['reference_step_n']}) | reported |", "",
              "\\* Criterion 1 deviation (declared): on steady-state healthy windows of the noise-free held-out conditions current_r and rpm are constant, so persistence is exact and the "
              "ratio is undefined (—); rpm is constant in every held-out condition because the noise is added to current and vibration only. PASS means every EVALUABLE reading is ≤ 0.8. "
              "This evaluation rule was added after seeing the undefined steady-state ratios; no faulted window was involved.", "",
              "Per condition (criterion 1 ratios current_r / rpm; criterion 3 false alarms; criterion 2 AUROC outer / inner):", "",
              "| Condition | RMSE ratio current_r | RMSE ratio rpm | False alarms | AUROC outer | AUROC inner |", "|---|---|---|---|---|---|"]
        for cond in HELD_OUT:
            p = c1["per_condition"][cond] or {}
            L.append(f"| {cond} | {_fmt((p.get('current_r') or {}).get('ratio'))} | {_fmt((p.get('rpm') or {}).get('ratio'))} | {_fmt(c3['per_condition'][cond]['false_alarm_rate'], '.1%')} | "
                     f"{_fmt(c2['per_fault']['outer_race']['auroc_per_condition'][cond])} | {_fmt(c2['per_fault']['inner_race']['auroc_per_condition'][cond])} |")
        L += ["", f"Detection at the fixed threshold, severity ×1: outer {_fmt(c2['per_fault']['outer_race']['detection_rate_x1'], '.1%')}, inner {_fmt(c2['per_fault']['inner_race']['detection_rate_x1'], '.1%')}; "
              f"severity ×0.6: outer {_fmt(c2['per_fault']['outer_race']['detection_rate_unseen_severity_x0.6'], '.1%')}, inner {_fmt(c2['per_fault']['inner_race']['detection_rate_unseen_severity_x0.6'], '.1%')}; "
              f"severity ×2 at 150 rad/s (an in-grid speed): AUROC outer {_fmt(c2['per_fault']['outer_race']['auroc_severity_x2_150rad_s'])}, inner {_fmt(c2['per_fault']['inner_race']['auroc_severity_x2_150rad_s'])}.", ""]
    if rep.get("selection"):
        L += ["## Variant for the GUI", "", rep["selection"], ""]
    if rep.get("agent"):
        a = rep["agent"]
        L += ["## Comparison with the simulation-based agent (same runs)", "", "Run level, one run per (held-out condition, healthy / outer / inner): AUROC of the agent's score and of each variant's mean window score.", "",
              "| Detector | Run-level AUROC |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in a["run_level_auroc"].items()]
        L += ["", f"Agent at its 0.7 alert threshold: alerts on {_fmt(a['agent_alert_rate_healthy'], '.0%')} of healthy runs and {_fmt(a['agent_alert_rate_faulty'], '.0%')} of faulted runs.", ""]
    if rep.get("edge"):
        L += ["## Criterion 6 — edge tiers (TFLite vs. Keras of the same run)", "",
              "| Run | Max abs diff (normalized) | Decision agreement | Detection faulty Keras / TFLite | Host ms / window (median) | Flex ops | Size (KB) |", "|---|---|---|---|---|---|---|"]
        for e in rep["edge"]:
            if not e["exported"]:
                L.append(f"| `{e['run']}` | not exported | | | | | |")
                continue
            L.append(f"| `{e['run']}` | {_fmt(e['max_abs_diff_normalized'], '.2e')} | {_fmt(e['decision_agreement'], '.1%')} | {_fmt(e['keras_detection_rate_faulty'], '.1%')} / {_fmt(e['tflite_detection_rate_faulty'], '.1%')} | "
                     f"{_fmt(e['host_ms_per_window_median'], '.3f')} | {e['flex_ops'] or 'none'} | {_fmt(e['size_kb'], '.1f')} |")
        L.append("")
    return "\n".join(L) + "\n"


def design_block(horizon: int) -> dict:
    from driveflow.sim.vibration import bearing_frequencies as bf

    g = bf.KAT_DATACENTER_6203_GEOMETRY
    speeds = (80.0, 100.0, 115.0, 125.0, 150.0, 175.0, 190.0, 200.0, 220.0, 250.0)
    freqs = {f"{w:g}": {k: bf.fault_order(k, g) * w / (2 * np.pi) for k in bf.FAULT_FREQUENCY_FUNCS} for w in speeds}
    top = max(max(v.values()) for v in freqs.values())
    return {"fault_frequencies_hz": freqs, "highest_fault_hz": top, "half_period_samples": 0.5 / top * 1e4, "horizon": horizon,
            "ripple_harmonics": "fundamental + 3 overtones (datagen/fault_injection._HARMONIC_WEIGHTS)"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="PC run dirs: the voltage_v variant first, then the one without")
    ap.add_argument("--names", nargs="+", default=["with_voltage_v", "without_voltage_v"])
    ap.add_argument("--edge-runs", nargs="*", type=Path, default=[])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "eval_cache_5class.parquet")
    ap.add_argument("--custom-cache", type=Path, default=ROOT / "data" / "ood_eval_cache_custom.parquet")
    ap.add_argument("--agent", action="store_true", help="also run the simulation-based agent on the same runs (several minutes)")
    ap.add_argument("--selection", default=None, help="text recorded as the GUI-variant decision")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    cache = pd.read_parquet(args.cache)
    custom = pd.read_parquet(args.custom_cache)
    steps = _step_runs()
    rep = {"meta": {"command": "python experiments/report_residual_anomaly.py " + " ".join(sys.argv[1:] if argv is None else argv), "date": time.strftime("%Y-%m-%d"),
                    "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}", "cache": str(args.cache.relative_to(ROOT)) if args.cache.is_relative_to(ROOT) else str(args.cache),
                    "held_out_conditions": list(HELD_OUT), "settle_s": SETTLE_S},
           "variants": {}}
    for name, run in zip(args.names, args.runs):
        rep["variants"][name] = evaluate_variant(run.resolve(), cache, custom, steps)
    first = load_run_model(args.runs[0].resolve(), "regressor")[2]
    rep["design"] = design_block(int(first["horizon"]))
    if args.selection:
        rep["selection"] = args.selection
    if args.agent:
        variants = {}
        for name, run in zip(args.names, args.runs):
            m, _c, met = load_run_model(run.resolve(), "regressor")
            variants[name] = (m, met)
        rep["agent"] = agent_comparison(cache, variants)
    if args.edge_runs:
        rep["edge"] = [edge_parity(r.resolve(), cache) for r in args.edge_runs]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dc_motor_residual_anomaly.json").write_text(json.dumps(rep, indent=1, default=float))
    (args.out / "dc_motor_residual_anomaly.md").write_text(render_md(rep))
    print(render_md(rep))


if __name__ == "__main__":
    main()
