"""Extended evaluation of the dc_motor PC classifiers (docs/AI_TECHNICAL_AUDIT.md Sec. 2.2/2.3).

Goes beyond experiments/report_classifier_generalization.py (accuracy/macro-F1 at ONE speed and
ONE severity) by measuring, on freshly simulated runs that were never used for training:
  - balanced accuracy, per-class precision/recall/F1, one-vs-rest FPR/FNR, confusion matrix;
  - fault-detection view: false-alarm rate (healthy -> any fault) and missed-detection rate;
  - calibration: ECE (10 bins), Brier score, confidence on correct vs. incorrect windows;
  - generalization under shifted conditions: speed, fault severity, sensor noise, plant drift,
    start-up transient;
  - behaviour on fault types the model has no class for (ball, cage): what it answers, how sure.

Nothing is re-trained and no recorded metric is modified. Windows use the training pipeline's own
builder (build_classification_windows) so the preprocessing is identical to training.

Caveat that bounds every number here (AUDIT_ANSWERS Q15): the scenario seed only changes the
vibration channels; current/rpm/torque are identical across seeds of the same condition. The
"n runs" per condition are therefore NOT independent operating conditions, and no confidence
interval over runs is reported.

Usage:
  python experiments/report_classifier_evaluation.py [--seeds 4] [--duration-s 1.5] [--workers 8]
      [--cache data_cache.parquet] [--out-dir docs/audit/ai]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES_FAULTS = (None, "outer_race", "inner_race")
OOD_FAULTS = ("ball", "cage")
SETTLE_S = 0.5

#: condition name -> (Scenario overrides, mechanical-severity scale, time window)
CONDITIONS = {
    "nominal_steady_state": ({}, 1.0, "steady"),
    "startup_transient": ({}, 1.0, "transient"),
    "speed_100_rad_s": ({"omega_ref_rad_s": 100.0}, 1.0, "steady"),
    "speed_200_rad_s": ({"omega_ref_rad_s": 200.0}, 1.0, "steady"),
    "severity_x0.5": ({"electrical_severity": 4.0}, 0.5, "steady"),
    "severity_x2": ({"electrical_severity": 16.0}, 2.0, "steady"),
    "noise_20pct": ({"electrical_noise_pct": 20.0, "mechanical_noise_pct": 20.0}, 1.0, "steady"),
    "noise_100pct": ({"electrical_noise_pct": 100.0, "mechanical_noise_pct": 100.0}, 1.0, "steady"),
    "plant_drift_psi-10pct_J+50pct": ({"motor_parameter_overrides": {"psi_e": 0.165 * 0.9, "j_rotor": 0.025 * 1.5}}, 1.0, "steady"),
    # Held out of experiments/generate_diagnosis_grid_dataset.py's grid (AI-N07): two speeds between
    # its levels, one beyond its 220 rad/s maximum, and a severity between its 0.5 and 0.75 levels.
    "unseen_speed_115_rad_s": ({"omega_ref_rad_s": 115.0}, 1.0, "steady"),
    "unseen_speed_190_rad_s": ({"omega_ref_rad_s": 190.0}, 1.0, "steady"),
    "extrapolated_speed_250_rad_s": ({"omega_ref_rad_s": 250.0}, 1.0, "steady"),
    "unseen_severity_x0.6": ({"electrical_severity": 4.8}, 0.6, "steady"),
}
#: conditions that share the nominal simulation (only the time window differs)
_SHARES_NOMINAL = {"startup_transient": "nominal_steady_state"}


def _simulate(job: tuple) -> pd.DataFrame:
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    cond, fault, seed, overrides, mech_scale, duration_s = job
    kw = dict(overrides)
    if fault is not None:
        kw["mechanical_severity"] = CALIBRATED_MECHANICAL_SEVERITY[fault] * mech_scale
    sid = f"eval_{cond}_{fault or 'normal'}_seed{seed}"
    df = records_to_dataframe(run_scenario(Scenario(scenario_id=sid, fault_type=fault, seed=seed, duration_s=duration_s, **kw)))
    df["condition"] = cond
    return df


def generate(seeds: list, duration_s: float, workers: int, extra_faults: tuple = ()) -> pd.DataFrame:
    jobs = []
    for cond, (overrides, mech_scale, _win) in CONDITIONS.items():
        if cond in _SHARES_NOMINAL:
            continue
        for fault in CLASSES_FAULTS + tuple(extra_faults):
            for seed in seeds:
                jobs.append((cond, fault, seed, overrides, mech_scale, duration_s))
    for fault in OOD_FAULTS:
        for seed in seeds:
            jobs.append(("ood_unseen_fault_type", fault, seed, {}, 1.0, duration_s))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        frames = pool.map(_simulate, jobs)
    return pd.concat(frames, ignore_index=True)


def _load_run(run_dir: Path):
    from driveflow.models.classifiers.builder import build_classifier
    from driveflow.models.classifiers.schemas import load_classifier_config

    metrics = json.loads((run_dir / "metrics.json").read_text())
    model = build_classifier(load_classifier_config(run_dir / "config.yaml"), n_channels=len(metrics["channels"]))
    model.load_weights(run_dir / "model.weights.h5")
    return model, metrics


def _windows(df: pd.DataFrame, classes: list, channels: list, window: int, level_scaling: dict | None = None):
    from driveflow.models.common import build_classification_windows

    return build_classification_windows(df, classes, channels, window, cap_per_class=10_000, level_scaling=level_scaling)


def _abstention(probs_cal: np.ndarray, y: np.ndarray | None, metrics: dict) -> dict | None:
    """Share of windows withheld by the run's abstention threshold, and accuracy on the rest."""
    cal = metrics.get("calibration")
    if not cal:
        return None
    keep = probs_cal.max(axis=1) >= cal["abstain_below"]
    out = {"threshold": cal["abstain_below"], "abstained_share": float(1 - keep.mean())}
    if y is not None:
        pred = probs_cal.argmax(axis=1)
        out["accuracy_on_answered"] = float(np.mean(pred[keep] == y[keep])) if keep.any() else None
        wrong = pred != y
        out["share_of_errors_withheld"] = float(np.mean(~keep[wrong])) if wrong.any() else None
    return out


def evaluate_run(run_dir: Path, data: pd.DataFrame, window: int = 512) -> dict:
    from driveflow.ai.calibration import apply_calibration
    from driveflow.ai.evaluation import classification_metrics, ood_confidence_summary, reliability_curve

    model, metrics = _load_run(run_dir)
    classes, channels = metrics["classes"], metrics["channels"]
    ls = metrics.get("level_scaling")
    out = {"run": str(run_dir.relative_to(REPO_ROOT)), "classes": classes, "channels": channels, "level_scaling": ls, "calibration": metrics.get("calibration"),
           "recorded": {k: metrics.get(k) for k in ("test_accuracy", "test_macro_f1", "n_test")}, "conditions": {}}
    for cond, (_o, _s, win) in CONDITIONS.items():
        src = _SHARES_NOMINAL.get(cond, cond)
        df = data[data["condition"] == src]
        df = df[df["timestamp_s"] >= SETTLE_S] if win == "steady" else df[df["timestamp_s"] < SETTLE_S]
        X, y_str, _g = _windows(df, classes, channels, window, ls)
        y = np.array([classes.index(c) for c in y_str])
        # Metrics on the probabilities the GUI shows: calibrated if the run has a calibration.
        probs = apply_calibration(model.predict(X, verbose=0), metrics)
        m = classification_metrics(y, probs, classes)
        m["reliability"] = reliability_curve(probs, y)
        m["abstention"] = _abstention(probs, y, metrics)
        out["conditions"][cond] = m
    ood = {}
    df_ood = data[(data["condition"] == "ood_unseen_fault_type") & (data["timestamp_s"] >= SETTLE_S)]
    for fault in OOD_FAULTS:
        X, _y, _g = _windows(df_ood, [fault], channels, window, ls)
        probs = apply_calibration(model.predict(X, verbose=0), metrics)
        ood[fault] = ood_confidence_summary(probs, classes)
        ood[fault]["abstention"] = _abstention(probs, None, metrics)
    out["ood_unseen_fault_types"] = ood
    return out


def _fmt_abst(a: dict | None) -> str:
    if not a:
        return "-"
    acc = "-" if a.get("accuracy_on_answered") is None else f"{a['accuracy_on_answered']:.3f}"
    wh = "-" if a.get("share_of_errors_withheld") is None else f"{a['share_of_errors_withheld']:.2f}"
    return f"{a['abstained_share']:.2f} / {acc} / {wh}"


def _md(report: dict) -> str:
    lines = []
    for r in report["models"]:
        lines.append(f"### `{r['run']}`\n")
        rec = r["recorded"]
        lines.append(f"Recorded at training time: test accuracy {rec['test_accuracy']:.3f}, macro-F1 {rec['test_macro_f1']:.3f} (n_test = {rec['n_test']} windows).\n")
        cal = r.get("calibration")
        lines.append(f"Preprocessing: rpm level {'kept (' + str(r['level_scaling']) + ')' if r.get('level_scaling') else 'z-scored per window'}. "
                     + (f"Calibration: temperature T = {cal['temperature']:.3f}, abstain below {cal['abstain_below']:.3f} (fitted on validation). Metrics below use calibrated probabilities.\n" if cal else "Uncalibrated softmax.\n"))
        cls = r["classes"]
        lines.append("| Condition | n | Accuracy | Balanced acc. | Macro-F1 | " + " | ".join(f"Recall {c}" for c in cls) + " | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |")
        lines.append("|---" * (9 + len(cls)) + "|")
        for cond, m in r["conditions"].items():
            fd = m["fault_detection"]
            cc = m["mean_confidence_correct"]
            ci = m["mean_confidence_incorrect"]
            lines.append(
                f"| {cond} | {m['n']} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | "
                + " | ".join(f"{m['per_class'][c]['recall']:.2f}" for c in cls)
                + f" | {fd['false_alarm_rate']:.3f} | {fd['missed_detection_rate']:.3f} | {m['ece']:.3f} | "
                + f"{'-' if cc is None else f'{cc:.2f}'} / {'-' if ci is None else f'{ci:.2f}'} | {_fmt_abst(m.get('abstention'))} |"
            )
        lines.append("")
        lines.append("Unseen fault types (no class in the model's vocabulary):\n")
        lines.append("| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |")
        lines.append("|---|---|---|---|---|---|")
        for fault, o in r["ood_unseen_fault_types"].items():
            share = ", ".join(f"{c} {v:.2f}" for c, v in o["predicted_share"].items())
            ab = o.get("abstention")
            lines.append(f"| {fault} | {o['n']} | {share} | {o['mean_confidence']:.3f} | {o['share_confidence_above_0.9']:.2f} | {'-' if not ab else round(ab['abstained_share'], 2)} |")
        lines.append("")
        cm = r["conditions"]["nominal_steady_state"]["confusion_matrix_rows_true"]
        lines.append(f"Confusion matrix, nominal steady state (rows = true {cls}, columns = predicted): `{cm}`\n")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=200)
    ap.add_argument("--duration-s", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 2) - 1)))
    ap.add_argument("--cache", type=Path, default=None, help="parquet to reuse/save the simulated runs")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "audit" / "ai")
    ap.add_argument("--extra-faults", default="", help="comma-separated fault types simulated in every condition besides healthy/outer/inner (A2: ball,cage)")
    ap.add_argument("--out-stem", default="classifier_evaluation", help="output file name stem in --out-dir")
    ap.add_argument("--run-dir", type=Path, action="append", default=None, help="model run(s) to evaluate; default: promoted pc classifier, its no-rpm-level ablation, the 2026-10-13 candidate and the pre-2026-10-18 production run")
    args = ap.parse_args(argv)

    t0 = time.time()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    if args.cache is not None and args.cache.exists():
        data = pd.read_parquet(args.cache)
    else:
        data = generate(seeds, args.duration_s, args.workers, tuple(f for f in args.extra_faults.split(",") if f))
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(args.cache)
    t_sim = time.time() - t0

    from driveflow.ai.registry import resolve

    pc = REPO_ROOT / "configs" / "classifiers" / "pc_server"
    # The production model, the ablation without the rpm level (same grid), the single-operating-
    # point candidate of 2026-10-13 and the production model before 2026-10-18 (AI-N06).
    runs = args.run_dir or [resolve("dc_motor", "pc", "classifier"), pc / "2026-10-18_run02", pc / "2026-10-13_run01", pc / "2026-09-01_run01"]
    models = [evaluate_run(Path(r).resolve(), data) for r in runs]
    report = {
        "command": "python experiments/report_classifier_evaluation.py " + " ".join(argv or []), "extra_faults": args.extra_faults,
        "seeds": seeds, "duration_s": args.duration_s, "settle_s": SETTLE_S, "window_samples": 512,
        "conditions": {k: {"overrides": v[0], "mechanical_severity_scale": v[1], "window": v[2]} for k, v in CONDITIONS.items()},
        "simulation_wall_time_s": round(t_sim, 1), "total_wall_time_s": round(time.time() - t0, 1),
        "models": models,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.out_stem}.json").write_text(json.dumps(report, indent=2, default=str))
    (args.out_dir / f"{args.out_stem}.md").write_text(_md(report))
    print(_md(report))
    print(f"simulation {t_sim:.0f} s, total {time.time() - t0:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
