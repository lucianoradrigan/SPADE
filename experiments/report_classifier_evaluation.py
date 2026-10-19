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


def generate(seeds: list, duration_s: float, workers: int) -> pd.DataFrame:
    jobs = []
    for cond, (overrides, mech_scale, _win) in CONDITIONS.items():
        if cond in _SHARES_NOMINAL:
            continue
        for fault in CLASSES_FAULTS:
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


def _windows(df: pd.DataFrame, classes: list, channels: list, window: int):
    from driveflow.models.common import build_classification_windows

    return build_classification_windows(df, classes, channels, window, cap_per_class=10_000)


def evaluate_run(run_dir: Path, data: pd.DataFrame, window: int = 512) -> dict:
    from driveflow.ai.evaluation import classification_metrics, ood_confidence_summary, reliability_curve

    model, metrics = _load_run(run_dir)
    classes, channels = metrics["classes"], metrics["channels"]
    out = {"run": str(run_dir.relative_to(REPO_ROOT)), "classes": classes, "channels": channels, "recorded": {k: metrics.get(k) for k in ("test_accuracy", "test_macro_f1", "n_test")}, "conditions": {}}
    for cond, (_o, _s, win) in CONDITIONS.items():
        src = _SHARES_NOMINAL.get(cond, cond)
        df = data[data["condition"] == src]
        df = df[df["timestamp_s"] >= SETTLE_S] if win == "steady" else df[df["timestamp_s"] < SETTLE_S]
        X, y_str, _g = _windows(df, classes, channels, window)
        y = np.array([classes.index(c) for c in y_str])
        probs = model.predict(X, verbose=0)
        m = classification_metrics(y, probs, classes)
        m["reliability"] = reliability_curve(probs, y)
        out["conditions"][cond] = m
    ood = {}
    df_ood = data[(data["condition"] == "ood_unseen_fault_type") & (data["timestamp_s"] >= SETTLE_S)]
    for fault in OOD_FAULTS:
        X, _y, _g = _windows(df_ood, [fault], channels, window)
        ood[fault] = ood_confidence_summary(model.predict(X, verbose=0), classes)
    out["ood_unseen_fault_types"] = ood
    return out


def _md(report: dict) -> str:
    lines = []
    for r in report["models"]:
        lines.append(f"### `{r['run']}`\n")
        rec = r["recorded"]
        lines.append(f"Recorded at training time: test accuracy {rec['test_accuracy']:.3f}, macro-F1 {rec['test_macro_f1']:.3f} (n_test = {rec['n_test']} windows).\n")
        cls = r["classes"]
        lines.append("| Condition | n | Accuracy | Balanced acc. | Macro-F1 | " + " | ".join(f"Recall {c}" for c in cls) + " | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect |")
        lines.append("|---" * (8 + len(cls)) + "|")
        for cond, m in r["conditions"].items():
            fd = m["fault_detection"]
            cc = m["mean_confidence_correct"]
            ci = m["mean_confidence_incorrect"]
            lines.append(
                f"| {cond} | {m['n']} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | "
                + " | ".join(f"{m['per_class'][c]['recall']:.2f}" for c in cls)
                + f" | {fd['false_alarm_rate']:.3f} | {fd['missed_detection_rate']:.3f} | {m['ece']:.3f} | "
                + f"{'-' if cc is None else f'{cc:.2f}'} / {'-' if ci is None else f'{ci:.2f}'} |"
            )
        lines.append("")
        lines.append("Unseen fault types (no class in the model's vocabulary):\n")
        lines.append("| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 |")
        lines.append("|---|---|---|---|---|")
        for fault, o in r["ood_unseen_fault_types"].items():
            share = ", ".join(f"{c} {v:.2f}" for c, v in o["predicted_share"].items())
            lines.append(f"| {fault} | {o['n']} | {share} | {o['mean_confidence']:.3f} | {o['share_confidence_above_0.9']:.2f} |")
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
    ap.add_argument("--run-dir", type=Path, action="append", default=None, help="model run(s) to evaluate; default: promoted pc classifier + 2026-10-13 candidate")
    args = ap.parse_args(argv)

    t0 = time.time()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    if args.cache is not None and args.cache.exists():
        data = pd.read_parquet(args.cache)
    else:
        data = generate(seeds, args.duration_s, args.workers)
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(args.cache)
    t_sim = time.time() - t0

    from driveflow.ai.registry import resolve

    runs = args.run_dir or [resolve("dc_motor", "pc", "classifier"), REPO_ROOT / "configs" / "classifiers" / "pc_server" / "2026-10-13_run01"]
    models = [evaluate_run(Path(r).resolve(), data) for r in runs]
    report = {
        "command": "python experiments/report_classifier_evaluation.py " + " ".join(argv or []),
        "seeds": seeds, "duration_s": args.duration_s, "settle_s": SETTLE_S, "window_samples": 512,
        "conditions": {k: {"overrides": v[0], "mechanical_severity_scale": v[1], "window": v[2]} for k, v in CONDITIONS.items()},
        "simulation_wall_time_s": round(t_sim, 1), "total_wall_time_s": round(time.time() - t0, 1),
        "models": models,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "classifier_evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    (args.out_dir / "classifier_evaluation.md").write_text(_md(report))
    print(_md(report))
    print(f"simulation {t_sim:.0f} s, total {time.time() - t0:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
