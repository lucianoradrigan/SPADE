"""Evaluate the OOD detector (ai/ood.py) on the promoted dc_motor PC classifier -- TODO A1.

Fitting (in-distribution only, anti-leakage rule of TODO A1):
- the run's own training and validation windows are rebuilt from the dataset recorded in its
  run_metadata.json, with the same windowing, cap per class, level scaling and grouped split
  (the split's group hashes are checked against metrics.json);
- class means and the shared covariance come from train, the thresholds from val
  (95 % of in-distribution validation windows kept).
Ball and cage windows are only scored, never used to choose anything. The design (penultimate
layer, shared covariance, 95 % coverage, both methods reported) was fixed before any OOD window
was scored.

Evaluation sets (freshly simulated, never used for training):
- in-distribution: the run's test split;
- unseen fault types (--ood-set): ball and cage (A1, 3-class model), or two custom bearing faults
  at orders 1.3 and 6.5 x shaft speed (A2, 5-class model), at speeds 80/125/175/220 rad/s x
  severity 0.5/1/2, 2 seeds;
- benign shifts (known classes, conditions outside the grid): 50 rad/s, 300 rad/s, and 150 %
  sensor noise at the default speed, 2 seeds per class.

Targets (TODO A1): >= 90 % of ball windows flagged unknown; >= 90 % of cage windows flagged unknown
or called a fault; accuracy on answered in-distribution windows drops by at most 1 point.

Usage:

    python experiments/report_ood_detector.py --out-dir docs/audit/ai [--cache data/ood_eval_cache.parquet]
    python experiments/report_ood_detector.py --run-dir configs/classifiers/pc_server_5class/<run> \
        --ood-set custom_orders --cache data/ood_eval_cache_custom.parquet
        [--write-to-run]   # store metrics.json["ood"] in the run (changes what the GUI shows)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.report_classifier_evaluation import SETTLE_S, _load_run, _simulate  # noqa: E402

RUN_DIR = ROOT / "configs" / "classifiers" / "pc_server" / "2026-10-18_run01"
OOD_SPEEDS = (80.0, 125.0, 175.0, 220.0)
OOD_SEVERITY_SCALES = (0.5, 1.0, 2.0)
SEEDS = (11, 12)
DURATION_S = 2.5
#: name -> list of (label, fault_type, order_override). "ball_cage" is A1's set (no ball/cage class
#: in the 3-class model). "custom_orders" is for the 5-class model of A2, where ball and cage are
#: classes: two custom bearing faults at characteristic orders 1.3 and 6.5 x shaft speed, away from
#: every built-in fault (outer 3.07, inner 4.93, ball 2.04, cage 0.38), with ball's mechanical
#: amplitude -- the "custom fault type" mechanism of the DC Motor page (viz/components/presets.py).
OOD_SETS = {
    "ball_cage": [("ball", "ball", None), ("cage", "cage", None)],
    "custom_orders": [("unknown_1p3x", "unknown_1p3x", 1.3), ("unknown_6p5x", "unknown_6p5x", 6.5)],
}
BENIGN = {
    "speed_50_rad_s": {"omega_ref_rad_s": 50.0},
    "speed_300_rad_s": {"omega_ref_rad_s": 300.0},
    "noise_150pct": {"electrical_noise_pct": 150.0, "mechanical_noise_pct": 150.0},
}


def _simulate_custom(job: tuple) -> pd.DataFrame:
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    cond, label, order, seed, speed, sev = job
    sc = Scenario(scenario_id=f"{cond}_seed{seed}", fault_type=label, fault_order_override=order, seed=seed, duration_s=DURATION_S,
                  omega_ref_rad_s=speed, electrical_severity=8.0 * sev, mechanical_severity=CALIBRATED_MECHANICAL_SEVERITY["ball"] * sev)
    df = records_to_dataframe(run_scenario(sc))
    df["condition"] = cond
    return df


def _sim(job):
    return _simulate_custom(job[1:]) if job[0] == "custom" else _simulate(job[1:])


def _jobs(ood_set: str = "ball_cage") -> list:
    jobs = []
    for label, fault, order in OOD_SETS[ood_set]:
        for speed in OOD_SPEEDS:
            for sev in OOD_SEVERITY_SCALES:
                for seed in SEEDS:
                    cond = f"ood_{label}_w{speed:g}_s{sev:g}"
                    if order is None:
                        jobs.append(("builtin", cond, fault, seed, {"omega_ref_rad_s": speed, "electrical_severity": 8.0 * sev}, sev, DURATION_S))
                    else:
                        jobs.append(("custom", cond, fault, order, seed, speed, sev))
    for cond, overrides in BENIGN.items():
        for fault in (None, "outer_race", "inner_race"):
            for seed in SEEDS:
                jobs.append(("builtin", f"benign_{cond}", fault, seed, overrides, 1.0, DURATION_S))
    return jobs


def simulate(workers: int, ood_set: str = "ball_cage") -> pd.DataFrame:
    with mp.get_context("spawn").Pool(workers) as pool:
        return pd.concat(pool.map(_sim, _jobs(ood_set)), ignore_index=True)


def rebuild_id_windows(metrics: dict, meta: dict):
    """Train/val/test windows of the run, rebuilt exactly as experiments/train_model.py did."""
    from sklearn.preprocessing import LabelEncoder

    from driveflow.models.common import build_classification_windows
    from driveflow.models.common.splits import grouped_train_val_test_split

    df = pd.read_parquet(ROOT / meta["dataset_path"])
    df = df[df["domain"] == "dc_motor"] if "domain" in df.columns else df
    classes = metrics["classes"]
    X, y_str, groups = build_classification_windows(df, classes, metrics["channels"], 512, cap_per_class=meta["cap_per_class"], level_scaling=metrics.get("level_scaling"))
    y = LabelEncoder().fit(classes).transform(y_str)
    split = grouped_train_val_test_split(groups, val_frac=0.2, test_frac=0.1, seed=metrics["split"]["seed"], y=y)
    if split.to_dict()["group_hashes"] != metrics["split"]["group_hashes"]:
        raise RuntimeError("rebuilt split does not match the run's recorded group hashes")
    return {s: (X[getattr(split, f"{s}_idx")], y[getattr(split, f"{s}_idx")]) for s in ("train", "val", "test")}


def _windows(df: pd.DataFrame, label: str, metrics: dict):
    from driveflow.models.common import build_classification_windows

    d = df[df["timestamp_s"] >= SETTLE_S]
    X, _y, _g = build_classification_windows(d, [label], metrics["channels"], 512, cap_per_class=100_000, level_scaling=metrics.get("level_scaling"))
    return X


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))], np.r_[id_scores, ood_scores]))


def evaluate(model, metrics: dict, ood: dict, id_sets: dict, sim: pd.DataFrame, ood_set: str = "ball_cage") -> dict:
    from driveflow.ai.calibration import UNCERTAIN, apply_calibration
    from driveflow.ai.ood import METHODS, scores

    classes = metrics["classes"]
    normal = classes.index("normal")
    abstain_below = metrics["calibration"]["abstain_below"]

    def decisions(x):
        p = apply_calibration(model.predict(x, batch_size=256, verbose=0), metrics)
        return p.argmax(axis=1), p.max(axis=1) < abstain_below, scores(model, x, ood)

    x_test, y_test = id_sets["test"]
    pred, unsure, s_test = decisions(x_test)
    out = {"in_distribution_test": {"n": int(len(y_test))}, "methods": {}}
    base_answered = ~unsure
    out["in_distribution_test"]["accuracy_on_answered_abstention_only"] = float(np.mean(pred[base_answered] == y_test[base_answered]))

    sets = {}
    labels = [label for label, _f, _o in OOD_SETS[ood_set]]
    for label in labels:
        x = _windows(sim[sim["condition"].str.startswith(f"ood_{label}_")], label, metrics)
        sets[f"ood_{label}"] = (x, None)
    for cond in BENIGN:
        d = sim[sim["condition"] == f"benign_{cond}"]
        xs, ys = [], []
        for k, c in enumerate(classes):
            x = _windows(d, c, metrics)
            if x is None:  # benign shifts are simulated for healthy / outer / inner only
                continue
            xs.append(x)
            ys.append(np.full(len(x), k))
        sets[f"benign_{cond}"] = (np.concatenate(xs), np.concatenate(ys))
    decided = {name: decisions(x) for name, (x, _y) in sets.items()}

    for m in METHODS:
        thr = ood[m]["threshold"]
        unk_test = s_test[m] > thr
        answered = ~unsure & ~unk_test
        r = {
            "threshold": thr,
            "in_distribution_false_unknown_rate": float(unk_test.mean()),
            "in_distribution_accuracy_on_answered": float(np.mean(pred[answered] == y_test[answered])),
            "in_distribution_answered_share": float(answered.mean()),
        }
        for name, (x, y) in sets.items():
            p_, uns_, s_ = decided[name]
            unk = s_[m] > thr
            entry = {"n": int(len(x)), "unknown_rate": float(unk.mean()), "auroc_vs_id_test": auroc(s_test[m], s_[m]),
                     "abstained_or_unknown_rate": float(np.mean(unk | uns_))}
            if name.startswith("ood_"):
                entry["unknown_or_called_fault_rate"] = float(np.mean(unk | (p_ != normal)))
                entry["confident_wrong_answer_rate_without_ood"] = float(np.mean(~uns_))
            if y is not None:
                ans = ~uns_ & ~unk
                entry["accuracy_on_answered"] = float(np.mean(p_[ans] == y[ans])) if ans.any() else None
            r[name] = entry
        first, second = (f"ood_{label}" for label in labels)
        r["targets"] = {
            f"{labels[0]}_unknown_ge_90pct": r[first]["unknown_rate"] >= 0.90,
            f"{labels[1]}_unknown_or_fault_ge_90pct": r[second]["unknown_or_called_fault_rate"] >= 0.90,
            "id_accuracy_drop_le_1pt": r["in_distribution_accuracy_on_answered"] >= out["in_distribution_test"]["accuracy_on_answered_abstention_only"] - 0.01,
        }
        out["methods"][m] = r
    return out


def _md(rep: dict) -> str:
    lines = [f"# OOD detector for the dc_motor PC classifier ({'A1' if rep['ood_set'] == 'ball_cage' else 'A2'})", "", f"Unknown-fault set: {rep['ood_set']} ({', '.join(rep['ood_labels'])}).", "", f"Mahalanobis features: {rep['mahalanobis_layer']}.", "",
             f"Run: `{rep['run']}`. Generated by `python experiments/report_ood_detector.py`; raw values in [`{rep['stem']}.json`]({rep['stem']}.json).",
             "Fitted on the run's own train (statistics) and val (thresholds, 95 % coverage) windows only; unknown-fault windows are evaluation-only.", "",
             f"In-distribution test windows: {rep['eval']['in_distribution_test']['n']}; accuracy on answered with abstention only: {rep['eval']['in_distribution_test']['accuracy_on_answered_abstention_only']:.3f}.", "",
             "| Metric | " + " | ".join(rep["eval"]["methods"]) + " |", "|---|" + "---|" * len(rep["eval"]["methods"])]
    ms = rep["eval"]["methods"]

    def row(label, f):
        lines.append(f"| {label} | " + " | ".join(f(ms[m]) for m in ms) + " |")

    row("ID false-unknown rate (test)", lambda r: f"{r['in_distribution_false_unknown_rate']:.3f}")
    row("ID accuracy on answered (abstention + OOD)", lambda r: f"{r['in_distribution_accuracy_on_answered']:.3f}")
    row("ID answered share", lambda r: f"{r['in_distribution_answered_share']:.3f}")
    labels = rep["ood_labels"]
    row(f"{labels[0]}: unknown rate (target ≥ 0.90)", lambda r: f"{r['ood_' + labels[0]]['unknown_rate']:.3f}")
    row(f"{labels[0]}: AUROC vs. ID test", lambda r: f"{r['ood_' + labels[0]]['auroc_vs_id_test']:.3f}")
    row(f"{labels[1]}: unknown rate", lambda r: f"{r['ood_' + labels[1]]['unknown_rate']:.3f}")
    row(f"{labels[1]}: unknown or called a fault (target ≥ 0.90)", lambda r: f"{r['ood_' + labels[1]]['unknown_or_called_fault_rate']:.3f}")
    row(f"{labels[1]}: AUROC vs. ID test", lambda r: f"{r['ood_' + labels[1]]['auroc_vs_id_test']:.3f}")
    for b in BENIGN:
        row(f"Benign {b}: false-unknown rate", lambda r, b=b: f"{r[f'benign_{b}']['unknown_rate']:.3f}")
        row(f"Benign {b}: accuracy on answered", lambda r, b=b: "—" if r[f'benign_{b}']['accuracy_on_answered'] is None else f"{r[f'benign_{b}']['accuracy_on_answered']:.3f}")
    row("Targets met", lambda r: ", ".join(k for k, v in r["targets"].items() if v) or "none")
    lines.append("")
    lines.append(f"Without the OOD check, {rep['eval']['methods'][next(iter(ms))]['ood_' + labels[0]]['confident_wrong_answer_rate_without_ood']:.3f} of {labels[0]} windows get a non-abstained answer (every such answer is wrong: the model has no class for this fault).")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "audit" / "ai")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--ood-set", choices=sorted(OOD_SETS), default="ball_cage")
    ap.add_argument("--layer", default=None, help="Mahalanobis features: None = the head's input (dense1), or e.g. 'gap'")
    ap.add_argument("--write-to-run", action="store_true", help="store the fitted parameters in the run's metrics.json['ood']")
    args = ap.parse_args(argv)

    from driveflow.ai.ood import fit_ood

    args.run_dir = args.run_dir.resolve()
    model, metrics = _load_run(args.run_dir)
    meta = json.loads((args.run_dir / "run_metadata.json").read_text())
    id_sets = rebuild_id_windows(metrics, meta)
    print({k: len(v[1]) for k, v in id_sets.items()}, "windows (train/val/test), split hashes match", flush=True)
    ood = fit_ood(model, *id_sets["train"], id_sets["val"][0], target_coverage=0.95, method="mahalanobis", layer=args.layer)

    if args.cache is not None and args.cache.exists():
        sim = pd.read_parquet(args.cache)
    else:
        sim = simulate(args.workers, args.ood_set)
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            sim.to_parquet(args.cache)

    rep = {"run": str(args.run_dir.relative_to(ROOT)), "ood_set": args.ood_set, "ood_labels": [lbl for lbl, _f, _o in OOD_SETS[args.ood_set]], "mahalanobis_layer": args.layer or "dense1 (head input)", "ood_params_summary": {m: {"threshold": ood[m]["threshold"]} for m in ("mahalanobis", "energy")},
           "jobs": {"ood_speeds": OOD_SPEEDS, "ood_severity_scales": OOD_SEVERITY_SCALES, "seeds": SEEDS, "duration_s": DURATION_S, "benign": BENIGN},
           "eval": evaluate(model, metrics, ood, id_sets, sim, args.ood_set)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = ("ood_detector" if args.ood_set == "ball_cage" else f"ood_detector_{args.ood_set}") + ("" if args.layer is None else f"_{args.layer}")
    rep["stem"] = stem
    (args.out_dir / f"{stem}.json").write_text(json.dumps(rep, indent=2) + "\n")
    (args.out_dir / f"{stem}.md").write_text(_md(rep))
    print(_md(rep))
    if args.write_to_run:
        metrics["ood"] = ood
        (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("wrote metrics.json['ood'] to", args.run_dir)


if __name__ == "__main__":
    main()
