"""AUDIT AI-09 / Q15: how far does the promoted dc_motor/pc/classifier's recorded test score
(metrics.json) generalize? Nothing is re-trained and no recorded number is changed; this only
re-evaluates the promoted weights.

Three measurements:
  1. Recorded split: the training-time grouped split is rebuilt (same dataset, channels, seed) to
     show which runs the recorded n_test windows come from.
  2. Fresh runs, training-like: new seeds, first 1.0 s of each run (the training dataset's
     duration, start-up transient included).
  3. Fresh runs, steady state: same runs, windows from t >= --settle-s only.

Uncertainty is a bootstrap over RUNS (stratified by class), not over windows: windows of one run
are strongly correlated, so a window-level interval would be falsely narrow.

With --run-dir, a non-promoted run (e.g. a candidate trained by experiments/train_model.py) is
evaluated on the same fresh runs instead; measurement 1 is then read from its own metrics.json.

Usage:  python experiments/report_classifier_generalization.py [--seeds 8] [--duration-s 3] [--run-dir DIR] [--json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from driveflow.ai.registry import load_promoted_model, resolve
from driveflow.datagen import healthy_and_faulted_grid, run_scenarios
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import load_classifier_config
from driveflow.datagen.export_parquet import records_to_dataframe
from driveflow.models.common import build_classification_windows, filter_diagnosis_domain, grouped_split, load_diagnosis_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATASET = REPO_ROOT / "data" / "diagnosis_dataset.parquet"
FAULT_TYPES = ("outer_race", "inner_race")


def _predict(model, X: np.ndarray) -> np.ndarray:
    return np.argmax(model.predict(X, verbose=0), axis=1)


def _evaluate(model, df: pd.DataFrame, classes: list, channels: list, window: int, n_boot: int, seed: int) -> dict:
    X, y_str, groups = build_classification_windows(df, classes, channels, window)
    y = np.array([classes.index(c) for c in y_str])
    pred = _predict(model, X)
    runs = sorted(set(groups))
    per_run = {g: float((pred[groups == g] == y[groups == g]).mean()) for g in runs}
    run_label = {g: y[groups == g][0] for g in runs}

    rng = np.random.default_rng(seed)
    by_class = {k: [g for g in runs if run_label[g] == k] for k in range(len(classes))}
    boot_acc, boot_f1 = [], []
    for _ in range(n_boot):
        pick = [g for k in by_class for g in rng.choice(by_class[k], size=len(by_class[k]), replace=True)] if all(by_class.values()) else list(rng.choice(runs, size=len(runs)))
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        boot_acc.append((pred[idx] == y[idx]).mean())
        boot_f1.append(f1_score(y[idx], pred[idx], average="macro", labels=list(range(len(classes))), zero_division=0))

    return {
        "n_runs": len(runs),
        "n_windows": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(len(classes))), zero_division=0)),
        "accuracy_95ci_over_runs": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
        "macro_f1_95ci_over_runs": [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))],
        "majority_class_accuracy": float(np.bincount(y, minlength=len(classes)).max() / len(y)),
        "per_class_recall": {c: float((pred[y == k] == k).mean()) if (y == k).any() else None for k, c in enumerate(classes)},
        "confusion_matrix_rows_true": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
        "per_run_accuracy": per_run,
    }


def recorded_split(model, metrics: dict, seed: int = 42) -> dict:
    """Rebuilds experiments/train_model.py's split (prepare_classification_splits, default seed
    42) on the training dataset and evaluates the promoted weights on each side of it."""
    df = load_diagnosis_dataset(TRAIN_DATASET)
    classes, channels = metrics["classes"], metrics["channels"]
    X, y_str, groups = build_classification_windows(df, classes, channels, 512)
    # Same two grouped_split calls as prepare_classification_splits, kept here because that
    # function returns the arrays but not which runs landed on each side.
    y = np.array([classes.index(c) for c in y_str])
    idx = np.arange(len(X))
    idx_tv, idx_test = grouped_split(idx, y, groups, 0.10, seed)
    idx_train, idx_val = grouped_split(idx_tv, y[idx_tv], groups[idx_tv], 0.222, seed)
    pred = _predict(model, X)
    side = {}
    for name, ix in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        side[name] = {
            "n_windows": int(len(ix)),
            "runs": sorted(set(groups[ix])),
            "classes_present": sorted({classes[k] for k in y[ix]}),
            "accuracy": float((pred[ix] == y[ix]).mean()),
        }
    run_meta = df.groupby("source_file").agg(duration_s=("timestamp_s", "max"), seed=("seed", "first"))
    return {
        "dataset": str(TRAIN_DATASET.relative_to(REPO_ROOT)),
        "n_runs": int(len(run_meta)),
        "run_duration_s": float(run_meta["duration_s"].round(2).max()),
        "seeds": sorted(int(s) for s in run_meta["seed"].unique()),
        "matches_recorded_counts": [side["train"]["n_windows"], side["val"]["n_windows"], side["test"]["n_windows"]] == [metrics["n_train"], metrics["n_val"], metrics["n_test"]],
        **side,
    }


def _load_run(run_dir: Path) -> tuple:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model = build_classifier(load_classifier_config(run_dir / "config.yaml"), n_channels=len(metrics["channels"]))
    model.load_weights(run_dir / "model.weights.h5")
    return model, metrics


def report(n_seeds: int = 8, first_seed: int = 100, duration_s: float = 3.0, settle_s: float = 0.5, n_boot: int = 2000, seed: int = 0, cache: Path = None, run_dir: Path = None) -> dict:
    if run_dir is None:
        model, _config, metrics = load_promoted_model("dc_motor", "pc", "classifier")
        run_dir = resolve("dc_motor", "pc", "classifier")
        split = recorded_split(model, metrics)
    else:
        model, metrics = _load_run(run_dir)
        split = None
    classes, channels = metrics["classes"], metrics["channels"]

    if cache is not None and cache.exists():
        fresh = pd.read_parquet(cache)
    else:
        scenarios = healthy_and_faulted_grid("audit_fresh", fault_types=FAULT_TYPES, seeds=tuple(range(first_seed, first_seed + n_seeds)), duration_s=duration_s, omega_ref_rad_s=150.0)
        fresh = filter_diagnosis_domain(records_to_dataframe([r for run in run_scenarios(scenarios) for r in run]))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            fresh.to_parquet(cache)

    return {
        "model_run": str(Path(run_dir).resolve().relative_to(REPO_ROOT)),
        "recorded": {k: metrics.get(k) for k in ("n_train", "n_val", "n_test", "test_accuracy", "test_macro_f1", "test_per_class_recall", "split")},
        "recorded_split": split,
        "fresh_seeds": [first_seed, first_seed + n_seeds - 1],
        "fresh_training_like_first_1s": _evaluate(model, fresh[fresh["timestamp_s"] < 1.0], classes, channels, 512, n_boot, seed),
        "fresh_steady_state": _evaluate(model, fresh[fresh["timestamp_s"] >= settle_s], classes, channels, 512, n_boot, seed),
        "settle_s": settle_s,
        "fresh_duration_s": duration_s,
    }


def _fmt(name: str, r: dict) -> str:
    lo, hi = r["accuracy_95ci_over_runs"]
    flo, fhi = r["macro_f1_95ci_over_runs"]
    recall = ", ".join(f"{c} {v:.2f}" for c, v in r["per_class_recall"].items())
    return (
        f"{name}: {r['n_runs']} runs / {r['n_windows']} windows\n"
        f"  accuracy {r['accuracy']:.3f} (95% CI over runs {lo:.3f}-{hi:.3f})   macro-F1 {r['macro_f1']:.3f} ({flo:.3f}-{fhi:.3f})\n"
        f"  majority-class baseline {r['majority_class_accuracy']:.3f}   recall: {recall}\n"
        f"  confusion (rows = true {list(r['per_class_recall'])}): {r['confusion_matrix_rows_true']}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8, help="fresh seeds per class")
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--duration-s", type=float, default=3.0)
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--cache", type=Path, default=None, help="parquet to reuse/save the fresh runs")
    ap.add_argument("--run-dir", type=Path, default=None, help="evaluate this run instead of the promoted one")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = report(args.seeds, args.first_seed, args.duration_s, args.settle_s, cache=args.cache, run_dir=args.run_dir)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    rec, rs = r["recorded"], r["recorded_split"]
    print(f"Model: {r['model_run']}")
    print(f"Recorded: test accuracy {rec['test_accuracy']:.3f} / macro-F1 {rec['test_macro_f1']:.3f} on n_test = {rec['n_test']} windows")
    if rs is not None:
        print(f"Training dataset: {rs['dataset']} -- {rs['n_runs']} runs x {rs['run_duration_s']:.1f} s, seeds {rs['seeds']}; split rebuilt matches recorded counts: {rs['matches_recorded_counts']}")
        for side in ("train", "val", "test"):
            s = rs[side]
            print(f"  {side:5s}: {s['n_windows']:3d} windows from {len(s['runs'])} runs, classes {s['classes_present']}, accuracy {s['accuracy']:.3f}")
    elif rec.get("split"):
        print(f"  split: {rec['split']['method']}, groups {rec['split']['n_groups']}, test class counts {rec['split']['class_counts'].get('test')}; recorded per-class recall {rec.get('test_per_class_recall')}")
    print()
    print(_fmt(f"Fresh seeds {r['fresh_seeds'][0]}-{r['fresh_seeds'][1]}, first 1.0 s (training-like, transient included)", r["fresh_training_like_first_1s"]))
    print()
    print(_fmt(f"Fresh seeds {r['fresh_seeds'][0]}-{r['fresh_seeds'][1]}, t >= {r['settle_s']} s of {r['fresh_duration_s']} s runs (steady state)", r["fresh_steady_state"]))


if __name__ == "__main__":
    main()
