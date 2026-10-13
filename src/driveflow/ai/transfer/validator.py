"""Model Validator: checks a fine-tuned model must pass before it is eligible for promotion.

Task-specific since AUDIT AI-05. The previous validator only had classification checks; for a
regressor it returned `mode_collapse_ok=True` and nothing else, so a forecaster emitting NaN or a
constant passed "validation". Now:

- Every check has an explicit status: PASS, FAIL, NOT_APPLICABLE (the check does not exist for
  this task -- e.g. mode collapse for a regressor; never counted as a pass) or INCOMPLETE (the
  check applies but cannot be decided -- missing baseline, missing threshold, missing history;
  fail-closed: blocks promotion like a FAIL).
- Classification metrics: accuracy, macro-F1, per-class recall, confusion matrix, prediction
  distribution (mode collapse), and a majority-class baseline on the same test windows (the
  classification counterpart of the persistence baseline below).
- Regression metrics: MAE, RMSE, RMSE per horizon step and per channel, finiteness, constant-output
  detection, and two baselines on the SAME data: the base (pre-fine-tuning) model and a naive
  persistence forecast (repeat the last context sample -- the simplest predictor a forecaster has
  to beat to be worth anything).
- Convergence (check_convergence) is now part of the report when a training history is given.

Thresholds are NOT invented here: evaluate() takes them explicitly. DEFAULT_THRESHOLDS are
proposed defaults shown in the UI as "requires researcher confirmation" (AUDIT_ANSWERS.md Q11/Q12).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCOMPLETE = "INCOMPLETE"


#: Proposed defaults, NOT validated acceptance criteria -- see module docstring.
#: classification: the fine-tuned model may lose at most this much macro-F1 vs. the base model
#: evaluated on the same independent test set. regression: its RMSE may exceed the base model's
#: RMSE on the same test set by at most this relative fraction.
DEFAULT_THRESHOLDS = {
    "classification": {"max_macro_f1_drop": 0.02, "min_class_recall": None},
    "regression": {"max_rmse_increase": 0.05},
}


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    value: float | None = None
    threshold: float | None = None
    required: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ValidationReport:
    task: str
    metrics: dict
    baseline_metrics: dict
    checks: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff every REQUIRED check is PASS or NOT_APPLICABLE (INCOMPLETE blocks)."""
        return all(c.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE) for c in self.checks if c.required)

    def to_dict(self) -> dict:
        return {"task": self.task, "passed": self.passed, "metrics": self.metrics, "baseline_metrics": self.baseline_metrics, "checks": [c.to_dict() for c in self.checks]}


def _predict(model, X):
    return np.asarray(model.predict(X, verbose=0))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    n_classes = n_classes or probs.shape[-1]
    finite = bool(np.all(np.isfinite(probs)))
    y_pred = np.argmax(np.nan_to_num(probs, nan=-np.inf), axis=-1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    recall, f1 = [], []
    for k in range(n_classes):
        tp = cm[k, k]
        support, predicted = cm[k].sum(), cm[:, k].sum()
        r = tp / support if support else np.nan
        prec = tp / predicted if predicted else 0.0
        recall.append(float(r))
        f1.append(float(2 * prec * r / (prec + r)) if support and (prec + r) > 0 else (0.0 if support else np.nan))
    present = [f for f, k in zip(f1, range(n_classes)) if cm[k].sum() > 0]
    return {
        "n": int(len(y_true)),
        "finite": finite,
        "accuracy": float(np.mean(y_pred == y_true)) if len(y_true) else None,
        "macro_f1": float(np.mean(present)) if present else None,
        "recall_per_class": recall,
        "confusion_matrix": cm.tolist(),
        "n_distinct_predicted": int(len(np.unique(y_pred))),
        "n_distinct_true": int(len(np.unique(y_true))),
    }


def regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    Y_true, Y_pred = np.asarray(Y_true, dtype=float), np.asarray(Y_pred, dtype=float)
    finite = bool(np.all(np.isfinite(Y_pred)))
    if not finite or Y_pred.shape != Y_true.shape:
        return {"n": int(len(Y_true)), "finite": finite, "shape_ok": Y_pred.shape == Y_true.shape, "mae": None, "rmse": None, "rmse_per_step": None, "rmse_per_channel": None, "pred_std": None, "target_std": float(np.std(Y_true))}
    err = Y_pred - Y_true
    step_axes = tuple(i for i in range(err.ndim) if i != 1) if err.ndim >= 2 else None
    chan_axes = tuple(range(err.ndim - 1)) if err.ndim >= 2 else None
    return {
        "n": int(len(Y_true)),
        "finite": True,
        "shape_ok": True,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rmse_per_step": np.sqrt(np.mean(err**2, axis=step_axes)).tolist() if step_axes is not None else None,
        "rmse_per_channel": np.sqrt(np.mean(err**2, axis=chan_axes)).tolist() if chan_axes is not None else None,
        # Spread of the predictions ACROSS samples: a model that ignores its input outputs the same
        # sequence for every window, so this collapses to ~0 while the targets keep varying.
        "pred_std": float(np.mean(np.std(Y_pred, axis=0))),
        "target_std": float(np.mean(np.std(Y_true, axis=0))),
    }


def persistence_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """Naive baseline: every future step equals the last context sample (per channel)."""
    last = np.asarray(X)[:, -1:, :]
    return np.repeat(last, horizon, axis=1)


class ModelValidator:
    """thresholds accepts either `min_accuracy` (an absolute floor) or `max_accuracy_drop` (a
    ceiling on how much worse than baseline_metrics["accuracy"] the new model may be) --
    min_accuracy wins if both are given. Those two keys drive the legacy validate() API; the
    task-specific evaluate() takes its own thresholds (see DEFAULT_THRESHOLDS)."""

    def __init__(self, baseline_metrics: dict, thresholds: dict):
        self.baseline_metrics = baseline_metrics
        self.thresholds = thresholds
        self.validation_results: dict = {}

    # ------------------------------------------------------------------ task-specific API
    def evaluate(self, finetuned_model, test_data: dict, task: str, baseline_model=None, history: dict | None = None, thresholds: dict | None = None) -> ValidationReport:
        """Evaluates finetuned_model (and baseline_model, if given) on the SAME test_data and
        returns a ValidationReport. `thresholds` defaults to self.thresholds; missing keys make
        the corresponding check INCOMPLETE, never PASS."""
        thresholds = self.thresholds if thresholds is None else thresholds
        X, y = np.asarray(test_data["X"]), np.asarray(test_data["y"])
        if task == "classification":
            report = self._evaluate_classifier(finetuned_model, X, y, baseline_model, thresholds)
        elif task == "regression":
            report = self._evaluate_regressor(finetuned_model, X, y, baseline_model, thresholds)
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
        report.checks.append(self.convergence_check(history))
        return report

    def _evaluate_classifier(self, model, X, y, baseline_model, thr) -> ValidationReport:
        probs = _predict(model, X)
        m = classification_metrics(y, probs)
        base = classification_metrics(y, _predict(baseline_model, X), n_classes=probs.shape[-1]) if baseline_model is not None else {}
        checks = [
            CheckResult("predictions_finite", CheckStatus.PASS if m["finite"] else CheckStatus.FAIL, "all output probabilities finite" if m["finite"] else "NaN/Inf in model output"),
        ]
        if m["n_distinct_true"] <= 1:
            checks.append(CheckResult("mode_collapse", CheckStatus.INCOMPLETE, "test set has a single class -- collapse cannot be distinguished from correctness"))
        else:
            ok = m["n_distinct_predicted"] > 1
            checks.append(CheckResult("mode_collapse", CheckStatus.PASS if ok else CheckStatus.FAIL, f"{m['n_distinct_predicted']} distinct class(es) predicted for {m['n_distinct_true']} present"))
        # Objective, threshold-free floor: "no worse than the base model" also passes when BOTH are
        # useless (observed: base and candidate at macro-F1 0.095 on 3 balanced classes). A model
        # must beat the majority-class predictor on the same test windows to be worth promoting.
        majority = np.full_like(np.asarray(y).astype(int), np.bincount(np.asarray(y).astype(int)).argmax())
        maj = classification_metrics(y, np.eye(probs.shape[-1])[majority])
        m["majority_baseline_macro_f1"] = maj["macro_f1"]
        if m["macro_f1"] is None or maj["macro_f1"] is None:
            checks.append(CheckResult("beats_majority_baseline", CheckStatus.INCOMPLETE, "macro-F1 undefined"))
        else:
            ok = m["macro_f1"] > maj["macro_f1"]
            checks.append(CheckResult("beats_majority_baseline", CheckStatus.PASS if ok else CheckStatus.FAIL, f"macro-F1 {m['macro_f1']:.3f} vs majority-class predictor {maj['macro_f1']:.3f}", m["macro_f1"], maj["macro_f1"]))
        drop = thr.get("max_macro_f1_drop")
        if baseline_model is None or base.get("macro_f1") is None:
            checks.append(CheckResult("macro_f1_vs_base", CheckStatus.INCOMPLETE, "base model not evaluated on the same test set"))
        elif drop is None:
            checks.append(CheckResult("macro_f1_vs_base", CheckStatus.INCOMPLETE, "no max_macro_f1_drop threshold configured"))
        elif m["macro_f1"] is None:
            checks.append(CheckResult("macro_f1_vs_base", CheckStatus.FAIL, "macro-F1 undefined"))
        else:
            ok = m["macro_f1"] >= base["macro_f1"] - drop
            checks.append(CheckResult("macro_f1_vs_base", CheckStatus.PASS if ok else CheckStatus.FAIL, f"fine-tuned {m['macro_f1']:.3f} vs base {base['macro_f1']:.3f} (allowed drop {drop:g})", m["macro_f1"], base["macro_f1"] - drop))
        min_recall = thr.get("min_class_recall")
        if min_recall is not None:
            recalls = [r for r in m["recall_per_class"] if not np.isnan(r)]
            ok = bool(recalls) and min(recalls) >= min_recall
            checks.append(CheckResult("min_class_recall", CheckStatus.PASS if ok else CheckStatus.FAIL, f"worst class recall {min(recalls) if recalls else float('nan'):.3f}", min(recalls) if recalls else None, min_recall))
        if thr.get("min_accuracy") is not None:
            ok = m["accuracy"] is not None and m["accuracy"] >= thr["min_accuracy"]
            checks.append(CheckResult("min_accuracy", CheckStatus.PASS if ok else CheckStatus.FAIL, f"accuracy {m['accuracy']:.3f}", m["accuracy"], thr["min_accuracy"]))
        return ValidationReport("classification", m, base, checks)

    def _evaluate_regressor(self, model, X, Y, baseline_model, thr) -> ValidationReport:
        pred = _predict(model, X)
        m = regression_metrics(Y, pred)
        base = regression_metrics(Y, _predict(baseline_model, X)) if baseline_model is not None else {}
        naive = regression_metrics(Y, persistence_forecast(X, Y.shape[1])) if Y.ndim == 3 and X.ndim == 3 and X.shape[-1] == Y.shape[-1] else {}
        m["persistence_rmse"] = naive.get("rmse")
        checks = [
            CheckResult("predictions_finite", CheckStatus.PASS if m["finite"] and m["shape_ok"] else CheckStatus.FAIL, "all predictions finite, shape matches target" if m["finite"] and m["shape_ok"] else "NaN/Inf in predictions or wrong output shape"),
            CheckResult("mode_collapse", CheckStatus.NOT_APPLICABLE, "class collapse does not exist for a regressor (see non_constant_output)", required=False),
        ]
        if not (m["finite"] and m["shape_ok"]):
            checks.append(CheckResult("non_constant_output", CheckStatus.FAIL, "predictions unusable"))
            checks.append(CheckResult("rmse_vs_base", CheckStatus.FAIL, "predictions unusable"))
            checks.append(CheckResult("beats_persistence", CheckStatus.FAIL, "predictions unusable"))
            return ValidationReport("regression", m, base, checks)
        tol = 1e-3 * max(m["target_std"], 1e-12)
        ok = m["pred_std"] > tol
        checks.append(CheckResult("non_constant_output", CheckStatus.PASS if ok else CheckStatus.FAIL, f"prediction spread {m['pred_std']:.3g} vs target spread {m['target_std']:.3g}", m["pred_std"], tol))
        inc = thr.get("max_rmse_increase")
        if baseline_model is None or base.get("rmse") is None:
            checks.append(CheckResult("rmse_vs_base", CheckStatus.INCOMPLETE, "base model not evaluated on the same test set"))
        elif inc is None:
            checks.append(CheckResult("rmse_vs_base", CheckStatus.INCOMPLETE, "no max_rmse_increase threshold configured"))
        else:
            limit = base["rmse"] * (1 + inc)
            ok = m["rmse"] <= limit
            checks.append(CheckResult("rmse_vs_base", CheckStatus.PASS if ok else CheckStatus.FAIL, f"fine-tuned RMSE {m['rmse']:.4f} vs base {base['rmse']:.4f} (limit {limit:.4f})", m["rmse"], limit))
        if naive.get("rmse") is None:
            checks.append(CheckResult("beats_persistence", CheckStatus.INCOMPLETE, "persistence baseline not computable for these shapes"))
        else:
            ok = m["rmse"] < naive["rmse"]
            checks.append(CheckResult("beats_persistence", CheckStatus.PASS if ok else CheckStatus.FAIL, f"RMSE {m['rmse']:.4f} vs persistence {naive['rmse']:.4f}", m["rmse"], naive["rmse"]))
        return ValidationReport("regression", m, base, checks)

    def convergence_check(self, history: dict | None) -> CheckResult:
        if not history or not history.get("loss"):
            return CheckResult("convergence", CheckStatus.INCOMPLETE, "no training history supplied")
        ok_train = self.check_convergence(history["loss"])
        val = history.get("val_loss") or []
        val_finite = bool(np.all(np.isfinite(np.asarray(val, dtype=float)))) if val else True
        ok = ok_train and val_finite
        detail = f"loss {history['loss'][0]:.4g} -> {history['loss'][-1]:.4g}" + (f", val_loss -> {val[-1]:.4g}" if val else ", no val_loss")
        return CheckResult("convergence", CheckStatus.PASS if ok else CheckStatus.FAIL, detail if ok else "diverged / non-finite loss: " + detail)

    # ------------------------------------------------------------------ legacy API (kept)
    def validate(self, finetuned_model, val_data: dict) -> tuple:
        """Legacy entry point, kept for existing callers/tests. Regressors now get real checks
        (finite + non-constant) instead of an unconditional mode_collapse_ok=True."""
        X_val, y_val = np.asarray(val_data["X"]), np.asarray(val_data["y"])
        preds = np.asarray(finetuned_model.predict(X_val, verbose=0))

        results: dict = {}
        if preds.ndim == 2 and preds.shape[-1] > 1 and np.issubdtype(y_val.dtype, np.integer):
            pred_labels = np.argmax(preds, axis=-1)
            accuracy = float(np.mean(pred_labels == y_val))
            results["accuracy"] = accuracy
            results["accuracy_ok"] = self.check_accuracy_threshold(accuracy)
            results["mode_collapse_ok"] = self.check_mode_collapse(pred_labels.tolist())
        else:
            m = regression_metrics(y_val, preds)
            results["rmse"], results["mae"] = m["rmse"], m["mae"]
            results["predictions_finite_ok"] = bool(m["finite"] and m["shape_ok"])
            results["non_constant_output_ok"] = bool(results["predictions_finite_ok"] and m["pred_std"] > 1e-3 * max(m["target_std"], 1e-12))
            results["mode_collapse"] = CheckStatus.NOT_APPLICABLE.value

        self.validation_results = results
        passed = all(v for k, v in results.items() if k.endswith("_ok"))
        return passed, results

    def check_accuracy_threshold(self, new_accuracy: float) -> bool:
        min_accuracy = self.thresholds.get("min_accuracy")
        if min_accuracy is not None:
            return new_accuracy >= min_accuracy
        baseline = self.baseline_metrics.get("accuracy")
        if baseline is None:
            return True
        max_drop = self.thresholds.get("max_accuracy_drop", 0.0)
        return new_accuracy >= baseline - max_drop

    def check_convergence(self, loss_history: list) -> bool:
        if len(loss_history) < 2:
            return False
        arr = np.asarray(loss_history, dtype=float)
        if not np.all(np.isfinite(arr)):
            return False
        return bool(arr[-1] <= arr[0])

    def check_mode_collapse(self, predictions: list) -> bool:
        return len(set(predictions)) > 1
