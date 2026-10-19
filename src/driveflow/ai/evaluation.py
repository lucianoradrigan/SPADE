"""Classifier evaluation metrics beyond accuracy (docs/AI_TECHNICAL_AUDIT.md Sec. 2.3).

classification_metrics() returns, for one (y_true, probs) set:
- accuracy, balanced accuracy, macro-F1;
- per-class precision / recall / F1 / support, and one-vs-rest false-positive and false-negative
  rates;
- the confusion matrix (rows = true class, columns = predicted);
- fault-detection view (`normal_class` vs. every other class): false-alarm rate (a healthy window
  reported as any fault) and missed-detection rate (a faulty window reported as normal) -- the two
  errors a monitoring operator actually experiences, which a 3-class accuracy hides;
- calibration: expected calibration error (ECE, equal-width confidence bins), Brier score, mean
  top-1 confidence on correct vs. incorrect predictions.

No threshold is decided here; the caller states and justifies its acceptance criteria.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """sum_b |B_b|/N * |acc(B_b) - conf(B_b)| over equal-width bins of top-1 confidence."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> list:
    """[(bin_center, mean_confidence, accuracy, count)] for a reliability diagram (empty bins skipped)."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            out.append((float((lo + hi) / 2), float(conf[in_bin].mean()), float(correct[in_bin].mean()), int(in_bin.sum())))
    return out


def classification_metrics(y_true, probs, classes: list, normal_class: str | None = "normal", n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(classes) or len(probs) != len(y_true):
        raise ValueError(f"probs must be (n, {len(classes)}) matching y_true, got {probs.shape} vs {y_true.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs contain non-finite values")
    labels = list(range(len(classes)))
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    n = cm.sum()
    per_class = {}
    for k, c in enumerate(classes):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = n - tp - fn - fp
        per_class[c] = {
            "precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k]),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
            "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        }
    conf = probs.max(axis=1)
    correct = y_pred == y_true
    one_hot = np.eye(len(classes))[y_true]
    out = {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else float(correct.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix_rows_true": cm.tolist(),
        "ece": expected_calibration_error(probs, y_true, n_bins),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
    }
    if normal_class is not None and normal_class in classes:
        kn = classes.index(normal_class)
        healthy, faulty = y_true == kn, y_true != kn
        out["fault_detection"] = {
            "false_alarm_rate": float((y_pred[healthy] != kn).mean()) if healthy.any() else None,
            "missed_detection_rate": float((y_pred[faulty] == kn).mean()) if faulty.any() else None,
        }
    return out


def ood_confidence_summary(probs: np.ndarray, classes: list) -> dict:
    """For windows of a condition the model has NO class for (e.g. ball/cage faults): what it
    predicts and how confident it is. A model without an abstention mechanism always answers;
    high confidence here means its confidence cannot be used to detect unknown conditions."""
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return {
        "n": int(len(probs)),
        "predicted_share": {c: float((pred == k).mean()) for k, c in enumerate(classes)},
        "mean_confidence": float(conf.mean()),
        "share_confidence_above_0.9": float((conf > 0.9).mean()),
    }
