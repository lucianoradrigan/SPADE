"""Model Validator: sanity checks a fine-tuned model must pass before it's eligible for promotion.

Phase 2: three real, independent checks -- accuracy regression vs. baseline/a floor, a loss curve
that actually decreased (not diverged/NaN'd), and predictions that didn't collapse to a single
class (a classifier that learned to always predict the majority class can still show a
plausible-looking raw accuracy on an imbalanced validation set -- check_mode_collapse catches
that specifically, separate from check_accuracy_threshold).
"""

from __future__ import annotations

import numpy as np


class ModelValidator:
    """thresholds accepts either `min_accuracy` (an absolute floor) or `max_accuracy_drop` (a
    ceiling on how much worse than baseline_metrics["accuracy"] the new model may be) --
    min_accuracy wins if both are given."""

    def __init__(self, baseline_metrics: dict, thresholds: dict):
        self.baseline_metrics = baseline_metrics
        self.thresholds = thresholds
        self.validation_results: dict = {}

    def validate(self, finetuned_model, val_data: dict) -> tuple:
        X_val, y_val = np.asarray(val_data["X"]), np.asarray(val_data["y"])
        preds = finetuned_model.predict(X_val, verbose=0)

        results: dict = {}
        if preds.ndim == 2 and preds.shape[-1] > 1 and np.issubdtype(y_val.dtype, np.integer):
            pred_labels = np.argmax(preds, axis=-1)
            accuracy = float(np.mean(pred_labels == y_val))
            results["accuracy"] = accuracy
            results["accuracy_ok"] = self.check_accuracy_threshold(accuracy)
            results["mode_collapse_ok"] = self.check_mode_collapse(pred_labels.tolist())
        else:
            # Regressor output: no accuracy/mode-collapse notion -- only the loss-convergence
            # check (run separately by the caller from its own training history) applies.
            results["mode_collapse_ok"] = True

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
