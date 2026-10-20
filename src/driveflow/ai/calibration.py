"""Confidence calibration and abstention for classifiers (docs/AI_TECHNICAL_AUDIT.md AI-N12).

A softmax output is not a probability: the shipped classifiers answered 0.99 on windows they got
wrong (another speed, half severity, fault types they have no class for). Two mechanisms, both
fitted on the VALIDATION split at training time and stored in the run's metrics.json:

- temperature scaling (Guo et al., 2017): p_T = softmax(log p / T), one scalar T > 0 chosen to
  minimize the validation negative log-likelihood. It rescales confidence without changing the
  predicted class, so accuracy is untouched;
- abstention: a calibrated top-1 confidence below `abstain_below` is reported as "uncertain"
  instead of a class. The threshold is the validation quantile that keeps `target_coverage` of the
  CORRECT validation windows, so it is set by a stated coverage target, not tuned on test data.

Neither makes an out-of-distribution input safe: a model can be confidently wrong on an unseen
condition and calibration fitted in-distribution does not see it. The evaluation report measures
how often the abstention catches such cases instead of assuming it does.

metrics.json keys written by experiments/train_model.py:
    "calibration": {"method": "temperature", "temperature": T, "fitted_on": "val", "val_nll_before": ..., "val_nll_after": ...,
                    "abstain_below": c, "target_coverage": 0.95}
Runs without the key are uncalibrated; apply_calibration() then returns the probabilities unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

UNCERTAIN = "uncertain"
#: Decision for a window the run's OOD detector (ai/ood.py, metrics.json["ood"]) flags as unlike
#: anything the classifier was trained on -- e.g. a fault type it has no class for.
UNKNOWN = "unknown"
_EPS = 1e-12


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0)) / float(temperature)
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1.0))))


def fit_temperature(probs: np.ndarray, y_true: np.ndarray) -> tuple:
    """(T, nll_before, nll_after); T searched on log scale in [0.05, 100]."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    res = minimize_scalar(lambda lt: _nll(apply_temperature(probs, np.exp(lt)), y), bounds=(np.log(0.05), np.log(100.0)), method="bounded")
    t = float(np.exp(res.x))
    return t, _nll(probs, y), _nll(apply_temperature(probs, t), y)


def abstention_threshold(calibrated_probs: np.ndarray, y_true: np.ndarray, target_coverage: float = 0.95) -> float:
    """Confidence below which a prediction is withheld: the (1 - target_coverage) quantile of the
    calibrated top-1 confidence over the correctly classified validation windows."""
    p = np.asarray(calibrated_probs)
    correct = p.argmax(axis=1) == np.asarray(y_true)
    if not correct.any():
        return 1.0
    return float(np.quantile(p.max(axis=1)[correct], 1.0 - target_coverage))


def fit_calibration(val_probs: np.ndarray, y_val: np.ndarray, target_coverage: float = 0.95) -> dict:
    t, before, after = fit_temperature(val_probs, y_val)
    cal = apply_temperature(val_probs, t)
    return {"method": "temperature", "temperature": t, "fitted_on": "val", "val_nll_before": before, "val_nll_after": after,
            "abstain_below": abstention_threshold(cal, y_val, target_coverage), "target_coverage": target_coverage}


def apply_calibration(probs: np.ndarray, metrics: dict) -> np.ndarray:
    cal = metrics.get("calibration")
    return apply_temperature(probs, cal["temperature"]) if cal else np.asarray(probs, dtype=np.float64)


def decide(probs: np.ndarray, classes: list, metrics: dict, ood_score: float | None = None) -> tuple:
    """(label, UNCERTAIN or UNKNOWN, calibrated probs) for ONE window's probability vector.

    ood_score: the window's score under the run's OOD method (ai/ood.py). It is only used when the
    run has metrics.json["ood"]; above that method's threshold the answer is UNKNOWN, which takes
    precedence over UNCERTAIN (an unfamiliar input is a different problem from a close call)."""
    p = apply_calibration(np.asarray(probs)[np.newaxis, :], metrics)[0]
    k = int(np.argmax(p))
    ood = metrics.get("ood")
    if ood and ood_score is not None and ood_score > ood[ood["method"]]["threshold"]:
        return UNKNOWN, p
    cal = metrics.get("calibration")
    if cal and p[k] < cal["abstain_below"]:
        return UNCERTAIN, p
    return classes[k], p
