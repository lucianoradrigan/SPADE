"""Condition state of one classifier answer -- expansion task E10a (spec Sec. 8.2).

Four mutually distinguishable outcomes, decided in this order:

1. ``input_invalid``  -- the window cannot be evaluated (missing channel, too few rows, a non-finite
   sample). Nothing is predicted.
2. ``unknown``  -- the run has a fitted OOD detector (ai/ood.py, task A1) and the window's score is
   above its calibrated threshold, whatever the top-class confidence is.
3. ``uncertain`` -- in distribution (or no detector), but the calibrated top-class probability is
   below the run's abstention threshold (ai/calibration.py).
4. ``known`` -- both checks passed.

When the run has no usable OOD detector the ``unknown`` state is unreachable and the assessment says
so (``ood_available`` False plus the reason); no score is ever made up. The same holds for
``uncertain`` when the run has no calibration. The decision itself is calibration.decide(), the
function the rest of SPADE already uses, so the GUI and the monitoring path cannot disagree.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from driveflow.ai.calibration import UNCERTAIN, UNKNOWN, decide
from driveflow.ai.ood import METHODS

INPUT_INVALID = "input_invalid"
KNOWN = "known"
STATE_UNCERTAIN = "uncertain"
STATE_UNKNOWN = "unknown"

LABELS = {
    INPUT_INVALID: "Input invalid",
    KNOWN: "Known condition",
    STATE_UNCERTAIN: "Uncertain — low confidence",
    STATE_UNKNOWN: "Unknown condition — REQUIRES VERIFICATION",
}
EXPLANATIONS = {
    INPUT_INVALID: "The input cannot be evaluated by this model, so nothing is predicted.",
    KNOWN: "The top class passed the confidence check and the input is within the conditions represented in training.",
    STATE_UNCERTAIN: "The classifier cannot distinguish reliably among its known classes.",
    STATE_UNKNOWN: "The input differs from the conditions represented in training.",
}


def window_problems(df: pd.DataFrame, channels: list, n: int) -> list[str]:
    """Why the last ``n`` rows of ``df`` cannot be a model input (empty list = valid)."""
    missing = [c for c in channels if c not in df.columns or df[c].isna().all()]
    if missing:
        return [f"Missing channel(s) this model needs: {', '.join(missing)}."]
    if len(df) < n:
        return [f"Need at least {n} rows for one window — only {len(df)} available."]
    tail = df[channels].iloc[-n:].to_numpy(dtype=np.float64)
    bad = [c for c, ok in zip(channels, np.isfinite(tail).all(axis=0)) if not ok]
    return [f"Non-finite value(s) (NaN or ±inf) in the last {n} rows of: {', '.join(bad)}."] if bad else []


def ood_availability(metrics: dict) -> tuple[bool, str]:
    """(available, reason). Available only when the run carries a detector fitted by ai/ood.fit_ood:
    a known method with a finite threshold calibrated on the validation split."""
    ood = metrics.get("ood")
    if not ood:
        return False, "This model has no fitted OOD detector (metrics.json has no 'ood' block)."
    method = ood.get("method")
    if method not in METHODS:
        return False, f"Unknown OOD method {method!r} in metrics.json."
    threshold = (ood.get(method) or {}).get("threshold")
    if threshold is None or not np.isfinite(threshold):
        return False, f"The {method} detector has no calibrated threshold."
    if (ood.get("fitted_on") or {}).get("thresholds") != "val":
        return False, "The OOD threshold was not calibrated on the validation split."
    return True, f"{method} score, threshold at {ood.get('target_coverage', 0.95):.0%} in-distribution coverage on the validation split."


@dataclasses.dataclass(frozen=True)
class ConditionAssessment:
    state: str
    problems: tuple = ()
    classes: tuple = ()
    probs: tuple = ()
    top_class: str | None = None
    confidence: float | None = None
    calibrated: bool = False
    abstain_below: float | None = None
    ood_available: bool = False
    ood_reason: str = ""
    ood_method: str | None = None
    ood_score: float | None = None
    ood_threshold: float | None = None

    @property
    def label(self) -> str:
        return LABELS[self.state]

    @property
    def explanation(self) -> str:
        return EXPLANATIONS[self.state]

    @property
    def suggested_class(self) -> str | None:
        """The class the GUI may name as a suggestion: only for ``known``."""
        return self.top_class if self.state == KNOWN else None


def invalid(problems: list[str], metrics: dict | None = None) -> ConditionAssessment:
    available, reason = ood_availability(metrics or {})
    return ConditionAssessment(INPUT_INVALID, problems=tuple(problems), ood_available=available, ood_reason=reason)


def assess(raw_probs: np.ndarray, classes: list, metrics: dict, ood_score: float | None) -> ConditionAssessment:
    """Assessment of one valid window from the model's raw probabilities and its OOD score (None when
    the run has no detector; a score passed for a run without one is ignored)."""
    available, reason = ood_availability(metrics)
    score = float(ood_score) if (available and ood_score is not None) else None
    label, p = decide(np.asarray(raw_probs), list(classes), metrics, score)
    k = int(np.argmax(p))
    state = STATE_UNKNOWN if label == UNKNOWN else STATE_UNCERTAIN if label == UNCERTAIN else KNOWN
    ood = metrics.get("ood") or {}
    cal = metrics.get("calibration")
    return ConditionAssessment(
        state, classes=tuple(classes), probs=tuple(float(x) for x in p), top_class=classes[k], confidence=float(p[k]),
        calibrated=cal is not None, abstain_below=cal["abstain_below"] if cal else None,
        ood_available=available, ood_reason=reason, ood_method=ood.get("method") if available else None, ood_score=score,
        ood_threshold=float(ood[ood["method"]]["threshold"]) if available else None,
    )
