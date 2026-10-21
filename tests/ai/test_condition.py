"""Decision rule of ai/condition.py (expansion task E10a, spec Sec. 8.2)."""

import numpy as np
import pandas as pd
import pytest

from driveflow.ai import condition

CLASSES = ["inner_race", "normal", "outer_race"]
OOD = {"method": "mahalanobis", "target_coverage": 0.95, "fitted_on": {"statistics": "train", "thresholds": "val"},
       "mahalanobis": {"threshold": 10.0}, "energy": {"threshold": -1.0}}
CAL = {"temperature": 1.0, "abstain_below": 0.6}


def test_known_when_confident_and_in_distribution():
    a = condition.assess(np.array([0.05, 0.9, 0.05]), CLASSES, {"ood": OOD, "calibration": CAL}, ood_score=3.0)
    assert a.state == condition.KNOWN and a.suggested_class == "normal" and a.label == "Known condition"
    assert a.ood_available and a.ood_score == 3.0 and a.ood_threshold == 10.0


def test_unknown_overrides_high_confidence():
    a = condition.assess(np.array([0.01, 0.98, 0.01]), CLASSES, {"ood": OOD, "calibration": CAL}, ood_score=50.0)
    assert a.state == condition.STATE_UNKNOWN and a.suggested_class is None
    assert a.label == "Unknown condition — REQUIRES VERIFICATION" and a.top_class == "normal"


def test_uncertain_below_abstention_threshold():
    a = condition.assess(np.array([0.4, 0.35, 0.25]), CLASSES, {"ood": OOD, "calibration": CAL}, ood_score=3.0)
    assert a.state == condition.STATE_UNCERTAIN and a.suggested_class is None
    assert a.explanation == "The classifier cannot distinguish reliably among its known classes."


def test_the_three_states_are_distinguishable():
    assert len({condition.LABELS[s] for s in (condition.KNOWN, condition.STATE_UNCERTAIN, condition.STATE_UNKNOWN, condition.INPUT_INVALID)}) == 4


@pytest.mark.parametrize("ood, reason", [
    (None, "no fitted OOD detector"),
    ({**OOD, "method": "knn"}, "Unknown OOD method"),
    ({**OOD, "mahalanobis": {"threshold": float("nan")}}, "no calibrated threshold"),
    ({**OOD, "fitted_on": {"thresholds": "test"}}, "not calibrated on the validation split"),
])
def test_unavailable_detector_never_yields_unknown(ood, reason):
    metrics = {"calibration": CAL, **({"ood": ood} if ood else {})}
    a = condition.assess(np.array([0.01, 0.98, 0.01]), CLASSES, metrics, ood_score=1e9)
    assert not a.ood_available and reason in a.ood_reason
    assert a.state == condition.KNOWN and a.ood_score is None and a.ood_threshold is None


def test_window_problems():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    assert condition.window_problems(df, ["a", "b"], 8) == []
    assert "Missing channel" in condition.window_problems(df, ["a", "c"], 8)[0]
    assert "Need at least 20 rows" in condition.window_problems(df, ["a"], 20)[0]
    df.loc[9, "b"] = np.inf
    assert "Non-finite" in condition.window_problems(df, ["a", "b"], 8)[0] and ": b." in condition.window_problems(df, ["a", "b"], 8)[0]
    df.loc[0, "a"] = np.nan  # outside the last 8 rows: not part of the input
    assert condition.window_problems(df, ["a"], 8) == []


def test_invalid_assessment_predicts_nothing():
    a = condition.invalid(["Missing channel(s) this model needs: x."], {"ood": OOD})
    assert a.state == condition.INPUT_INVALID and a.top_class is None and a.probs == () and a.ood_available
