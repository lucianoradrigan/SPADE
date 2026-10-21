"""Findings of expansion task E14 (docs/AI_TECHNICAL_AUDIT.md §10): AI-N18, AI-N19, AI-N20, read from
the recorded report so a regenerated report updates them."""

import json
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "ai" / "dc_motor_residual_anomaly.json"


def _rep():
    return json.loads(REPORT.read_text())


@pytest.mark.xfail(strict=True, reason="AI-N18 (open): start-up windows inflate the residual threshold; ~50 % of faulted windows flagged at AUROC 1.0")
def test_ai_n18_gui_detector_flags_most_faulted_windows():
    c2 = _rep()["variants"]["with_voltage_v"]["criterion_2"]["per_fault"]
    assert min(c2[f]["detection_rate_x1"] for f in ("outer_race", "inner_race")) >= 0.9


def test_ai_n19_steady_state_persistence_is_exact():
    """Documented: healthy steady-state rpm is constant in every held-out condition."""
    rep = _rep()["variants"]["with_voltage_v"]
    assert rep["criterion_1"]["pooled"]["rpm"]["ratio"] is None and "rpm" in rep["criterion_1"]["steady_state_undefined_channels"]


@pytest.mark.xfail(strict=True, reason="AI-N20 (open): distilled edge residual detectors flag almost no faulted window")
def test_ai_n20_edge_detectors_usable():
    rep = _rep()["variants"]
    for name in ("rpi5_tcn_distilled", "esp32_distilled"):
        c2 = rep[name]["criterion_2"]["per_fault"]
        assert min(c2[f]["detection_rate_x1"] for f in ("outer_race", "inner_race")) >= 0.5


def test_criteria_of_the_gui_variant_recorded():
    rep = _rep()
    v = rep["variants"]["with_voltage_v"]
    assert v["criterion_1"]["pass"] and v["criterion_2"]["pass"] and v["criterion_3"]["pass"]
    assert rep["design"]["horizon"] < rep["design"]["half_period_samples"]
    assert all(e["exported"] and not e["flex_ops"] for e in rep["edge"])
