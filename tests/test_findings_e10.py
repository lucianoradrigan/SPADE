"""Findings registered from expansion tasks E10a/E10b (round 2026-10-21, T1): AI-N15, AI-N16, AI-N17
(docs/AI_TECHNICAL_AUDIT.md §9) and S-14 (docs/SIMULATION_STABILITY_REPORT.md §6).

Open defects are strict xfails with their ID: they start passing -- and so fail as XPASS -- only when
the defect is fixed and the evidence regenerated. Documented behaviour is pinned by plain tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PINN_REPORT = ROOT / "docs" / "audit" / "ai" / "pinn_identification.json"


def _pinn_summary():
    return pd.DataFrame(json.loads(PINN_REPORT.read_text())["summary"])


@pytest.mark.xfail(strict=True, reason="AI-N15 (open): the PINN estimates C systematically low; mean error -9 % at 5 % noise")
def test_ai_n15_pinn_capacitance_unbiased_with_noise():
    """Recorded evidence: every PINN C estimate at 1-5 % noise is below the truth (report runs)."""
    runs = pd.DataFrame(json.loads(PINN_REPORT.read_text())["runs"])
    c = runs[(runs.estimator == "pinn") & (runs.param == "C") & (runs.noise_pct == 5.0)]
    assert abs(c["rel_error_pct"].mean()) < 2.0


def test_ai_n15_bias_is_recorded_as_negative_and_growing_with_noise():
    runs = pd.DataFrame(json.loads(PINN_REPORT.read_text())["runs"])
    c = runs[(runs.estimator == "pinn") & (runs.param == "C")]
    mean = c.groupby("noise_pct")["rel_error_pct"].mean()
    assert (mean < 0).all() and list(mean) == sorted(mean, reverse=True)


@pytest.mark.xfail(strict=True, reason="AI-N16 (open): the PINN 5-95 % bootstrap interval covers the truth in 52-78 % of noisy runs, not 90 %")
def test_ai_n16_pinn_interval_has_nominal_coverage():
    s = _pinn_summary()
    noisy = s[(s.estimator == "pinn") & (s.noise_pct > 0)]
    assert (noisy["coverage"] >= 0.85).all()


def test_ai_n16_gui_says_coverage_not_validated():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "driveflow" / "viz" / "app.py"), default_timeout=300)
    at.run()
    at.switch_page("pages/vsc_dpc.py").run()
    at.select_slider(key="pid_noise").set_value(1.0).run()
    at.radio(key="pid_estimator").set_value("least_squares").run()
    at.number_input(key="pid_nboot_least_squares").set_value(10).run()
    at.button(key="pid_run").click().run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "Coverage not validated." in text and "AI-N16" in text


def test_ai_n17_promoted_classifier_has_no_ood_detector():
    """Documented (D6): the promoted dc_motor PC classifier carries no 'ood' block, so the Unknown
    condition cannot be reported for it. Changing that is a decision, not a fix."""
    from driveflow.ai.condition import ood_availability
    from driveflow.ai.registry import resolve

    metrics = json.loads((resolve("dc_motor", "pc", "classifier") / "metrics.json").read_text())
    available, reason = ood_availability(metrics)
    assert not available and "no fitted OOD detector" in reason


@pytest.mark.simulation
@pytest.mark.parametrize("c_scale, diverges", [(0.5, False), (0.4, True)])
def test_s14_dpc_loop_vs_capacitor_at_the_nominal_load(c_scale, diverges):
    """S-14 (documented): at 8.0064 ohm the DPC loop is stable at C x0.5 and diverges at C x0.4."""
    from driveflow.datagen import Scenario, run_scenario

    with np.errstate(all="ignore"):
        df = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="s14", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.1,
                                                             vsc_capacitor_scale=c_scale)))
    vc = np.hypot(df["vc_real"].to_numpy(), df["vc_imag"].to_numpy())
    blew_up = bool((not np.isfinite(vc).all()) or np.nanmax(vc[len(vc) // 2:]) > 500.0)
    assert blew_up is diverges
