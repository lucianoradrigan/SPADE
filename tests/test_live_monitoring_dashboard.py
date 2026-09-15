"""Tests for the Live Monitoring dashboard tab (Phase 5): AppTest exercises the landing page and
the full "Run monitoring" flow (simulation-based detector + rule-based GatewayAgent, both feeding
one ServerAgent) for both domains, plus a real (if slow -- see the module's own note) check that
the vsc_dpc rule's hysteresis actually gates firing across two runs.
"""

import time
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "dashboard.py")


class TestLandingPage:
    def test_lm_card_is_present(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        assert at.button(key="enter_phase_LM").label == "Enter Fase LM →"


class TestRunMonitoring:
    def test_dc_motor_renders_and_runs_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_LM").click().run()
        assert not at.exception
        at.sidebar.button(key="lm_run_button").click().run()
        assert not at.exception
        status_texts = [el.value for el in at.success] + [el.value for el in at.error]
        assert any("Status:" in text for text in status_texts)

    def test_vsc_dpc_renders_and_runs_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_LM").click().run()
        at.sidebar.selectbox(key="lm_domain").set_value("vsc_dpc").run()
        assert not at.exception
        at.sidebar.button(key="lm_run_button").click().run()
        assert not at.exception

    def test_alert_history_accumulates_across_multiple_runs(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_LM").click().run()
        at.sidebar.button(key="lm_run_button").click().run()
        first_rows = at.dataframe[0].value.shape[0] if at.dataframe else 0
        at.sidebar.button(key="lm_run_button").click().run()
        second_rows = at.dataframe[0].value.shape[0] if at.dataframe else 0
        assert second_rows > first_rows


class TestRuleHysteresis:
    """Real timing, not mocked -- confirms the vsc_dpc divergence rule (hysteresis_seconds: 2)
    genuinely doesn't fire on the very first observation of an out-of-range value, and does once
    it's held for >= 2 real seconds across two 'Run monitoring' clicks -- exactly why a user
    clicking once with R in [1, 3]Ω and seeing "Status: OK" is expected, not broken (the module's
    own caption says as much)."""

    def test_does_not_fire_on_the_first_observation_but_does_on_the_second(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_LM").click().run()
        at.sidebar.selectbox(key="lm_domain").set_value("vsc_dpc").run()
        at.sidebar.checkbox(key="lm_load_r_custom").set_value(True).run()
        at.sidebar.number_input(key="lm_load_r_custom_input").set_value(2.0).run()

        at.sidebar.button(key="lm_run_button").click().run()
        assert not at.exception
        assert any("Status: OK" in s.value for s in at.success)

        time.sleep(2.2)
        at.sidebar.button(key="lm_run_button").click().run()
        assert not at.exception
        assert any("Status: ALERT" in e.value for e in at.error)
        assert any("fired" in e.value for e in at.error)


class TestClassifierConfidenceDrift:
    """The third Agent Consensus pillar (Sec. 1.2: rule-based + simulation-based + ML confidence
    drift). dc_motor has a real promoted PC-tier classifier to test against; vsc_dpc doesn't
    (Sec. 8's own status note), which must be reported clearly, not silently skipped.

    The "eventually reports drift" behaviour needs > confidence_window (20) recorded
    confidence points, which would mean 22 full AppTest reruns (each re-executing the whole
    dashboard script, including a real physics simulation) if driven through the UI -- too slow
    for a test suite (timed out past 180s the first time this was tried). So that part is
    exercised directly against _record_classifier_confidence_if_available + a real ServerAgent,
    no Streamlit involved, and only a single UI run is checked through AppTest as a smoke test
    that the wiring itself doesn't crash and renders the right widgets."""

    def test_dc_motor_accumulates_confidence_and_eventually_reports_drift(self):
        from driveflow.datagen import Scenario, run_scenario
        from driveflow.monitoring.agents.agent_server import ServerAgent
        from driveflow.viz.live_monitoring_dashboard import _record_classifier_confidence_if_available

        server_agent = ServerAgent()
        confidence = None
        for i in range(22):  # > ServerAgent's default confidence_window (20)
            records = run_scenario(Scenario(scenario_id=f"drift_{i}", fault_type=None, duration_s=0.15, seed=i))
            df = pd.DataFrame.from_records(records)
            confidence = _record_classifier_confidence_if_available("dc_motor", df, server_agent)
        assert confidence is not None
        drift = server_agent.check_confidence_drift("dc_motor")
        assert drift is not None
        assert 0.0 <= drift.baseline_mean <= 1.0
        assert 0.0 <= drift.recent_mean <= 1.0

    def test_dc_motor_ui_smoke_renders_confidence_metric_on_a_single_run(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_LM").click().run()
        at.sidebar.button(key="lm_run_button").click().run()
        assert not at.exception
        metric_values = [m.value for m in at.metric]
        assert any("%" in v for v in metric_values)  # "This run's confidence"

    def test_vsc_dpc_reports_no_classifier_available(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_LM").click().run()
        at.sidebar.selectbox(key="lm_domain").set_value("vsc_dpc").run()
        at.sidebar.button(key="lm_run_button").click().run()
        assert not at.exception
        caption_texts = " ".join(c.value for c in at.caption)
        assert "No promoted PC-tier classifier" in caption_texts
