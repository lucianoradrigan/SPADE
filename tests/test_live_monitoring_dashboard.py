"""Tests for the Live Monitoring dashboard tab (Phase 5): AppTest exercises the landing page and
the full "Run monitoring" flow (simulation-based detector + rule-based GatewayAgent, both feeding
one ServerAgent) for both domains, plus a real (if slow -- see the module's own note) check that
the vsc_dpc rule's hysteresis actually gates firing across two runs.
"""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.switch_page("pages/monitoring.py").run()
    return at


def _kpi_text(at) -> str:
    return " ".join(str(getattr(h, "body", "")) for h in at.get("html"))


class TestRunMonitoring:
    def test_dc_motor_renders_and_runs_without_exceptions(self):
        at = _open()
        assert not at.exception
        at.button(key="lm_run_button").click().run()
        assert not at.exception
        assert "Overall domain status" in _kpi_text(at)
        assert "SIMULATED MONITORING" in _kpi_text(at)

    def test_vsc_dpc_renders_and_runs_without_exceptions(self):
        at = _open()
        at.selectbox(key="lm_domain").set_value("vsc_dpc").run()
        assert not at.exception
        at.button(key="lm_run_button").click().run()
        assert not at.exception

    def test_alert_history_accumulates_across_multiple_runs(self):
        at = _open()
        at.button(key="lm_run_button").click().run()
        first_rows = at.dataframe[0].value.shape[0] if at.dataframe else 0
        at.button(key="lm_run_button").click().run()
        second_rows = at.dataframe[0].value.shape[0] if at.dataframe else 0
        assert second_rows > first_rows


class TestRuleHysteresis:
    """AI_TECHNICAL_AUDIT AI-N02: the gateway runs on the simulated run's own timeline, so a single
    'Run monitoring' click with R < R* fires the vsc_dpc divergence rule (10 ms hysteresis inside a
    0.05 s run) regardless of how long the operator waits between clicks. It used to need a second
    click at least 2 wall-clock seconds later. A following pass with a stable R resolves it."""

    def test_fires_on_the_first_pass_and_resolves_on_a_stable_pass(self):
        at = _open()
        at.selectbox(key="lm_domain").set_value("vsc_dpc").run()
        at.checkbox(key="lm_load_r_custom").set_value(True).run()
        at.number_input(key="lm_load_r_custom_input").set_value(2.0).run()

        at.button(key="lm_run_button").click().run()
        assert not at.exception
        assert any("dpc_load_resistance_divergence: fired" in e.value for e in at.error)
        assert "CRITICAL" in _kpi_text(at)  # severity high -> CRITICAL, from the active-alert set

        at.number_input(key="lm_load_r_custom_input").set_value(8.0064).run()
        at.button(key="lm_run_button").click().run()
        assert not at.exception
        assert "CRITICAL" not in _kpi_text(at)


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
        from driveflow.viz.live_monitoring_dashboard import _record_classifier_confidence_if_available  # legacy shim, still supported

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
        at = _open()
        at.button(key="lm_run_button").click().run()
        assert not at.exception
        metric_values = [m.value for m in at.metric]
        assert any("%" in v for v in metric_values)  # "This run's confidence"
        assert "temperature-calibrated" in _kpi_text(at)  # the promoted run records its calibration (AI-N12)

    def test_vsc_dpc_reports_no_classifier_available(self):
        at = _open()
        at.selectbox(key="lm_domain").set_value("vsc_dpc").run()
        at.button(key="lm_run_button").click().run()
        assert not at.exception
        caption_texts = " ".join(c.value for c in at.caption)
        assert "No promoted PC-tier classifier" in caption_texts


class TestStatusHonesty:
    def test_status_is_unknown_before_the_first_pass(self):
        at = _open()
        assert not at.exception
        assert "UNKNOWN" in _kpi_text(at)
        assert "HEALTHY" not in _kpi_text(at).split("Overall domain status")[1][:200]

    def test_acknowledge_does_not_clear_an_active_alert(self):
        from driveflow.monitoring.agents.agent_gateway import Alert
        from driveflow.monitoring.agents.agent_server import ServerAgent

        at = AppTest.from_file(APP_PATH, default_timeout=120)
        sa = ServerAgent()
        sa.record_alert("dc_motor", Alert("rule_x", "high", "alert", 1.0))
        at.session_state["lm_server_agent"] = sa
        at.run()
        at.switch_page("pages/monitoring.py").run()
        at.button(key="lm_ack_rule_x_default").click().run()
        assert not at.exception
        assert "CRITICAL" in _kpi_text(at)
