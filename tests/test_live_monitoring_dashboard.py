"""Tests for the Live Monitoring dashboard tab (Phase 5): AppTest exercises the landing page and
the full "Run monitoring" flow (simulation-based detector + rule-based GatewayAgent, both feeding
one ServerAgent) for both domains, plus a real (if slow -- see the module's own note) check that
the vsc_dpc rule's hysteresis actually gates firing across two runs.
"""

import time
from pathlib import Path

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
