"""Tests for the Sec. 8 step 9 monitoring agents (docs/design_ai_layer_transversal.md Sec. 4.3):
GatewayAgent (Raspberry Pi 5 tier -- stateful rule evaluation with hysteresis/debounce) and
ServerAgent (PC tier -- cross-domain alert aggregation and classifier confidence drift)."""

import pytest

from driveflow.monitoring.agents import Alert, ConfidenceDriftReport, GatewayAgent, ServerAgent, TelemetryFieldMissing
from driveflow.monitoring.rules.schema import Rule, RuleSet


def _ruleset(hysteresis_seconds=0.0):
    return RuleSet(
        domain="vsc_dpc",
        rules=(
            Rule(
                name="load_resistance_low",
                condition="load_resistance_ohm >= 1.0 and load_resistance_ohm <= 3.0",
                severity="high",
                hysteresis_seconds=hysteresis_seconds,
                action="alert",
            ),
        ),
    )


class TestGatewayAgentHysteresis:
    def test_no_alert_while_condition_is_false(self):
        agent = GatewayAgent(_ruleset())
        events = agent.step({"load_resistance_ohm": 5.0}, timestamp=0.0)
        assert events == []
        assert agent.active_alerts() == []

    def test_fires_immediately_when_hysteresis_is_zero(self):
        agent = GatewayAgent(_ruleset(hysteresis_seconds=0.0))
        events = agent.step({"load_resistance_ohm": 2.0}, timestamp=0.0)
        assert len(events) == 1
        assert events[0] == Alert("load_resistance_low", "high", "alert", 0.0, resolved=False)
        assert agent.active_alerts() == ["load_resistance_low"]

    def test_does_not_fire_before_hysteresis_elapses(self):
        agent = GatewayAgent(_ruleset(hysteresis_seconds=2.0))
        assert agent.step({"load_resistance_ohm": 2.0}, timestamp=0.0) == []
        assert agent.step({"load_resistance_ohm": 2.0}, timestamp=1.0) == []
        assert agent.active_alerts() == []

    def test_fires_once_hysteresis_elapses(self):
        agent = GatewayAgent(_ruleset(hysteresis_seconds=2.0))
        agent.step({"load_resistance_ohm": 2.0}, timestamp=0.0)
        agent.step({"load_resistance_ohm": 2.0}, timestamp=1.0)
        events = agent.step({"load_resistance_ohm": 2.0}, timestamp=2.0)
        assert len(events) == 1
        assert events[0].resolved is False

    def test_debounce_does_not_refire_while_condition_stays_true(self):
        agent = GatewayAgent(_ruleset(hysteresis_seconds=0.0))
        agent.step({"load_resistance_ohm": 2.0}, timestamp=0.0)
        events = agent.step({"load_resistance_ohm": 2.0}, timestamp=1.0)
        assert events == []
        assert agent.active_alerts() == ["load_resistance_low"]

    def test_clears_when_condition_goes_false(self):
        agent = GatewayAgent(_ruleset(hysteresis_seconds=0.0))
        agent.step({"load_resistance_ohm": 2.0}, timestamp=0.0)
        events = agent.step({"load_resistance_ohm": 5.0}, timestamp=1.0)
        assert len(events) == 1
        assert events[0].resolved is True
        assert agent.active_alerts() == []

    def test_hysteresis_timer_resets_after_a_false_reading(self):
        agent = GatewayAgent(_ruleset(hysteresis_seconds=2.0))
        agent.step({"load_resistance_ohm": 2.0}, timestamp=0.0)
        agent.step({"load_resistance_ohm": 5.0}, timestamp=1.0)  # condition drops -- timer resets
        events = agent.step({"load_resistance_ohm": 2.0}, timestamp=2.0)  # only 0s held since reset
        assert events == []

    def test_missing_telemetry_field_raises(self):
        agent = GatewayAgent(_ruleset())
        with pytest.raises(TelemetryFieldMissing):
            agent.step({"some_other_field": 1.0}, timestamp=0.0)


class TestServerAgentAlertAggregation:
    def test_domain_status_defaults_to_ok(self):
        server = ServerAgent()
        assert server.domain_status("dc_motor") == "ok"

    def test_unresolved_alert_sets_domain_status_to_alert(self):
        server = ServerAgent()
        server.record_alert("vsc_dpc", Alert("load_resistance_low", "high", "alert", 0.0))
        assert server.domain_status("vsc_dpc") == "alert"
        assert server.domain_status("dc_motor") == "ok"  # domains stay independent

    def test_resolved_alert_clears_domain_status(self):
        server = ServerAgent()
        server.record_alert("vsc_dpc", Alert("load_resistance_low", "high", "alert", 0.0))
        server.record_alert("vsc_dpc", Alert("load_resistance_low", "high", "alert", 1.0, resolved=True))
        assert server.domain_status("vsc_dpc") == "ok"

    def test_alert_history_is_ordered_and_filterable_by_domain(self):
        server = ServerAgent()
        server.record_alert("vsc_dpc", Alert("r1", "high", "alert", 0.0))
        server.record_alert("dc_motor", Alert("r2", "medium", "alert", 1.0))
        assert [e["domain"] for e in server.alert_history()] == ["vsc_dpc", "dc_motor"]
        assert [e["rule_name"] for e in server.alert_history(domain="dc_motor")] == ["r2"]


class TestServerAgentConfidenceDrift:
    def test_returns_none_without_enough_history(self):
        server = ServerAgent(confidence_window=5, baseline_window=50)
        for _ in range(5):
            server.record_classifier_confidence("dc_motor", 0.9)
        assert server.check_confidence_drift("dc_motor") is None

    def test_returns_none_for_unknown_domain(self):
        server = ServerAgent()
        assert server.check_confidence_drift("dc_motor") is None

    def test_flags_drift_when_recent_confidence_drops(self):
        server = ServerAgent(confidence_window=5, baseline_window=50, drift_threshold=0.15)
        for _ in range(10):
            server.record_classifier_confidence("dc_motor", 0.95)
        for _ in range(5):
            server.record_classifier_confidence("dc_motor", 0.5)
        report = server.check_confidence_drift("dc_motor")
        assert isinstance(report, ConfidenceDriftReport)
        assert report.suggest_retraining is True
        assert report.recent_mean < report.baseline_mean

    def test_does_not_flag_drift_when_confidence_is_stable(self):
        server = ServerAgent(confidence_window=5, baseline_window=50, drift_threshold=0.15)
        for _ in range(15):
            server.record_classifier_confidence("dc_motor", 0.9)
        report = server.check_confidence_drift("dc_motor")
        assert report.suggest_retraining is False

    def test_domains_track_independent_confidence_history(self):
        server = ServerAgent(confidence_window=5, baseline_window=50, drift_threshold=0.15)
        for _ in range(10):
            server.record_classifier_confidence("dc_motor", 0.95)
        for _ in range(5):
            server.record_classifier_confidence("dc_motor", 0.5)
        for _ in range(15):
            server.record_classifier_confidence("vsc_dpc", 0.9)
        assert server.check_confidence_drift("dc_motor").suggest_retraining is True
        assert server.check_confidence_drift("vsc_dpc").suggest_retraining is False

    def test_confidence_window_must_be_smaller_than_baseline_window(self):
        with pytest.raises(ValueError):
            ServerAgent(confidence_window=50, baseline_window=50)
