"""Monitoring agents and alerts -- docs/AI_TECHNICAL_AUDIT.md Sec. 5.

Separates the three stages the platform must keep apart: DETECTION (GatewayAgent fires),
ACKNOWLEDGEMENT (operator has seen it -- ServerAgent.acknowledge) and RESOLUTION (the condition
went away -- a resolved Alert). Only resolution changes health.
"""

from pathlib import Path

import numpy as np
import pytest

from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
from driveflow.datagen import Scenario, run_scenario
from driveflow.monitoring.agents.agent_gateway import Alert, GatewayAgent
from driveflow.monitoring.agents.agent_server import ServerAgent
from driveflow.monitoring.rules.schema import Rule, RuleSet, load_ruleset
from driveflow.sim.vsc_system import load_feedback_spectral_radius

pytestmark = pytest.mark.ai

RULES = Path(__file__).resolve().parents[2] / "src" / "driveflow" / "monitoring" / "rules" / "vsc_dpc.yaml"


def _fires(agent: GatewayAgent, telemetry: dict, hysteresis: float) -> bool:
    agent.step(telemetry, 0.0)
    agent.step(telemetry, hysteresis)
    return bool(agent.active_alerts())


class TestVscRuleThresholdIsPhysical:
    """AI-N03: the shipped rule watched R in [1, 3] ohm; the plant is open-loop unstable for every
    R < R* = 3.3707 ohm (sim.vsc_system.load_feedback_spectral_radius)."""

    @pytest.mark.parametrize("r_ohm", [0.2, 0.5, 0.99, 1.0, 2.0, 3.0, 3.2, 3.36, 3.38, 3.5, 4.0, 8.0064, 20.0])
    def test_rule_fires_iff_the_plant_is_open_loop_unstable(self, r_ohm):
        ruleset = load_ruleset(RULES)
        rule = ruleset.rules[0]
        fired = _fires(GatewayAgent(ruleset), {"load_resistance_ohm": r_ohm}, rule.hysteresis_seconds)
        assert fired == (load_feedback_spectral_radius(r_ohm) > 1.0)


def _gateway(hysteresis=2.0):
    return GatewayAgent(RuleSet("t", (Rule("high_x", "x > 1.0", "high", hysteresis, "alert"),)))


class TestHysteresisAndDebounce:
    def test_persistent_condition_fires_once(self):
        g = _gateway()
        events = [e for k in range(100) for e in g.step({"x": 5.0}, k * 0.1)]
        assert len(events) == 1 and not events[0].resolved

    def test_chattering_condition_shorter_than_hysteresis_never_fires(self):
        """Bounded alarm rate: a condition toggling every 0.5 s (< 2 s hysteresis) raises
        nothing. Flip side, documented as AI-N08: an intermittent real fault whose condition
        never holds for 2 s continuously is never reported (no off-delay / deadband)."""
        g = _gateway()
        events = [e for k in range(200) for e in g.step({"x": 5.0 if (k // 5) % 2 == 0 else 0.0}, k * 0.1)]
        assert events == []

    def test_without_off_delay_one_clean_sample_clears_an_active_alert(self):
        """clear_hysteresis_seconds = 0 (the schema default) keeps the original behaviour."""
        g = _gateway()
        g.step({"x": 5.0}, 0.0)
        g.step({"x": 5.0}, 2.0)
        events = g.step({"x": 0.5}, 2.1)
        assert len(events) == 1 and events[0].resolved


class TestOffDelay:
    """AI-N08 (fixed): an optional clear_hysteresis_seconds keeps an active alert until the
    condition has been false for that long."""

    def _g(self):
        return GatewayAgent(RuleSet("t", (Rule("high_x", "x > 1.0", "high", 2.0, "alert", clear_hysteresis_seconds=1.0),)))

    def test_short_clean_gap_does_not_resolve(self):
        g = self._g()
        g.step({"x": 5.0}, 0.0)
        assert g.step({"x": 5.0}, 2.0)[0].resolved is False
        assert g.step({"x": 0.5}, 2.1) == [] and g.step({"x": 0.5}, 2.9) == []
        assert g.step({"x": 5.0}, 3.0) == []  # the condition came back: still the same alert
        assert g.active_alerts() == ["high_x"]

    def test_clean_for_the_off_delay_resolves_once(self):
        g = self._g()
        g.step({"x": 5.0}, 0.0)
        g.step({"x": 5.0}, 2.0)
        events = [e for k in range(30) for e in g.step({"x": 0.5}, 2.1 + 0.05 * k)]
        assert len(events) == 1 and events[0].resolved and events[0].timestamp == pytest.approx(3.1)
        assert g.active_alerts() == []

    def test_negative_off_delay_is_rejected(self):
        from driveflow.monitoring.rules.schema import RuleValidationError

        with pytest.raises(RuleValidationError, match="clear_hysteresis_seconds"):
            Rule("r", "x > 0", "low", 0.0, "alert", clear_hysteresis_seconds=-1.0)


class TestShippedVscRuleTiming:
    """AI-N02 (fixed): evaluated on telemetry time, the shipped rule fires within one 0.05 s
    dashboard run (500 samples at tau = 1e-4 s)."""

    def test_fires_within_one_dashboard_run(self):
        ruleset = load_ruleset(RULES)
        g = GatewayAgent(ruleset)
        events = [e for k in range(500) for e in g.step({"load_resistance_ohm": 2.0}, k * 1e-4)]
        assert len(events) == 1 and not events[0].resolved
        assert events[0].timestamp <= 0.05


class TestNonFiniteTelemetry:
    """AI-N04: NaN made every comparison False and silently RESOLVED active alerts."""

    def test_nan_does_not_resolve_an_active_alert(self):
        g = _gateway()
        g.step({"x": 5.0}, 0.0)
        g.step({"x": 5.0}, 2.0)
        assert g.step({"x": float("nan")}, 2.1) == []
        assert g.active_alerts() == ["high_x"] and g.indeterminate_rules() == ["high_x"]

    def test_nan_does_not_fire_and_breaks_hysteresis_continuity(self):
        g = _gateway()
        g.step({"x": 5.0}, 0.0)
        g.step({"x": float("inf")}, 1.0)
        assert g.step({"x": 5.0}, 2.0) == []  # timer restarted at 2.0
        assert len(g.step({"x": 5.0}, 4.0)) == 1

    def test_recovers_when_finite_data_returns(self):
        g = _gateway()
        g.step({"x": float("nan")}, 0.0)
        g.step({"x": 0.0}, 0.1)
        assert g.indeterminate_rules() == [] and g.active_alerts() == []


class TestDetectionAcknowledgementResolution:
    def test_acknowledgement_does_not_resolve_and_does_not_carry_over_to_a_new_occurrence(self):
        s = ServerAgent()
        s.record_alert("vsc_dpc", Alert("r", "high", "alert", 1.0))
        assert s.acknowledge("vsc_dpc", "r", by="operator")
        assert s.domain_health("vsc_dpc") == "CRITICAL"  # seen, still active
        s.record_alert("vsc_dpc", Alert("r", "high", "alert", 2.0, resolved=True))
        assert s.domain_health("vsc_dpc") == "HEALTHY"
        s.record_alert("vsc_dpc", Alert("r", "high", "alert", 3.0))
        (active,) = s.active_alerts("vsc_dpc")
        assert active.acknowledged_by is None and active.first_seen == 3.0

    def test_repeated_reports_do_not_duplicate_active_alerts_but_are_all_logged(self):
        s = ServerAgent()
        for t in (1.0, 2.0, 3.0):
            s.record_alert("dc_motor", Alert("r", "medium", "alert", t))
        assert len(s.active_alerts("dc_motor")) == 1 and s.active_alerts("dc_motor")[0].last_seen == 3.0
        assert len(s.alert_history("dc_motor")) == 3

    def test_no_event_lost_across_two_sources_with_the_same_rule(self):
        s = ServerAgent()
        s.record_alert("dc_motor", Alert("r", "high", "alert", 1.0), source_id="a")
        s.record_alert("dc_motor", Alert("r", "medium", "alert", 1.0), source_id="b")
        s.record_alert("dc_motor", Alert("r", "high", "alert", 2.0, resolved=True), source_id="a")
        assert [a.source_id for a in s.active_alerts("dc_motor")] == ["b"]
        assert s.domain_health("dc_motor") == "WARNING"


def test_dc_agent_features_exist_in_the_telemetry_schema():
    """AI-N05: the agent declared 'i_a', which no telemetry frame has; detect_anomaly() silently
    dropped it, so the current channel never contributed to the anomaly score."""
    rec = run_scenario(Scenario(scenario_id="schema", duration_s=0.001))[0]
    missing = [f for f in DCMotorAnomalyDetector().monitored_features if f not in rec]
    assert missing == []


def test_dc_agent_score_is_finite_and_bounded_for_a_real_window():
    agent = DCMotorAnomalyDetector()
    recs = run_scenario(Scenario(scenario_id="agent_win", fault_type="outer_race", duration_s=0.03))
    telemetry = {f: np.array([r[f] for r in recs]) for f in agent.monitored_features}
    score, closest = agent.detect_anomaly(telemetry)
    assert 0.0 <= score <= 1.0 and closest in agent.HYPOTHESES
