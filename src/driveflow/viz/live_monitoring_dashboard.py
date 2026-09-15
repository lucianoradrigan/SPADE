"""Live Monitoring tab (Phase 5): runs a domain's simulation-based anomaly detector
(driveflow.agents) AND, where a real rule exists (today: only vsc_dpc), its rule-based
GatewayAgent (driveflow.monitoring.agents) against one sample telemetry run, feeds both into one
session-scoped ServerAgent as the "Agent Consensus" (a per-domain status badge + alert history --
Sec. 5.3's own design), and shows the simulation-based agent's own feature-importance explanation.

Kept in its own module -- same reasoning as ai_dashboard.py/transfer_learning_dashboard.py's own
docstrings -- so dashboard.py only gains a landing-page card + one-line dispatch. Must not import
from driveflow.viz.dashboard.

Integration note: ServerAgent.record_alert() already accepts "an agent_gateway.Alert or any
duck-typed equivalent (e.g. a future ESP32 watchdog event)" per its own docstring -- a
simulation-based agent's (score, diagnosis) is exactly that kind of equivalent event once
converted to an Alert via _alert_from_anomaly_score below, not a new integration mechanism.
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
from driveflow.agents.explainer import AnomalyExplainer
from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.monitoring.agents.agent_gateway import Alert, GatewayAgent
from driveflow.monitoring.agents.agent_server import ServerAgent
from driveflow.monitoring.rules.schema import load_all_rulesets
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM

#: monitoring/rules/*.yaml, relative to this file -- same convention
#: tests/test_monitoring_rules_schema.py's own RULES_DIR uses (no shared constant exported by
#: monitoring.rules.schema itself to import instead).
_RULES_DIR = Path(__file__).resolve().parents[1] / "monitoring" / "rules"

_DOMAIN_LABELS = {"dc_motor": "Fase A -- DC motor diagnosis", "vsc_dpc": "Fase B -- VSC / DPC"}
_AGENT_CLASS_BY_DOMAIN = {"dc_motor": DCMotorAnomalyDetector, "vsc_dpc": VSCDPCAnomalyDetector}
_ANOMALY_ALERT_THRESHOLD = 0.7


def _alert_from_anomaly_score(rule_name: str, score: float, timestamp: float) -> Alert:
    """A simulation-based agent's (score, diagnosis) has no notion of hysteresis/debounce the way
    a GatewayAgent rule does -- one detect_anomaly() call, one Alert, always both "fired" and
    immediately "resolved" in the same monitoring pass unless score clears the threshold, since
    there's no persistent state across telemetry windows here (that would need
    SimulationBasedAgent itself to track a rolling window, out of scope for Phase 5)."""
    if score >= _ANOMALY_ALERT_THRESHOLD:
        severity = "high" if score >= 0.85 else "medium"
        return Alert(rule_name=rule_name, severity=severity, action="alert", timestamp=timestamp, resolved=False)
    return Alert(rule_name=rule_name, severity="low", action="alert", timestamp=timestamp, resolved=True)


def _get_server_agent() -> ServerAgent:
    """One ServerAgent per Streamlit SESSION, in st.session_state -- NOT st.cache_resource, which
    would be shared across every session/user connected to this server process (found by testing
    with two independent AppTest sessions in the same pytest run: a naive cache_resource version
    leaked one session's alert history into the other's "empty" history). History accumulates
    across every "Run monitoring" click in this one session, which is the point of an Alert
    Timeline that's actually per-user."""
    if "lm_server_agent" not in st.session_state:
        st.session_state["lm_server_agent"] = ServerAgent()
    return st.session_state["lm_server_agent"]


#: Rulesets themselves are read-only and identical for every session, so caching them globally
#: (unlike the stateful agents above) is correct, not the same bug.
@st.cache_resource
def _load_rulesets() -> dict:
    return load_all_rulesets(_RULES_DIR)


def _get_gateway_agent(domain: str) -> GatewayAgent | None:
    """One GatewayAgent per (session, domain) -- same session_state reasoning as
    _get_server_agent: its hysteresis/debounce state is meaningless if shared across sessions."""
    rulesets = _load_rulesets()
    if domain not in rulesets:
        return None
    key = f"lm_gateway_agent_{domain}"
    if key not in st.session_state:
        st.session_state[key] = GatewayAgent(rulesets[domain])
    return st.session_state[key]


def _slider_with_custom(container, label, min_value, max_value, value, step=None, format=None, help=None, key=None):
    """Same pattern as dashboard.py's own helper of the same name (duplicated, not imported --
    this module must not depend on driveflow.viz.dashboard): a slider for the common range, plus
    a "Custom value" checkbox for an unbounded number_input. Needed here specifically so the
    load-resistance rule (fires for R in [1, 3]Ω) is actually reachable -- the plain slider's own
    floor is MIN_STABLE_LOAD_RESISTANCE_OHM (~3.37Ω), same as Fase B's sidebar, which sits ABOVE
    the rule's own range; without this there would be no way to ever see it fire from this tab."""
    base_key = key or label
    slider_val = container.slider(label, min_value, max_value, value, step=step, format=format, help=help, key=f"{base_key}_slider")
    use_custom = container.checkbox("Custom value", key=f"{base_key}_custom", help=f'Type a "{label}" value outside {min_value}-{max_value} above.')
    if not use_custom:
        return slider_val
    custom_val = container.number_input(f"{label} (custom)", value=float(slider_val), step=step or 0.1, format=format, key=f"{base_key}_custom_input")
    return custom_val


def _render_sample_controls(domain: str) -> dict:
    if domain == "dc_motor":
        fault_label = st.sidebar.selectbox("Fault type", ["healthy", "outer_race", "inner_race"], key="lm_fault_type")
        fault_type = None if fault_label == "healthy" else fault_label
        kwargs = {"fault_type": fault_type, "duration_s": 0.15, "seed": 0}
        if fault_type is not None:
            kwargs["electrical_severity"] = st.sidebar.slider("Electrical severity (Nm)", 0.0, 20.0, 8.0, key="lm_elec_severity")
            kwargs["mechanical_severity"] = st.sidebar.slider("Mechanical severity", 0.0, 0.2, 0.05, format="%.3f", key="lm_mech_severity")
        return kwargs
    load_resistance_ohm = _slider_with_custom(
        st.sidebar, "Load resistance R (Ω)", MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, float(_VSC_R_OHM), format="%.4f", key="lm_load_r",
        help="The rule-based agent watches this directly -- use 'Custom value' to drag it into [1, 3]Ω (below the slider's own floor) and see the rule fire.",
    )
    return {"controller_type": "DPC", "plant_config_id": "vsc_dpc_v1", "load_resistance_ohm": load_resistance_ohm, "duration_s": 0.05, "seed": 0}


def _render_fase_lm():
    st.sidebar.markdown(
        '<div class="df-sidebar-title">Live Monitoring</div>'
        '<div class="df-sidebar-hint">Simulation-based + rule-based agents, aggregated by one ServerAgent</div>',
        unsafe_allow_html=True,
    )
    domain = st.sidebar.selectbox("Domain", list(_DOMAIN_LABELS), format_func=lambda d: _DOMAIN_LABELS[d], key="lm_domain")
    scenario_kwargs = _render_sample_controls(domain)

    server_agent = _get_server_agent()
    gateway_agent = _get_gateway_agent(domain)

    if st.sidebar.button("Run monitoring", type="primary", key="lm_run_button"):
        scenario_id = "lm_dc" if domain == "dc_motor" else "lm_vsc"
        records = run_scenario(Scenario(scenario_id=scenario_id, **scenario_kwargs))
        df = pd.DataFrame.from_records(records)

        agent = _AGENT_CLASS_BY_DOMAIN[domain]()
        telemetry = {feat: df[feat].to_numpy(dtype=float) for feat in agent.monitored_features if feat in df.columns}
        score, diagnosis = agent.detect_anomaly(telemetry)
        hypothesis_sim = agent.cache.get((diagnosis, round(len(next(iter(telemetry.values()))) * 1e-4, 6)))
        explanation = AnomalyExplainer(score, diagnosis).explain(telemetry, hypothesis_sim) if hypothesis_sim else None

        now = time.time()
        server_agent.record_alert(domain, _alert_from_anomaly_score("simulation_based_anomaly", score, now))

        rule_events = []
        if gateway_agent is not None and domain == "vsc_dpc":
            rule_events = gateway_agent.step({"load_resistance_ohm": scenario_kwargs["load_resistance_ohm"]}, timestamp=now)
            for event in rule_events:
                server_agent.record_alert(domain, event)

        st.session_state["lm_result"] = {
            "domain": domain, "score": score, "diagnosis": diagnosis, "explanation": explanation,
            "rule_events": rule_events, "hypotheses": list(agent.HYPOTHESES),
        }

    st.markdown(f"##### Agent Consensus -- {_DOMAIN_LABELS[domain]}")
    status = server_agent.domain_status(domain)
    (st.success if status == "ok" else st.error)(f"Status: {status.upper()}")
    if gateway_agent is None:
        st.caption("No rule-based agent for this domain yet (monitoring/rules/ has no YAML for it) -- consensus here is simulation-based only.")

    result = st.session_state.get("lm_result")
    if result is not None and result["domain"] == domain:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Simulation-based detector")
            st.metric("Anomaly score", f"{result['score']:.2f}", help=f"Closest hypothesis: {result['diagnosis']} (of {result['hypotheses']})")
            if result["explanation"]:
                st.caption(result["explanation"]["summary"])
                st.bar_chart(pd.Series(result["explanation"]["feature_importance"], name="importance"))
        with col2:
            st.markdown("##### Rule-based agent")
            if result["rule_events"]:
                for event in result["rule_events"]:
                    (st.error if not event.resolved else st.success)(f"{event.rule_name}: {'fired' if not event.resolved else 'cleared'} ({event.severity})")
            else:
                st.caption(
                    "No rule fired this run -- if the condition IS out of range, this is expected the first "
                    "time: the rule needs it to hold continuously for its own hysteresis_seconds before firing "
                    "(2s for the vsc_dpc divergence rule). Click 'Run monitoring' again a couple seconds later "
                    "with the same value to see it actually fire."
                )

    st.markdown("##### Alert history (this session)")
    history = server_agent.alert_history(domain)
    if history:
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
    else:
        st.caption("Nothing recorded yet -- click 'Run monitoring' in the sidebar.")
