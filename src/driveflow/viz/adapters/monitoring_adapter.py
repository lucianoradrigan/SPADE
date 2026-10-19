"""One SIMULATED monitoring pass (guide Sec. 3.4 Monitoring / Q26).

"Run monitoring" calls run_scenario() to synthesise telemetry -- there is no device, stream,
heartbeat or acquisition clock behind it, so everything produced here is labelled SIMULATED, never
LIVE / CONNECTED. Three agents can contribute (Sec. 1.2's Agent Consensus):
- simulation-based anomaly detector (driveflow.agents) -- always available;
- rule-based GatewayAgent -- only where monitoring/rules/<domain>.yaml exists (today: vsc_dpc);
- classifier confidence drift -- only where a PC-tier classifier is promoted (today: dc_motor).
Each is reported as AVAILABLE / NOT_CONFIGURED, and an agent is only shown as having produced a
signal if it actually emitted an event in this pass.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
from driveflow.agents.explainer import AnomalyExplainer
from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector
from driveflow.datagen import Scenario, run_scenario
from driveflow.monitoring.agents.agent_gateway import Alert, GatewayAgent
from driveflow.monitoring.agents.agent_server import ServerAgent
from driveflow.monitoring.rules.schema import load_all_rulesets
from driveflow.viz.adapters.ai_adapter import classify_last_window

RULES_DIR = Path(__file__).resolve().parents[2] / "monitoring" / "rules"
AGENT_CLASS_BY_DOMAIN = {"dc_motor": DCMotorAnomalyDetector, "vsc_dpc": VSCDPCAnomalyDetector}
#: Pre-existing heuristic thresholds of the legacy LM tab (score in [0, 1] from the simulation-
#: based detector). Kept unchanged -- they are NOT calibrated alarm limits (AUDIT_ANSWERS Q25).
ANOMALY_ALERT_THRESHOLD = 0.7
ANOMALY_HIGH_THRESHOLD = 0.85


def alert_from_anomaly_score(rule_name: str, score: float, timestamp: float) -> Alert:
    if score >= ANOMALY_ALERT_THRESHOLD:
        severity = "high" if score >= ANOMALY_HIGH_THRESHOLD else "medium"
        return Alert(rule_name=rule_name, severity=severity, action="alert", timestamp=timestamp, resolved=False)
    return Alert(rule_name=rule_name, severity="low", action="alert", timestamp=timestamp, resolved=True)


@st.cache_resource(show_spinner=False)
def load_rulesets() -> dict:
    """Rule sets are read-only and identical for every session -- safe to cache globally."""
    return load_all_rulesets(RULES_DIR)


def gateway_agent(domain: str) -> GatewayAgent | None:
    rulesets = load_rulesets()
    if domain not in rulesets:
        return None
    key = f"lm_gateway_agent_{domain}"
    if key not in st.session_state:
        st.session_state[key] = GatewayAgent(rulesets[domain])
    return st.session_state[key]


def record_classifier_confidence_if_available(domain: str, df: pd.DataFrame, server_agent: ServerAgent) -> dict | None:
    """Feeds the promoted classifier's confidence on the tail of this run into the drift monitor.
    Returns the classifier output (AI suggestion) or None if not available for this domain."""
    out = classify_last_window(domain, df)
    if out is not None:
        server_agent.record_classifier_confidence(domain, out["confidence"])
    return out


def step_gateway_over_run(gateway: GatewayAgent, domain: str, df: pd.DataFrame, telemetry: dict) -> list:
    """Evaluates the rules on every sample of this run, on TELEMETRY time (AI_TECHNICAL_AUDIT
    AI-N02): the run's own timestamps, offset by the simulated time of this domain's earlier passes
    so the clock keeps increasing across clicks. Hysteresis is therefore measured on the process,
    not on how long the operator waits between clicks. `telemetry` holds the fields that are
    constant during the run (e.g. the configured load resistance)."""
    clock_key = f"lm_gateway_clock_{domain}"
    offset = float(st.session_state.get(clock_key, 0.0))
    t = df["timestamp_s"].to_numpy(dtype=float) if "timestamp_s" in df.columns else np.arange(len(df)) * 1e-4
    events = []
    for ts in t:
        events.extend(gateway.step(telemetry, timestamp=offset + float(ts)))
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    st.session_state[clock_key] = offset + (float(t[-1]) + dt if len(t) else 0.0)
    return events


def run_monitoring_pass(domain: str, scenario_kwargs: dict, server_agent: ServerAgent) -> dict:
    scenario_id = "lm_dc" if domain == "dc_motor" else "lm_vsc"
    records = run_scenario(Scenario(scenario_id=scenario_id, **scenario_kwargs))
    df = pd.DataFrame.from_records(records)

    agent = AGENT_CLASS_BY_DOMAIN[domain]()
    telemetry = {feat: df[feat].to_numpy(dtype=float) for feat in agent.monitored_features if feat in df.columns}
    # The setpoint is known here, so the hypotheses use it instead of an estimate (AI-N09).
    operating_point = {"omega_ref_rad_s": float(scenario_kwargs.get("omega_ref_rad_s", Scenario.omega_ref_rad_s))} if domain == "dc_motor" else {}
    score, diagnosis = agent.detect_anomaly(telemetry, operating_point=operating_point)
    hypothesis_sim = agent.hypothesis_simulation(diagnosis, len(next(iter(telemetry.values()))))
    explanation = AnomalyExplainer(score, diagnosis).explain(telemetry, hypothesis_sim) if hypothesis_sim else None

    now = time.time()
    server_agent.record_evaluation(domain, now, source_mode="simulated")
    sim_alert = alert_from_anomaly_score("simulation_based_anomaly", score, now)
    server_agent.record_alert(domain, sim_alert, source_id="simulation_agent")

    gateway = gateway_agent(domain)
    rule_events = []
    if gateway is not None and domain == "vsc_dpc":
        rule_events = step_gateway_over_run(gateway, domain, df, {"load_resistance_ohm": scenario_kwargs["load_resistance_ohm"]})
        for event in rule_events:
            # The ServerAgent tracks staleness on the wall clock; the telemetry time stays in
            # the gateway's own state.
            server_agent.record_alert(domain, dataclasses.replace(event, timestamp=now), source_id="gateway")

    classifier = record_classifier_confidence_if_available(domain, df, server_agent)
    tail = df.iloc[int(len(df) * 0.8):]
    rpm = float(tail["rpm"].mean()) if domain == "dc_motor" and "rpm" in df and df["rpm"].notna().any() else None
    agents = {
        "Simulation-based detector": {"availability": "AVAILABLE", "emitted": not sim_alert.resolved, "detail": f"score {score:.2f}, closest hypothesis {diagnosis}"},
        "Rule-based gateway": {"availability": "AVAILABLE" if gateway is not None else "NOT_CONFIGURED", "emitted": bool(rule_events), "detail": (f"{len(rule_events)} event(s)" if gateway is not None else "no monitoring/rules YAML for this domain")},
        "Classifier confidence drift": {"availability": "AVAILABLE" if classifier is not None else "NOT_CONFIGURED", "emitted": False, "detail": (f"confidence {classifier['confidence']:.1%} ({classifier['model_id']})" if classifier else "no promoted PC-tier classifier for this domain")},
    }
    return {
        "domain": domain, "timestamp": now, "score": float(score), "diagnosis": diagnosis, "explanation": explanation,
        "rule_events": rule_events, "hypotheses": list(agent.HYPOTHESES), "classifier": classifier,
        "confidence": classifier["confidence"] if classifier else None, "rpm": rpm, "agents": agents,
        "n_rows": int(len(df)), "scenario": dict(scenario_kwargs),
    }
