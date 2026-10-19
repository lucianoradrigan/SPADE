"""Monitoring page (legacy LM tab; guide Sec. 3.4, figures/gui/concept_v2/04_live_monitoring.png).

Labelled SIMULATED MONITORING throughout: "Run monitoring" synthesises telemetry with
run_scenario() -- there is no device, stream or heartbeat. LIVE / CONNECTED is reserved for a real
telemetry source, which this repository does not have (AUDIT_ANSWERS Q26/Q30).

Domain status comes from ServerAgent.domain_health(), i.e. from ALL active alerts (AUDIT AI-08):
UNKNOWN before the first pass, never HEALTHY by default. The motor illustration is driven by the
same derivation (driveflow.monitoring.motor_status)."""

import datetime as dt

import pandas as pd
import streamlit as st

from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.monitoring.motor_status import derive_motor_view
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM
from driveflow.viz.adapters.monitoring_adapter import gateway_agent, run_monitoring_pass
from driveflow.viz.adapters.session import record_run, server_agent
from driveflow.viz.components.alerts_table import alert_log_frame, alerts_table
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.metric_cards import check_list, kpi_row
from driveflow.viz.components.motor_status import render_motor_status
from driveflow.viz.components.page_header import badge_html, page_header, source_badge_html, status_badge_html

DOMAIN_LABELS = {"dc_motor": "DC motor (dc_motor)", "vsc_dpc": "VSC / DPC (vsc_dpc)"}
_TONE = {"HEALTHY": "healthy", "WARNING": "warning", "CRITICAL": "critical", "OFFLINE": "unknown", "UNKNOWN": "unknown"}


def _controls(domain: str) -> dict:
    if domain == "dc_motor":
        label = st.selectbox("Fault condition", ["healthy", "outer_race", "inner_race"], key="lm_fault_type")
        fault = None if label == "healthy" else label
        kwargs = {"fault_type": fault, "duration_s": 0.6, "seed": 0}
        if fault is not None:
            kwargs["electrical_severity"] = st.slider("Electrical severity (N·m)", 0.0, 20.0, 8.0, key="lm_elec_severity")
            kwargs["mechanical_severity"] = st.slider("Mechanical severity", 0.0, 0.2, 0.05, format="%.3f", key="lm_mech_severity")
        return kwargs
    r = slider_with_custom(st, "Load resistance R (Ω)", MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, float(_VSC_R_OHM), format="%.4f", key="lm_load_r",
                           help="The rule-based agent watches R directly; use 'Custom value' to enter R < R* ≈ 3.37 Ω (below the slider floor, the open-loop-unstable region) and see the divergence rule fire: it is evaluated on every sample of the run and fires once the condition has held for its 10 ms hysteresis.")
    return {"controller_type": "DPC", "plant_config_id": "vsc_dpc_v1", "load_resistance_ohm": r, "duration_s": 0.05, "seed": 0}


def _ts(t) -> str:
    return "—" if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def render() -> None:
    page_header(
        "Operations / Monitoring",
        "Agent consensus & alert analysis",
        "Status derived from all active alerts. Each agent's contribution stays visible and auditable.",
        [source_badge_html("simulated", "Telemetry is synthesised by run_scenario(); no device is connected."), badge_html("SIMULATED MONITORING", "primary", "◆")],
    )
    sa = server_agent()
    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl:
        domain = st.selectbox("Domain", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="lm_domain")
        with st.container(border=True):
            st.markdown("**Simulated telemetry**")
            kwargs = _controls(domain)
            clicked = st.button("Run monitoring", type="primary", key="lm_run_button", width="stretch", icon=":material/play_arrow:")
        st.caption("Each click is one monitoring pass: one physics run, the simulation-based detector, the rule agent (if configured) and the classifier-confidence monitor (if a classifier is promoted).")
    if clicked:
        with view, st.spinner("Running the agents (several physics simulations + a model evaluation, ~15–20 s)…"):
            result = run_monitoring_pass(domain, kwargs, sa)
        st.session_state["lm_result"] = result
        record_run("monitoring", {"domain": domain, **kwargs}, {"health": sa.domain_health(domain), "score": round(result["score"], 3)})

    health = sa.domain_health(domain)
    active = sa.active_alerts(domain)
    result = st.session_state.get("lm_result")
    result = result if result and result["domain"] == domain else None
    with view:
        conf = result["confidence"] if result else None
        drift = sa.check_confidence_drift(domain)
        kpi_row([
            {"label": "Overall domain status", "value": health, "sub": "max severity of active alerts" if active else ("evaluated, no active alert" if health == "HEALTHY" else "not evaluated yet"), "tone": _TONE[health]},
            {"label": "Active alerts", "value": str(len(active)), "sub": ", ".join(a.rule_name for a in active) or "none", "tone": "critical" if active else "unknown"},
            {"label": "Classifier confidence", "value": "—" if conf is None else f"{conf:.1%}", "sub": (("temperature-calibrated" if (result.get("classifier") or {}).get("calibrated") else "uncalibrated softmax") + "; not a health probability") if conf is not None else "no promoted classifier for this domain", "tone": "training"},
            {"label": "Confidence drift", "value": "—" if drift is None else f"{drift.drift:+.3f}", "sub": "needs > 20 passes" if drift is None else ("suggests retraining" if drift.suggest_retraining else "no significant drift"), "tone": "warning" if drift and drift.suggest_retraining else "unknown"},
        ])
        st.caption(f"Last evaluation: {_ts(sa.last_report(domain)[0])} · source: SIMULATED")
        left, right = st.columns([1.6, 1], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown("##### Agent consensus")
                if result is None:
                    st.caption("No monitoring pass for this domain yet — click Run monitoring.")
                else:
                    items = []
                    for name, a in result["agents"].items():
                        status = "not_applicable" if a["availability"] == "NOT_CONFIGURED" else ("warning" if a["emitted"] else "pass")
                        items.append((f"{name} · {a['availability']}", status, a["detail"] + (" — emitted an event this pass" if a["emitted"] else "")))
                    check_list(items)
                    st.caption("Consensus rule: the domain takes the HIGHEST severity among all active alerts from any agent. Agents do not vote; a disagreement (e.g. rule clear, detector alerting) is shown, not averaged away. Classifier confidence never raises an alert by itself.")
            if result is not None:
                with st.container(border=True):
                    st.markdown("##### Simulation-based detector")
                    st.metric("Anomaly score", f"{result['score']:.2f}", help=f"Closest hypothesis: {result['diagnosis']} (of {result['hypotheses']}). Alert at ≥ 0.70 (medium), ≥ 0.85 (high) — pre-existing heuristic thresholds, not calibrated alarm limits.")
                    if result["explanation"]:
                        st.caption(result["explanation"]["summary"])
                        st.bar_chart(pd.Series(result["explanation"]["feature_importance"], name="importance"))
                with st.container(border=True):
                    st.markdown("##### Rule-based agent")
                    if gateway_agent(domain) is None:
                        st.caption("No rule-based agent for this domain (monitoring/rules/ has no YAML for it) — consensus here is simulation-based only.")
                    elif result["rule_events"]:
                        for e in result["rule_events"]:
                            (st.error if not e.resolved else st.success)(f"{e.rule_name}: {'fired' if not e.resolved else 'cleared'} ({e.severity})")
                    else:
                        st.caption("No rule transition this pass. Rules are evaluated on the run's own timeline: a condition must hold for its hysteresis_seconds of simulated time (10 ms for the vsc_dpc divergence rule) before firing, and an active alert resolves after the condition has been false for its clear_hysteresis_seconds.")
                with st.container(border=True):
                    st.markdown("##### Classifier confidence drift")
                    if result["confidence"] is None:
                        st.caption("No promoted PC-tier classifier for this domain (or not enough rows for one window).")
                    else:
                        st.metric("This run's confidence", f"{result['confidence']:.1%}")
                        if drift is None:
                            st.caption("Not enough history for a drift report yet (needs more than 20 passes).")
                        else:
                            st.caption(f"Baseline {drift.baseline_mean:.1%} vs. recent {drift.recent_mean:.1%}")
        with right:
            if domain == "dc_motor":
                c = (result or {}).get("classifier") or {}
                evaluated = sa.last_report(domain)[0] is not None
                view_model = derive_motor_view(sa if evaluated else None, domain, motor_id="dc_motor/sim", source_mode="simulated" if evaluated else "unavailable", rpm=(result or {}).get("rpm"), timestamp_utc=_ts(sa.last_report(domain)[0]) if evaluated else None, prediction=c.get("predicted"), confidence=c.get("confidence"), model_id=c.get("model_id"))
                render_motor_status(view_model)
                with st.expander("Motor details"):
                    st.markdown(f"- **Asset:** simulated DcPermanentlyExcitedMotor (`dc_perm_ex_v1`)\n- **Signals behind the status:** simulation-based detector on current_r / rpm / torque_nm / acc_* (synthetic); no temperature sensor exists\n- **Status:** {view_model.status.upper()} — {view_model.status_reason}\n- **Alarms:** {', '.join(view_model.active_alerts) or 'none'}")
            else:
                with st.container(border=True):
                    st.html('<div class="spade-root spade-badges">' + status_badge_html(health.lower()) + "</div>")
                    st.caption("No rotating machine in this domain (Voltage Source Converter) — no motor illustration.")
        st.markdown("##### Active alerts")
        if not active:
            st.caption("No active alert for this domain.")
        for a in active:
            c1, c2, c3 = st.columns([4, 2, 1])
            c1.markdown(f"**{a.rule_name}** · {a.severity} → {a.level} · source `{a.source_id}`")
            c2.caption(f"first {_ts(a.first_seen)} · last {_ts(a.last_seen)}" + (f" · acknowledged by {a.acknowledged_by}" if a.acknowledged_by else ""))
            if c3.button("Acknowledge", key=f"lm_ack_{a.rule_name}_{a.source_id}", disabled=a.acknowledged_by is not None, help="Marks the alert as seen. It stays ACTIVE until its source resolves it."):
                sa.acknowledge(domain, a.rule_name, by="operator", source_id=a.source_id)
                st.rerun()
        st.markdown("##### Alert history (this session)")
        history = sa.alert_history(domain)
        if history:
            alerts_table(alert_log_frame(history))
        else:
            st.caption("Nothing recorded yet — click Run monitoring.")
        st.page_link("pages/alert_history.py", label="Open the full alert history", icon=":material/list_alt:")
