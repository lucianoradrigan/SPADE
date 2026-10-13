"""Overview (guide Sec. 3.4, figures/gui/concept_v2/01_workspace_overview.png).

Everything shown is real: workspace checks come from the filesystem/registry, "recent runs" are
this session's own run log, and the motor thumbnail is derived from this session's monitoring
state (UNKNOWN until a monitoring pass has run). The mockup's illustrative numbers (98.4% data
integrity, 72,480 samples, ...) are deliberately not reproduced."""

import streamlit as st

from driveflow.monitoring.motor_status import derive_motor_view
from driveflow.viz.adapters.session import run_log, server_agent
from driveflow.viz.adapters.workspace import workspace_status
from driveflow.viz.components.metric_cards import check_list, kpi_row
from driveflow.viz.components.motor_status import render_motor_status
from driveflow.viz.components.page_header import badge_html, page_header, source_badge_html
from driveflow.viz.views.system_diagram import render_system_diagram

_WORKFLOW = [
    ("pages/dc_motor.py", "1 · Configure & simulate", "DC motor / PMSM / VSC physics"),
    ("pages/ai_diagnostics.py", "2 · Diagnose", "promoted classifier / forecaster"),
    ("pages/transfer_learning.py", "3 · Adapt", "gated transfer learning"),
    ("pages/model_registry.py", "4 · Validate & deploy", "registry, lineage, edge export"),
    ("pages/monitoring.py", "5 · Monitor", "agent consensus (simulated)"),
]


def render() -> None:
    status = workspace_status()
    n_ok = sum(c["ok"] for c in status["checks"])
    all_ok = n_ok == len(status["checks"])
    page_header(
        "Workspace overview",
        "Electric drives, simulation & AI",
        "One traceable workflow: configure a physical system, simulate it, diagnose it with a promoted model, adapt the model safely, and monitor it.",
        [source_badge_html("simulated", "No hardware ingestion exists in this repository; every signal is simulated."),
         badge_html(f"{n_ok}/{len(status['checks'])} CHECKS PASS", "healthy" if all_ok else "warning", "✓" if all_ok else "▲")],
    )
    kpi_row([
        {"label": "Physical systems", "value": "3", "sub": "DC motor · PMSM (control demo) · VSC", "tone": "primary"},
        {"label": "Promoted models", "value": str(status["n_promoted"]), "sub": "registry entries (domain/tier/block)", "tone": "training"},
        {"label": "Diagnosis domains", "value": "2", "sub": "dc_motor · vsc_dpc (never mixed)", "tone": "primary"},
        {"label": "Workspace checks", "value": f"{n_ok}/{len(status['checks'])}", "sub": "verified on every page load", "tone": "healthy" if all_ok else "warning"},
    ])

    st.markdown("##### Engineering workflow")
    cols = st.columns(len(_WORKFLOW))
    for col, (page, label, sub) in zip(cols, _WORKFLOW):
        with col, st.container(border=True):
            st.page_link(page, label=label)
            st.caption(sub)

    left, mid, right = st.columns([1.1, 1.3, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown("##### Quick launch")
        st.page_link("pages/dc_motor.py", label="DC motor — PI/MPC simulation & fault injection", icon=":material/electric_bolt:")
        st.page_link("pages/pmsm.py", label="PMSM / FOC — MTPA control-law demo (no diagnosis)", icon=":material/radio_button_checked:")
        st.page_link("pages/vsc_dpc.py", label="VSC / DPC — trained network, complex plane", icon=":material/power:")
        st.page_link("pages/ai_diagnostics.py", label="AI Diagnostics — promoted models", icon=":material/psychology:")
        st.page_link("pages/monitoring.py", label="Monitoring — agent consensus", icon=":material/monitor_heart:")
    with mid, st.container(border=True):
        st.markdown("##### Recent runs (this session)")
        log = run_log()
        if not log:
            st.caption("No run yet in this session. Results appear here with their run ID, time and configuration hash.")
        else:
            st.dataframe(
                [{"run_id": r["run_id"], "utc": r["utc"], "config": r["config_hash"], **{k: v for k, v in r["summary"].items() if not isinstance(v, (dict, list))}} for r in reversed(log)],
                hide_index=True, width="stretch", height=230,
            )
    with right, st.container(border=True):
        st.markdown("##### Motor status")
        sa = server_agent()
        last = st.session_state.get("lm_result") or {}
        src = "simulated" if sa.last_report("dc_motor")[0] is not None else "unavailable"
        view = derive_motor_view(sa if src == "simulated" else None, "dc_motor", motor_id="dc_motor/sim", source_mode=src, rpm=last.get("rpm") if last.get("domain") == "dc_motor" else None)
        render_motor_status(view, compact=True)
        st.caption(f"{view.status.upper()} — {view.status_reason}.")
        st.page_link("pages/monitoring.py", label="Open monitoring", icon=":material/arrow_forward:")

    with st.expander("Workspace checks", expanded=not all_ok):
        check_list([(c["name"], "pass" if c["ok"] else "fail", c["detail"]) for c in status["checks"]])

    st.divider()
    st.markdown("##### System diagram")
    render_system_diagram()
    st.caption("Concept renders in the design package are illustrative; this page shows only values computed from this repository.")
