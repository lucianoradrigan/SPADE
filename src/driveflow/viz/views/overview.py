"""Overview (guide Sec. 3.4, figures/gui/concept_v2/01_workspace_overview.png).

Everything shown is real: workspace checks come from the filesystem/registry, "recent runs" are
this session's own run log, and the motor thumbnail is derived from this session's monitoring
state (UNKNOWN until a monitoring pass has run). The mockup's illustrative numbers (98.4% data
integrity, 72,480 samples, ...) are deliberately not reproduced."""

from html import escape

import streamlit as st

from driveflow.monitoring.motor_status import derive_motor_view
from driveflow.viz.app import PAGES
from driveflow.viz.adapters.session import run_log, server_agent
from driveflow.viz.adapters.workspace import workspace_status
from driveflow.viz.components.board_card import board_cards
from driveflow.viz.components.citation import render_citation
from driveflow.viz.components.metric_cards import check_list, kpi_row
from driveflow.viz.components.motor_status import render_motor_status
from driveflow.viz.components.page_header import badge_html, page_header, source_badge_html
from driveflow.viz.theme.tokens import GROUP_ACCENT, SERIES
from driveflow.viz.views.system_diagram import render_system_diagram

_WORKFLOW = [
    ("pages/dc_motor.py", "1 · Simulate", "DC motor / PMSM / VSC physics"),
    ("pages/ai_diagnostics.py", "2 · Diagnose", "promoted classifier / forecaster"),
    ("pages/transfer_learning.py", "3 · Adapt", "gated transfer learning"),
    ("pages/model_registry.py", "4 · Validate", "registry, lineage, promotion"),
    ("pages/edge_deployment.py", "5 · Deploy", "ESP32, Raspberry Pi, C2000"),
    ("pages/monitoring.py", "6 · Monitor", "agent consensus (simulated)"),
]


_EDGE_NOTES = {
    "esp32": ".tflite int8 for a sensor node. Exported, not run on a board.",
    "rpi5": ".tflite float16 for a gateway, in Python or C++. Exported, not run on a board.",
    "c2000": "Generated C next to the control code. Measured on a real F28379D, with simulated windows.",
}

#: The platform's video guides, in viewing order: (YouTube id, title, what it covers, accent).
VIDEO_GUIDES = [
    ("mP82vIYR_RQ", "Platform overview", "An engineering workbench for simulation, control & AI", SERIES["reference"]),
    ("t5ri_OAHDnQ", "Simulation", "Configure parameters, run experiments & analyze results", GROUP_ACCENT["Simulation"]),
    ("6XH58EuwGEs", "Design & Control", "Controller tuning, optimization & reinforcement learning", GROUP_ACCENT["Design & Control"]),
    ("S4xAtLYSiMQ", "Intelligence", "AI diagnostics, parameter sensitivity & model registry", GROUP_ACCENT["Intelligence"]),
]

#: Tool groups shown at the bottom of the overview, in the sidebar's order:
#: (section, what the section is for, [(page, tool, what the tool does)]).
_GROUPS = [
    ("Simulation",
     "Physics-based models of the drives and converters. Every signal used elsewhere in SPADE starts here: "
     "configure a plant, run it under a controller, inject conditions and inspect the waveforms.",
     [("pages/dc_motor.py", "DC Motor", "PI/MPC speed control of a DC motor with bearing-fault injection, electrical and vibration signals, and the AI suggestion next to the injected condition."),
      ("pages/pmsm.py", "PMSM / FOC", "Field-oriented control of a salient PMSM in the dq frame, comparing MTPA with i_d = 0 at equal current. Control demo only, no diagnosis."),
      ("pages/vsc_dpc.py", "VSC / DPC", "Direct power control of a voltage source converter by a trained network tracking a rotating reference, shown in the complex plane."),
      ("pages/dcdc_converters.py", "DC-DC Converters", "Averaged synchronous buck under an average-current-mode PI cascade, plus the DAB modulation selector and phase-shift diagram."),
      ("pages/plecs_plant_studio.py", "PLECS Plant Studio", "Builds new plants from circuits: PLECS RPC sweeps for datasets and verified FMUs (OpenModelica or PLECS) as Gymnasium plants.")]),
    ("Design & Control",
     "Controller design on top of the simulated plants. Tools here propose candidate controllers and benchmark them "
     "against the classical baseline; nothing replaces the default controller automatically.",
     [("pages/tuning_studio.py", "Tuning Studio", "Optimizes controller parameters on simulated scenarios and compares the candidate with the baseline on a separate evaluation grid."),
      ("pages/rl_lab.py", "RL Lab", "Defines environment and reward, trains DQN or DDPG agents as jobs and benchmarks the policies against the classical controllers.")]),
    ("Intelligence",
     "Data-driven models trained on simulated data: fault classification and forecasting, safe adaptation to new "
     "conditions, and the registry that records which model is promoted and why.",
     [("pages/ai_diagnostics.py", "AI Diagnostics", "Runs the promoted classifier/forecaster of a domain on a simulated sample or an uploaded file, or trains a new classifier on a custom grid."),
      ("pages/edge_deployment.py", "Edge Deployment", "Takes a model to an ESP32, a Raspberry Pi 5 or a TI C2000, with export, ready-to-copy inference code and a tutorial per board."),
      ("pages/transfer_learning.py", "Transfer Learning", "Guided fine-tuning of a promoted model with dataset checks, an independent test set and a fail-closed promotion gate with human approval."),
      ("pages/model_registry.py", "Model Registry", "Every trained run on disk with its promotion status, lineage, validation gate and edge-export evidence.")]),
    ("Operations",
     "Supervision of the running system. Monitoring agents turn telemetry into alerts and a consensus status; "
     "in this repository the telemetry is simulated.",
     [("pages/monitoring.py", "Monitoring", "Status derived from all active alerts, with each agent's contribution to the consensus visible and auditable."),
      ("pages/alert_history.py", "Alert History", "Every alert event, resolution and acknowledgement recorded in this session, in arrival order.")]),
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

    with st.container(border=True):
        st.markdown("##### Deploy a model to edge hardware")
        st.caption("From a trained model to code running on the board, with ready-to-copy inference code and a tutorial for each one.")
        board_cards(_EDGE_NOTES)
        st.page_link("pages/edge_deployment.py", label="Open Edge Deployment", icon=":material/developer_board:")

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

    st.divider()
    _render_tool_groups()

    st.divider()
    _render_video_guides()

    st.divider()
    st.markdown("##### System diagram")
    render_system_diagram()
    st.caption("Concept renders in the design package are illustrative; this page shows only values computed from this repository.")

    with st.expander("Workspace checks", expanded=not all_ok):
        check_list([(c["name"], "pass" if c["ok"] else "fail", c["detail"]) for c in status["checks"]])

    st.divider()
    render_citation()


def _render_video_guides() -> None:
    st.markdown("##### :material/play_circle: Video guides")
    st.caption("New to SPADE? Start with the platform overview, then watch the guide for the section you need.")
    for col, (video, title, covers, tone) in zip(st.columns(len(VIDEO_GUIDES), gap="small"), VIDEO_GUIDES):
        with col:
            st.html(f'<div class="spade-root spade-video-label" style="--tone:{tone}"><b>{escape(title)}</b><span>{escape(covers)}</span></div>')
            st.video(f"https://www.youtube.com/watch?v={video}")


def _render_tool_groups() -> None:
    """The four working sections of the sidebar: a flow strip, then one card per group with what the
    group is for and a link + one-line description per tool. Colours come from GROUP_ACCENT."""
    icons = {path: icon for items in PAGES.values() for path, _, icon in items}
    st.markdown("##### Tool groups")
    st.caption("SPADE's tools are organised in four groups that follow the engineering workflow, from physics to supervision.")
    nodes = []
    for i, (section, _, tools) in enumerate(_GROUPS, 1):
        if i > 1:
            nodes.append('<div class="spade-groupflow-arrow" aria-hidden="true">→</div>')
        n = len(tools)
        nodes.append(f'<div class="spade-groupflow-node" style="--tone:{GROUP_ACCENT[section]}"><div class="spade-groupflow-num">{i:02d}</div>'
                     f'<div><b>{escape(section)}</b><span>{n} tool{"s" if n != 1 else ""}</span></div></div>')
    st.html(f'<div class="spade-root spade-groupflow">{"".join(nodes)}</div>')

    for row in (list(enumerate(_GROUPS[:2], 1)), list(enumerate(_GROUPS[2:], 3))):
        for col, (i, (section, about, tools)) in zip(st.columns(2, gap="medium"), row):
            tone = GROUP_ACCENT[section]
            with col, st.container(border=True):
                n = len(tools)
                st.html(f'<div class="spade-root spade-group" style="--tone:{tone}"><div class="spade-group-top">'
                        f'<span class="spade-group-num">{i:02d}</span><span class="spade-group-title">{escape(section)}</span>'
                        f'<span class="spade-group-count">{n} TOOL{"S" if n != 1 else ""}</span></div>'
                        f'<div class="spade-group-about">{escape(about)}</div></div>')
                for page, tool, desc in tools:
                    st.page_link(page, label=f"**{tool}**", icon=icons.get(page))
                    st.html(f'<div class="spade-root spade-tool-desc" style="--tone:{tone}">{escape(desc)}</div>')
