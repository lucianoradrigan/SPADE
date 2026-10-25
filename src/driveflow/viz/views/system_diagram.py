"""Platform block diagram (ported from the legacy single-file dashboard's landing page).

Clicking a box writes session_state["diagram_component_detail"]; the matching text below the
chart explains what that component really is and what it does NOT do. The layout/detail key sets
are checked against each other at import time (self-enforcing, same as before)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from driveflow.viz.theme.tokens import DARK

def _diagram_box(fig, cx, cy, w, h, text, color):
    """One labeled rectangle for _render_system_diagram -- a filled shape (the box) plus a
    centered annotation (the text), since Plotly shapes carry no text of their own. Fill is only
    a ~15% tint of `color` (not a solid fill), so the active theme's own text colour reads
    fine on every box regardless of which semantic `color` it uses."""
    fig.add_shape(
        type="rect", x0=cx - w / 2, y0=cy - h / 2, x1=cx + w / 2, y1=cy + h / 2,
        line=dict(color=color, width=1.5), fillcolor=f"{color}26",
    )
    fig.add_annotation(x=cx, y=cy, text=text, showarrow=False, font=dict(size=11), align="center")


#: Grid spacing (data units) for _diagram_click_target's invisible hit-target points -- small
#: enough that no point inside the smallest box (2.7x0.85) is more than ~half a grid cell from
#: its nearest marker.
_CLICK_GRID_STEP = 0.3


def _diagram_click_target(fig, cx, cy, w, h, component_key):
    """Makes the rectangle at (cx, cy, w, h) clickable via Streamlit's plotly on_select, which
    only sees DATA-TRACE points (Plotly's own click/selection events are point-proximity based),
    not `add_shape` rectangles or annotations -- clicking a `fill="toself"` shape's interior isn't
    reliably exposed through that API either. Standard workaround: a dense grid of invisible
    marker points covering the box's area, one scatter trace per box, each point's `customdata`
    carrying `component_key` so the click handler in _render_system_diagram knows which box was
    hit regardless of which grid point ended up nearest the actual click.

    `hoverinfo="none"`, NOT `"skip"` -- found by testing clicks in the actual running app and
    getting nothing back: Plotly's docs distinguish the two ("skip": the trace is excluded from
    hover AND click event handling entirely; "none": events still fire, only the tooltip itself
    is suppressed). "skip" silently ate every click on these invisible points."""
    xs = np.arange(cx - w / 2, cx + w / 2 + 1e-9, _CLICK_GRID_STEP)
    ys = np.arange(cy - h / 2, cy + h / 2 + 1e-9, _CLICK_GRID_STEP)
    grid_x, grid_y = np.meshgrid(xs, ys)
    n = grid_x.size
    fig.add_trace(
        go.Scatter(
            x=grid_x.ravel(), y=grid_y.ravel(), mode="markers",
            # opacity=0 (fully transparent), not just a low value -- some rendering paths skip
            # hit-testing entirely for a fully-transparent element even though Plotly's own docs
            # say opacity shouldn't affect event handling; 0.02 keeps every point visually
            # invisible against the dark canvas while guaranteeing it isn't treated as absent.
            marker=dict(size=14, opacity=0.02), customdata=[component_key] * n,
            hoverinfo="none", showlegend=False, name=component_key,
        )
    )


def _diagram_arrow(fig, x0, y0, x1, y1, dash=False):
    """One connector for _render_system_diagram -- solid for data/artifact flow (an annotation
    arrow, real arrowhead), dashed for the monitoring layer (Sec. 4.3: telemetry -> rule agent ->
    aggregated alert, same -.-> convention the design doc's own Mermaid diagram uses). Dashed
    ones are a `line` shape instead of an annotation arrow -- go.layout.Annotation has no dash
    property on its arrow, only shapes do; direction is unambiguous anyway from the layout
    (monitoring always flows left-to-right / bottom-to-top here, same as the solid arrows)."""
    if dash:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=DARK["unknown"], width=1.3, dash="dot"))
    else:
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1, axref="x", ayref="y", xref="x", yref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.3, arrowcolor=DARK["unknown"],
            opacity=0.9, standoff=4,
        )


#: Per-component detail shown below the diagram (Plotly shapes have no reliable click-to-select
#: in Streamlit -- fill/shape hit-testing isn't part of st.plotly_chart's on_select API, only
#: data-trace points are -- so the boxes carry an invisible marker grid instead). Ordered the same
#: bottom-to-top path the diagram itself reads in: plants, then the two lanes (AI on the left,
#: design & control on the right), then deployment and the dashboard; monitoring last. Every number
#: is the promoted run's own metrics.json (configs/registry.yaml), not an illustrative value.
_DIAGRAM_COMPONENT_DETAILS = {
    "DC motor + PMSM": (
        "Physics simulation: `DcPermanentlyExcitedMotor` under a native PI cascade (speed or torque) or MPC, "
        "with bearing-fault injection on two paths (electrical/MCSA + synthetic vibration). This is domain "
        "`dc_motor`'s data source. The PMSM is a separate FOC/MTPA control-law demo with no diagnosis. See the "
        "DC Motor and PMSM / FOC pages (Simulation section)."
    ),
    "VSC + DPC network": (
        "Physics plus a separately pre-trained Direct Power Control network (ported from "
        "DPC4PowerElectronics, not trained by this AI layer) tracking a rotating voltage reference. This is "
        "domain `vsc_dpc`'s data source -- see the VSC / DPC page."
    ),
    "DC-DC converters": (
        "`sim/buck_system.py`: an averaged synchronous buck (48 V input, 220 µH, 220 µF, 6 Ω load) under a PI "
        "cascade, discretised with an exact zero-order hold at 20 µs. The DAB tab is a modulation selector "
        "only: SPADE has no DAB simulation (it needs PLECS). See the DC-DC Converters page."
    ),
    "PLECS Plant Studio": (
        "External circuits: datasets from PLECS Standalone sweeps over its local XML-RPC interface, run as "
        "jobs (needs a PLECS licence on the same computer). A surrogate fitted to a sweep becomes a Gymnasium "
        "environment, `SPADE/<Profile>Surrogate-v0`, and is compared with the native buck. SPADE's authors have "
        "no PLECS licence, so no PLECS run has been executed here: researchers with PLECS use the PLECS "
        "researcher kit (docs/guides/PLECS_RESEARCHER_KIT.md)."
    ),
    "Common data layer": (
        "`models/common/windowing.py`: turns either domain's raw simulation rows into fixed-size windows "
        "(for the classifier) or `(input_window, horizon)` forecast pairs (for the regressor). Enforces the "
        "domain isolation design doc Sec. 1 requires -- dc_motor and vsc_dpc rows are never combined into "
        "one training set, even though the mechanism (windowing code) is shared."
    ),
    "Gymnasium environments": (
        "`envs/registry.py` registers `SPADE/DcMotor-v0`, `SPADE/Vsc-v0`, `SPADE/Buck-v0`, `SPADE/Pmsm-v0` "
        "(native plants) and `SPADE/BuckSurrogate-v0`. The packaged surrogate is fitted to SPADE's own "
        "averaged buck, not to PLECS: it validates the surrogate mechanics, not fidelity to any PLECS circuit. "
        "Observation, action, reward and limits of each one are on the Environments page."
    ),
    "Classifier (CNN)": (
        "`models/classifiers/builder.py` -- Conv1D+SE blocks, config-driven. `dc_motor` only. Promoted PC tier: "
        "`pc_server_5class_w2048/2026-10-20_run01`, 5 classes (ball, cage, inner_race, normal, outer_race), "
        "test accuracy 90.3 % and macro-F1 0.90 on 600 test windows. No `vsc_dpc` classifier is promoted: "
        "capacitor degradation is synthetic-only. A classifier can be trained from AI Diagnostics → Train a model."
    ),
    "Regressor (LSTM/TCN)": (
        "`models/regressors/builder.py` -- config-driven forecaster. `vsc_dpc` only. Promoted PC tier: "
        "`pc_full/2026-10-19_run01`, stacked LSTM [128, 64], normalized test RMSE 0.073 (persistence "
        "baseline: 1.06). The `dc_motor` forecaster is trained but not promoted."
    ),
    "Tuning Studio": (
        "PSO, Bayesian optimization or NSGA-II over the DC-motor PI speed loop, run as a job. Each candidate "
        "is simulated on the native DC-motor closed-loop bench (`validation/benches.py`, not through a "
        "Gymnasium environment), tuned on one grid and compared with the baseline on a separate evaluation "
        "grid. The result is a candidate: \"Use as default\" is blocked by the approval gate (decision D1)."
    ),
    "RL Lab": (
        "DQN or DDPG trained on SPADE's Gymnasium environments as a job, then benchmarked against the plant's "
        "classical controllers on the same grid -- the benchmark is shown even when RL loses. Policies are "
        "candidates: \"Use as default\" is blocked (decision D4)."
    ),
    "Model registry": (
        "`ai/registry.py` -- one manifest (`configs/registry.yaml`, 6 entries today) mapping `(domain, tier, "
        "block)` to the promoted run folder (weights + the exact config that produced them + metrics). "
        "Promotion (`experiments/promote_run.py`) is a deliberate, separate step from training -- a training "
        "run existing doesn't mean it is the one in use."
    ),
    "Experiment runs & jobs": (
        "`driveflow.jobs` + `driveflow.runs`: one SQLite database in the workspace holds the job queue and every "
        "run (configuration hash, seed, versions, logs, reproducible command). Long jobs -- PLECS sweeps, tuning, "
        "RL training, classifier training -- execute in a separate worker process "
        "(`python -m driveflow.jobs.worker serve`); the GUI only queues, observes and cancels. See the "
        "Experiment Runs page."
    ),
    "Raspberry Pi 5": (
        "**Not trained from scratch.** Distilled from the promoted PC-tier model (the \"teacher\"): `--distill` "
        "blends the true label/value with the teacher's own prediction (`--distill-alpha` 0.5). Smaller "
        "architectures: a ResNet1D-SE classifier with 2 Conv1D blocks instead of 3, and a dilated causal TCN "
        "regressor (7 × 32 filters) instead of the PC tier's LSTM. Exported to TFLite as **float16** (no "
        "calibration data needed). Current numbers: dc_motor classifier 90.2 % (teacher 90.3 %, same 5 classes "
        "and test split); vsc_dpc regressor RMSE 0.064 (teacher 0.073)."
    ),
    "ESP32": (
        "Same distillation as Raspberry Pi 5, two tiers stricter: **(1) no recurrent layers at all** -- the "
        "regressor config schema structurally forbids `recurrent_type: lstm`/`gru` on this tier, so the "
        "regressor is a small causal TCN (`layers: [8, 8]`) and the classifier uses DS-CNN blocks "
        "(depthwise + pointwise Conv1D). **(2) TFLite export is int8** (weights AND activations), calibrated on "
        "windows sampled from the training dataset (simulated), never on random noise. Current numbers: vsc_dpc regressor RMSE 0.18. "
        "The dc_motor classifier is still the earlier **3-class** run (inner_race, normal, outer_race; "
        "90.1 % on its own 3-class test split, not comparable with the 5-class tiers)."
    ),
    "Dashboard — Engineering Workbench": (
        "`viz/app.py`: the multipage Streamlit app (Workspace, Simulation, Design & Control, Intelligence, "
        "Operations, Help). AI Diagnostics runs the promoted **PC-tier** models and labels the output as an AI "
        "suggestion with its model version; Model Registry lists every run with its lineage and edge "
        "artifacts. Edge (.tflite) models are not executed in the dashboard: they are listed and downloadable "
        "only. Every signal is simulated -- there is no hardware ingestion."
    ),
    "ESP32 watchdog": (
        "Not a model, not related to the \"ESP32\" box's classifier/regressor -- this is the "
        "**rule-based monitoring** tier (Sec. 4.3), pure hard-coded thresholds with zero ML: "
        "`monitoring/rules/vsc_dpc.yaml` + a validated schema (`schema.py`), e.g. the R < R* ≈ 3.37 Ω divergence "
        "rule for vsc_dpc. No training, no distillation -- just YAML data plus an AST-checked boolean "
        "condition (never `eval()`'d against arbitrary code)."
    ),
    "GatewayAgent — RPi5": (
        "`monitoring/agents/agent_gateway.py` -- evaluates a domain's ruleset against a telemetry stream "
        "with per-rule hysteresis (must hold N seconds before firing) and debounce (doesn't re-fire every "
        "tick while still true). Can run without a connection to the PC tier."
    ),
    "ServerAgent — PC": (
        "`monitoring/agents/agent_server.py` -- aggregates the Alert events a GatewayAgent already produced "
        "into a per-domain badge status + history, and separately tracks classifier confidence over time to "
        "flag drift and suggest retraining. Does not evaluate telemetry itself. Shown on the Monitoring page "
        "(SIMULATED)."
    ),
}


#: Box palette: one colour per role, reused across lanes (plants, data/environments, learners,
#: registry/jobs, edge tiers, dashboard, monitoring).
_EDGE = "#F472B6"
_UI = "#2563EB"

#: Single source of truth for the diagram's boxes -- (component_key, cx, cy, w, h, color,
#: display_text). component_key must match a key in _DIAGRAM_COMPONENT_DETAILS (checked below and
#: by tests/test_dashboard_landing.py); display_text is the multi-line label shown IN the box, while
#: component_key is the short name the click handler uses. Columns: monitoring at x≈1.45; the AI
#: lane over x 3.2–10.6; the design & control lane over x 10.8–18.2.
_DIAGRAM_LAYOUT = [
    ("DC motor + PMSM", 5.0, 1.0, 3.6, 1.05, DARK["primary"], "DC motor + PMSM<br>physics · fault injection"),
    ("VSC + DPC network", 8.8, 1.0, 3.6, 1.05, DARK["training"], "VSC + DPC network<br>physics + trained network"),
    ("DC-DC converters", 12.6, 1.0, 3.6, 1.05, DARK["primary"], "DC-DC converters<br>averaged buck · DAB selector"),
    ("PLECS Plant Studio", 16.4, 1.0, 3.6, 1.05, DARK["training"], "PLECS Plant Studio<br>RPC sweeps → surrogate"),
    ("Common data layer", 6.9, 2.6, 7.4, 0.95, DARK["text_muted"], "Common data layer — Scenario · windowing · dataset<br>(domains never mixed, design doc Sec. 1)"),
    ("Gymnasium environments", 14.5, 2.6, 7.4, 0.95, DARK["text_muted"], "Gymnasium environments — SPADE/*-v0<br>DcMotor · Vsc · Buck · Pmsm · BuckSurrogate"),
    ("Classifier (CNN)", 5.0, 4.15, 3.6, 0.95, DARK["healthy"], "Classifier (CNN)<br>dc_motor · 5 classes · PC"),
    ("Regressor (LSTM/TCN)", 8.8, 4.15, 3.6, 0.95, DARK["healthy"], "Regressor (LSTM/TCN)<br>vsc_dpc · PC"),
    ("Tuning Studio", 12.6, 4.15, 3.6, 0.95, DARK["healthy"], "Tuning Studio<br>PSO · Bayesian · NSGA-II"),
    ("RL Lab", 16.4, 4.15, 3.6, 0.95, DARK["healthy"], "RL Lab<br>DQN · DDPG vs classical"),
    ("Model registry", 6.9, 5.65, 7.4, 0.95, DARK["warning"], "Model registry — ai/registry.py<br>(domain, tier, block) → promoted artifact"),
    ("Experiment runs & jobs", 14.5, 5.65, 7.4, 0.95, DARK["warning"], "Experiment runs & jobs — worker + run store<br>results stay candidates (approval gate)"),
    ("Raspberry Pi 5", 5.0, 7.1, 3.6, 0.95, _EDGE, "Raspberry Pi 5<br>distilled + TFLite float16"),
    ("ESP32", 8.8, 7.1, 3.6, 0.95, _EDGE, "ESP32<br>distilled + TFLite int8"),
    ("Dashboard — Engineering Workbench", 10.7, 8.55, 15.0, 0.95, _UI, "Dashboard — SPADE Engineering Workbench<br>simulation · design & control · intelligence · operations"),
    ("ESP32 watchdog", 1.45, 1.0, 2.7, 0.85, DARK["unknown"], "ESP32 watchdog<br>hard thresholds, no ML"),
    ("GatewayAgent — RPi5", 1.45, 3.3, 2.7, 0.85, DARK["unknown"], "GatewayAgent — RPi5<br>hysteresis + debounce"),
    ("ServerAgent — PC", 1.45, 5.65, 2.7, 0.85, DARK["unknown"], "ServerAgent — PC<br>alert history + drift"),
]
#: Self-enforcing instead of relying on a separate test to catch drift: every AppTest run of the
#: landing page executes this at import/script-run time, so a mismatch surfaces immediately as
#: `at.exception` rather than silently leaving a diagram box whose click has nowhere to go.
assert {row[0] for row in _DIAGRAM_LAYOUT} == set(_DIAGRAM_COMPONENT_DETAILS), (
    f"diagram/detail key mismatch: {({row[0] for row in _DIAGRAM_LAYOUT} ^ set(_DIAGRAM_COMPONENT_DETAILS))}"
)


def _build_system_diagram_figure() -> go.Figure:
    """Pure figure construction (no `st.*` calls) -- separated from _render_system_diagram so it
    can be unit-tested directly (tests/test_dashboard_landing.py) without a Streamlit script
    context. A block diagram of the whole platform (not just the AI layer): both macro-phases'
    physics/plants plus the DC-DC and PLECS plants, the config-driven IA layer
    (docs/design_ai_layer_transversal.md), the model registry, edge-tier distillation, the design &
    control lane (environments, tuning, RL, jobs) and the monitoring agents, laid out bottom-to-top
    the same way data actually flows. Plotly shapes/annotations/traces, not a new dependency
    (st.graphviz_chart would need the `graphviz` package AND the system `dot` binary neither of
    which this project already depends on; a JS-based Mermaid embed would need loading a script
    from a CDN at runtime, which this offline-capable research tool avoids elsewhere too).

    Each box is clickable (_diagram_click_target's invisible marker grid) -- clicking one selects
    it in the detail panel below (session_state["diagram_component_detail"])."""
    fig = go.Figure()

    for _key, cx, cy, w, h, color, display_text in _DIAGRAM_LAYOUT:
        _diagram_box(fig, cx, cy, w, h, display_text, color)
        _diagram_click_target(fig, cx, cy, w, h, _key)

    # Data/artifact flow (solid). AI lane: plants → common data layer → classifier/regressor →
    # registry → edge tiers and the dashboard. Design & control lane: plants → Gymnasium
    # environments → RL Lab → jobs/runs → dashboard. The Tuning Studio has no incoming arrow on
    # purpose: it simulates the native DC-motor bench directly, not an environment, and a line from
    # the DC-motor box would cut across the data layer (its detail text says so).
    _diagram_arrow(fig, 5.0, 1.525, 6.0, 2.125)
    _diagram_arrow(fig, 8.8, 1.525, 7.8, 2.125)
    _diagram_arrow(fig, 10.2, 1.525, 11.2, 2.125)
    _diagram_arrow(fig, 12.6, 1.525, 12.6, 2.125)
    _diagram_arrow(fig, 16.4, 1.525, 16.4, 2.125)
    _diagram_arrow(fig, 6.0, 3.075, 5.0, 3.675)
    _diagram_arrow(fig, 7.8, 3.075, 8.8, 3.675)
    _diagram_arrow(fig, 16.4, 3.075, 16.4, 3.675)
    _diagram_arrow(fig, 5.0, 4.625, 6.0, 5.175)
    _diagram_arrow(fig, 8.8, 4.625, 7.8, 5.175)
    _diagram_arrow(fig, 12.6, 4.625, 12.6, 5.175)
    _diagram_arrow(fig, 16.4, 4.625, 16.4, 5.175)
    _diagram_arrow(fig, 6.0, 6.125, 5.0, 6.625)
    _diagram_arrow(fig, 7.8, 6.125, 8.8, 6.625)
    _diagram_arrow(fig, 6.9, 6.125, 6.9, 8.075)
    _diagram_arrow(fig, 14.5, 6.125, 14.5, 8.075)

    # Monitoring flow (dashed) -- domain telemetry into the tier-appropriate rule agent, alerts
    # escalating PC-ward, and the aggregated status reaching the dashboard as a badge. Enters the
    # left-hand column from the common data layer (both domains' own telemetry, simplified to one
    # entry point), which sits right next to it.
    _diagram_arrow(fig, 3.2, 2.6, 2.8, 1.0, dash=True)
    _diagram_arrow(fig, 1.45, 1.425, 1.45, 2.875, dash=True)
    _diagram_arrow(fig, 1.45, 3.725, 1.45, 5.225, dash=True)
    _diagram_arrow(fig, 1.45, 6.075, 3.2, 8.55, dash=True)

    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 18.4], fixedrange=True),
        yaxis=dict(visible=False, range=[0, 9.3], scaleanchor="x", scaleratio=1, fixedrange=True),
        showlegend=False,
        # NOT dragmode=False -- fixedrange=True on both axes above already makes pan/zoom
        # impossible (that alone was "no quiero que el gráfico se mueva"), and dragmode=False
        # risks disabling more of Plotly's pointer-event handling than just dragging (the same
        # bundled-source click/select gating referenced below sits near dragmode-related code
        # paths). Leaving dragmode at its default keeps click-to-select unambiguous.
        # Plotly's own default clickmode is "event" -- a plain click (no drag) fires
        # plotly_click but does NOT populate the selection Streamlit's on_select reads
        # (plotly_selected), which needs "select" in clickmode. Found in Plotly.js's own bundled
        # source (clickmode.indexOf("select")>-1 gating the select-on-click path) after box
        # clicks in the running app produced nothing -- explicit here rather than trusting
        # Streamlit's wrapper to set it for us.
        clickmode="event+select",
    )
    return fig


def render_system_diagram():
    """No selectbox -- click a box directly (Sec. "la barra... me incomoda"). `fixedrange` on
    both axes plus `dragmode=False` keep the chart from panning/zooming on a click that drags a
    few pixels, which is what "no quiero que el gráfico se mueva" was about; scrollZoom off in
    the config below covers the mouse-wheel case the layout options don't."""
    fig = _build_system_diagram_figure()
    event = st.plotly_chart(
        fig, width="stretch", key="system_diagram_chart",
        on_select="rerun", selection_mode="points",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    # Sync a diagram click into session_state -- but only once per distinct click: Streamlit's
    # plotly selection is sticky (persists across unrelated reruns) and PlotlySelectionState is
    # explicitly read-only (can't be cleared from code), so re-asserting the same click's key
    # every single rerun would be harmless here (there's no other widget it could fight anymore)
    # but is kept anyway -- cheap and avoids relying on that stickiness being harmless forever.
    clicked_points = event.selection.points if event else []
    if clicked_points:
        clicked_key = clicked_points[0].get("customdata", [None])[0]
        if clicked_key in _DIAGRAM_COMPONENT_DETAILS and clicked_key != st.session_state.get("_diagram_last_clicked"):
            st.session_state["_diagram_last_clicked"] = clicked_key
            st.session_state["diagram_component_detail"] = clicked_key

    st.caption("Solid arrows: data/artifact flow, in two lanes -- AI (simulation → common data layer → classifier/regressor → registry → edge distillation → dashboard) and design & control (plants → Gymnasium environments → RL Lab / Tuning Studio → experiment runs → dashboard). Dashed arrows: the monitoring layer (Sec. 4.3) — domain telemetry into a tier-appropriate rule agent, escalating toward the PC tier, surfaced as a dashboard badge. Click a box above for its own detail.")

    selected = st.session_state.get("diagram_component_detail")
    if selected is None:
        st.caption("👆 Click any box above to see what it does.")
    else:
        st.info(_DIAGRAM_COMPONENT_DETAILS[selected])

