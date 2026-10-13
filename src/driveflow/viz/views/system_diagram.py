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
#: data-trace points are -- so this is a selectbox + panel instead of an actual click-on-the-box
#: interaction). Ordered the same top-to-bottom/left-to-right path the diagram itself reads in.
_DIAGRAM_COMPONENT_DETAILS = {
    "DC motor + PMSM (Fase A)": (
        "Physics simulation, no ML yet at this stage: `DcPermanentlyExcitedMotor` under a native PI cascade "
        "(speed or torque), plus a separate PMSM FOC/MTPA demo, with bearing-fault injection on two paths "
        "(electrical/MCSA + synthetic vibration). This is domain `dc_motor`'s data source -- see the DC Motor and "
        "PMSM / FOC pages (Simulation section)."
    ),
    "VSC + trained DPC network (Fase B)": (
        "Also physics, plus a separately pre-trained Direct Power Control network (ported from "
        "DPC4PowerElectronics, not part of this AI layer's own training) tracking a rotating voltage "
        "reference. This is domain `vsc_dpc`'s data source -- see the VSC / DPC page."
    ),
    "Common data layer": (
        "`models/common/windowing.py`: turns either domain's raw simulation rows into fixed-size windows "
        "(for the classifier) or `(input_window, horizon)` forecast pairs (for the regressor). Enforces the "
        "domain isolation design doc Sec. 1 requires -- dc_motor and vsc_dpc rows are never combined into "
        "one training set, even though the mechanism (windowing code) is shared."
    ),
    "Classifier (CNN)": (
        "`models/classifiers/builder.py` -- Conv1D+SE blocks, config-driven "
        "(`configs/classifiers/pc_server.yaml` for this tier). Only trained for `dc_motor` so far (89.5% "
        "test accuracy) -- `vsc_dpc`'s classifier is deliberately not built yet, blocked on a separability "
        "verdict (Fase D.2, INSTRUCTIONS.md Sec. 6 Paso 0)."
    ),
    "Regressor (LSTM/GRU/TCN)": (
        "`models/regressors/builder.py` -- stacked LSTM/GRU with optional attention, config-driven "
        "(`configs/regressors/pc_full.yaml` for this tier). Trained for `vsc_dpc` (0.06 normalized test "
        "RMSE) -- `dc_motor` has no regressor yet (no forecasting need identified for that domain so far)."
    ),
    "Model registry": (
        "`ai/registry.py` -- one manifest (`configs/registry.yaml`) mapping `(domain, tier, block)` to the "
        "promoted run folder (weights + the exact config that produced them + metrics). Promotion "
        "(`experiments/promote_run.py`) is a deliberate, separate step from training -- a training run "
        "existing doesn't mean it's live."
    ),
    "Raspberry Pi 5": (
        "**Not trained from scratch.** Distilled from the promoted PC-tier model (the \"teacher\"): "
        "`python experiments/train_model.py --config configs/classifiers/rpi5_resnet1d_se.yaml "
        "--dataset ... --distill` blends the true label/value with the teacher's own prediction "
        "(`--distill-alpha`, default 0.5 -- 1.0 would mean no distillation at all). The architecture itself "
        "is also smaller: fewer/narrower Conv1D+SE blocks for the classifier, a single-layer GRU instead of "
        "the PC tier's stacked LSTM for the regressor. Once promoted, exported to TFLite as **float16** "
        "(`python -m experiments.export_tflite --tier rpi5 ...` -- halves weight size, no calibration data "
        "needed, since RPi5 runs full Linux and can afford it). Current real numbers: dc_motor classifier "
        "86.8% acc (vs. 89.5% teacher), vsc_dpc regressor 0.20 RMSE (vs. 0.06 teacher) -- the accuracy drop "
        "is the expected cost of a smaller model, not a bug."
    ),
    "ESP32": (
        "Same distillation idea as Raspberry Pi 5 (a smaller model, trained from the PC-tier teacher's "
        "predictions via `--distill`), but two tiers stricter: **(1) no recurrent layers at all** -- the "
        "config schema structurally forbids `recurrent_type: lstm`/`gru` here (a real guardrail, not just a "
        "convention), so the regressor uses a small causal TCN instead (dilated Conv1D, `layers: [8, 8]`) "
        "and the classifier uses DS-CNN blocks (depthwise+pointwise Conv1D) instead of RPi5's Conv1D+SE. "
        "**(2) TFLite export is int8** (weights AND activations quantized), which needs REAL calibration "
        "windows from an actual dataset (`--dataset data/....parquet`) -- calibrating against random noise "
        "would badly misjudge the activation ranges the model actually sees. Current real numbers: dc_motor "
        "classifier 86.8% acc, vsc_dpc regressor 0.56 RMSE -- notably worse than RPi5's 0.20 (a much smaller "
        "TCN, and the first training attempt at this tier undertrained at 20 epochs -- 0.95 RMSE -- before "
        "80 epochs got it to 0.56)."
    ),
    "ESP32 watchdog": (
        "Not a model, not related to the \"ESP32\" box above's classifier/regressor -- this is the "
        "**rule-based monitoring** tier (Sec. 4.3), pure hard-coded thresholds with zero ML: "
        "`monitoring/rules/vsc_dpc.yaml` + a validated schema (`schema.py`), e.g. the R∈[1,3]Ω divergence "
        "rule for vsc_dpc. No training, no distillation -- just YAML data plus an AST-checked boolean "
        "condition (never `eval()`'d against arbitrary code)."
    ),
    "GatewayAgent — RPi5": (
        "`monitoring/agents/agent_gateway.py` -- evaluates a domain's ruleset against a telemetry stream "
        "with per-rule hysteresis (must hold N seconds before firing) and debounce (doesn't re-fire every "
        "tick while still true). Can run without a connection to the PC tier below."
    ),
    "ServerAgent — PC": (
        "`monitoring/agents/agent_server.py` -- aggregates the Alert events a GatewayAgent already produced "
        "into a per-domain badge status + history, and separately tracks classifier confidence over time to "
        "flag drift and suggest retraining. Does not evaluate telemetry itself."
    ),
    "Dashboard — IA tab": (
        "`viz/views/ai_diagnostics.py` (AI Diagnostics page) -- pick a domain, generate a sample run or upload "
        "a file, and see the promoted **PC-tier** classifier/regressor output, labelled as an AI suggestion "
        "with its model version. The rest of the AI layer has its own pages now: Transfer Learning (gated "
        "fine-tuning), Model Registry (versions, lineage, artifacts, edge export status) and Monitoring "
        "(agent consensus, derived from ALL active alerts). Edge (.tflite) predictions are not executed in "
        "the dashboard; only the artifacts are listed and downloadable."
    ),
}


#: Single source of truth for the diagram's boxes -- (component_key, cx, cy, w, h, color,
#: display_text). component_key must match a key in _DIAGRAM_COMPONENT_DETAILS (checked by
#: tests/test_dashboard_landing.py); display_text is the multi-line label shown IN the box, while
#: component_key is the short name used by the click handler/selectbox below the diagram.
_DIAGRAM_LAYOUT = [
    ("DC motor + PMSM (Fase A)", 2.3, 1.0, 3.8, 1.05, DARK["primary"], "DC motor + PMSM<br>(Fase A — physics simulation)"),
    ("VSC + trained DPC network (Fase B)", 7.3, 1.0, 3.8, 1.05, DARK["training"], "VSC + trained DPC network<br>(Fase B — physics + real network)"),
    ("Common data layer", 4.8, 2.6, 6.6, 0.95, DARK["text_muted"], "Common data layer — Scenario · windowing · dataset<br>(domains never mixed, see design doc Sec. 1)"),
    ("Classifier (CNN)", 3.0, 4.15, 3.2, 0.95, DARK["healthy"], "Classifier (CNN)<br>tier: PC"),
    ("Regressor (LSTM/GRU/TCN)", 6.6, 4.15, 3.6, 0.95, DARK["healthy"], "Regressor (LSTM/GRU/TCN)<br>tier: PC"),
    ("Model registry", 4.8, 5.65, 7.2, 0.95, DARK["warning"], "Model registry — ai/registry.py<br>(domain, tier, block) → promoted artifact"),
    ("Raspberry Pi 5", 3.0, 7.1, 3.6, 0.95, "#F472B6", "Raspberry Pi 5<br>distilled + TFLite float16"),
    ("ESP32", 6.6, 7.1, 3.6, 0.95, "#F472B6", "ESP32<br>distilled + TFLite int8"),
    ("Dashboard — IA tab", 4.8, 8.55, 7.0, 0.95, "#2563EB", "Dashboard — IA tab<br>classifier/regressor panels + status badges"),
    ("ESP32 watchdog", 10.9, 1.0, 2.7, 0.85, DARK["unknown"], "ESP32 watchdog<br>hard thresholds, no ML"),
    ("GatewayAgent — RPi5", 10.9, 3.3, 2.7, 0.85, DARK["unknown"], "GatewayAgent — RPi5<br>hysteresis + debounce"),
    ("ServerAgent — PC", 10.9, 5.65, 2.7, 0.85, DARK["unknown"], "ServerAgent — PC<br>alert history + confidence drift"),
]
#: Self-enforcing instead of relying on a separate test to catch drift: every AppTest run of the
#: landing page executes this at import/script-run time, so a mismatch surfaces immediately as
#: `at.exception` rather than silently leaving a diagram box whose click has nowhere to go (or a
#: selectbox entry no box ever points at).
assert {row[0] for row in _DIAGRAM_LAYOUT} == set(_DIAGRAM_COMPONENT_DETAILS), (
    f"diagram/detail key mismatch: {({row[0] for row in _DIAGRAM_LAYOUT} ^ set(_DIAGRAM_COMPONENT_DETAILS))}"
)


def _build_system_diagram_figure() -> go.Figure:
    """Pure figure construction (no `st.*` calls) -- separated from _render_system_diagram so it
    can be unit-tested directly (tests/test_dashboard_landing.py) without a Streamlit script
    context. A block diagram of the whole platform (not just the AI layer): both macro-phases'
    physics/plants, the config-driven IA layer (docs/design_ai_layer_transversal.md), the model
    registry, edge-tier distillation, and the monitoring agents, laid out top-to-bottom the same
    way data actually flows. Plotly shapes/annotations/traces, not a new dependency
    (st.graphviz_chart would need the `graphviz` package AND the system `dot` binary neither of
    which this project already depends on; a JS-based Mermaid embed would need loading a script
    from a CDN at runtime, which this offline-capable research tool avoids elsewhere too).

    Each box is clickable (_diagram_click_target's invisible marker grid) -- clicking one selects
    it in the detail panel below, same as picking it from the selectbox; both write to the same
    session_state key, so either path stays in sync with the other."""
    fig = go.Figure()

    for _key, cx, cy, w, h, color, display_text in _DIAGRAM_LAYOUT:
        _diagram_box(fig, cx, cy, w, h, display_text, color)
        _diagram_click_target(fig, cx, cy, w, h, _key)

    # Data/artifact flow (solid).
    _diagram_arrow(fig, 2.3, 1.55, 4.0, 2.1)
    _diagram_arrow(fig, 7.3, 1.55, 5.6, 2.1)
    _diagram_arrow(fig, 4.0, 3.1, 3.0, 3.65)
    _diagram_arrow(fig, 5.6, 3.1, 6.6, 3.65)
    _diagram_arrow(fig, 3.0, 4.65, 4.0, 5.15)
    _diagram_arrow(fig, 6.6, 4.65, 5.6, 5.15)
    _diagram_arrow(fig, 4.0, 6.15, 3.0, 6.6)
    _diagram_arrow(fig, 5.6, 6.15, 6.6, 6.6)
    _diagram_arrow(fig, 4.8, 6.15, 4.8, 8.05)

    # Monitoring flow (dashed) -- domain telemetry into the tier-appropriate rule agent, alerts
    # escalating PC-ward, and the aggregated status reaching the dashboard as a badge. Enters the
    # column from the common data layer (both domains' own telemetry, simplified to one entry
    # point) rather than from box 1/2 directly -- a straight line from either of those would cut
    # across the OTHER domain's box, since both sit in the same row as the watchdog.
    _diagram_arrow(fig, 7.5, 2.125, 10.9, 1.425, dash=True)
    _diagram_arrow(fig, 10.9, 1.425, 10.9, 2.875, dash=True)
    _diagram_arrow(fig, 10.9, 3.725, 10.9, 5.225, dash=True)
    _diagram_arrow(fig, 10.9, 6.075, 8.3, 8.55, dash=True)

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 12.6], fixedrange=True),
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

    st.caption("Solid arrows: data/artifact flow (simulation → common data layer → classifier/regressor → registry → edge distillation → dashboard). Dashed arrows: the monitoring layer (Sec. 4.3) — domain telemetry into a tier-appropriate rule agent, escalating toward the PC tier, surfaced as a dashboard badge. Click a box above for its own detail.")

    selected = st.session_state.get("diagram_component_detail")
    if selected is None:
        st.caption("👆 Click any box above to see what it does.")
    else:
        st.info(_DIAGRAM_COMPONENT_DETAILS[selected])

