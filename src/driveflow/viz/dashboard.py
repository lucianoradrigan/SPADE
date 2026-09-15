"""Streamlit interface matching the 3-tab structure of this session's earlier HTML dashboard
artifact (https://claude.ai/code/artifact/dbb26b3d-...): "Corrientes y vibración", "Envolvente
(ω, i_a) DC", "Plano i_d-i_q PMSM" -- now backed by a live local server instead of a frozen data
snapshot. INSTRUCTIONS.md Sec. 7's Fase E, pulled forward and scoped down to Fase A (no Fase C/D
integration yet -- no trained/validated models to show there).

NOT a literal reuse of GEM's `motor_dashboard.py`, despite INSTRUCTIONS.md Sec. 7 suggesting it as
a base: that file is deeply coupled to `gymnasium.Env`'s step-callback pattern
(`on_step_begin`/`on_step_end`/`on_reset_end`, `env.physical_system`, `env.reference_generator`)
and live-updating matplotlib windows meant for a local desktop training loop -- the same kind of
Gym-Env coupling this project has decoupled from everywhere else (see
control/classical/pi_controller.py's docstring for the same situation with `gem_controllers`).
Rebuilt natively against `driveflow.datagen.Scenario`/`run_scenario()` and
`control/classical/pmsm_foc.py` instead; only the general idea (configurable per-signal plots)
carries over.

Chart titles are rendered as Streamlit markdown ABOVE each `st.plotly_chart`, never as Plotly's
own `layout.title` -- found by screenshotting the actual rendered page (not by inspection) that
Plotly's title and a top-anchored legend both fight for the same vertical space and visibly
overlap; keeping the two title mechanisms apart sidesteps that entirely rather than fine-tuning
coordinates against it.

Visual identity: an "industrial workbench" theme (Grafana / JetBrains UI / LabVIEW Modern UI, not
a generic AI-template look) in dark OR light mode, picked at runtime by the toggle button next to
the header (persisted in st.session_state["theme_mode"], defaulting to dark) -- reversing an
earlier commit to a single fixed dark theme, per explicit later feedback that a light option was
wanted back. CANVAS/SURFACE/BORDER/etc. below are no longer fixed hex constants: they're resolved
from _PALETTES[theme_mode] once per script run, so every one of their ~100 usage sites throughout
this file (the CSS injected below, every Plotly figure's colors) automatically follows the current
mode with no per-site changes needed. .streamlit/config.toml's [theme] is left on the dark values
as a static fallback for whatever native-widget chrome the CSS overrides below don't reach (some
deep BaseWeb internals aren't practically targetable -- see the CSS block's own "native widget
re-theme" comment); it cannot itself switch at runtime without a server restart.

Run with:
    streamlit run src/driveflow/viz/dashboard.py
(needs the `viz` extra: `uv pip install -e ".[viz]"`)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from driveflow.control.classical import generate_mtpa_vs_naive_cloud
from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS as PMSM_DEFAULTS
from driveflow.datagen import Scenario, run_flow, run_scenario
from driveflow.control.dpc import COLUMNS as DPC_COLUMNS
from driveflow.control.dpc import HORIZON as DPC_HORIZON
from driveflow.control.dpc import build_dpc_network, simulate_horizon
from driveflow.viz.ai_dashboard import _render_fase_ia
from driveflow.viz.transfer_learning_dashboard import _render_fase_tl
from driveflow.viz.dpc_upload_validation import DPC_AUTOFILL_COLUMNS, DPC_REQUIRED_COLUMNS, validate_dpc_upload
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V, RotatingReference
from driveflow.datagen.runner import _DPC_WEIGHTS_PATH, _VSC_R_OHM
from driveflow.datagen.runner import TAU as _TAU_S
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.sim import DcPermanentlyExcitedMotor
from driveflow.sim.vibration import bearing_frequencies as bf
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM

st.set_page_config(page_title="driveflow", layout="wide", page_icon="📈")

# ---------------------------------------------------------------------------
# Design tokens -- two full palettes (dark/light), not fixed hex constants. dark's values stay
# numerically identical to .streamlit/config.toml's [theme] block (that file can't itself switch
# at runtime, see module docstring); light is a fresh set with the same semantic roles, darkened
# where needed (ACCENT/INNER/MTPA_COLOR/NAIVE_COLOR/ISOLINE_COLOR/REF_GREY) for contrast against
# a white surface instead of a dark one.
# ---------------------------------------------------------------------------
_PALETTES = {
    "dark": dict(
        CANVAS="#0F172A", SURFACE="#1E293B", BORDER="#334155", INK="#E2E8F0", INK_2="#94A3B8",
        ACCENT="#0EA5E9", ACCENT_2="#2563EB", GRID="#293548",
        INNER="#F472B6", REF_GREY="#64748B", MTPA_COLOR="#34D399", NAIVE_COLOR="#A78BFA", ISOLINE_COLOR="#FB923C",
    ),
    "light": dict(
        CANVAS="#F8FAFC", SURFACE="#FFFFFF", BORDER="#CBD5E1", INK="#0F172A", INK_2="#64748B",
        ACCENT="#0284C7", ACCENT_2="#2563EB", GRID="#E2E8F0",
        INNER="#DB2777", REF_GREY="#475569", MTPA_COLOR="#059669", NAIVE_COLOR="#7C3AED", ISOLINE_COLOR="#EA580C",
    ),
}
theme_mode = st.session_state.get("theme_mode", "dark")
_palette = _PALETTES[theme_mode]
CANVAS = _palette["CANVAS"]  # page/app background
SURFACE = _palette["SURFACE"]  # cards, sidebar, plot backgrounds
BORDER = _palette["BORDER"]  # hairline borders/separators
INK = _palette["INK"]  # primary text
INK_2 = _palette["INK_2"]  # secondary/muted text
ACCENT = _palette["ACCENT"]  # primary interactive (sliders, active tab, primary buttons)
ACCENT_2 = _palette["ACCENT_2"]  # secondary accent (hover states)
GRID = _palette["GRID"]  # chart gridlines
PLOT_BG = SURFACE

HEALTHY = ACCENT  # main measured trace (current/rpm/torque/acc, operating-envelope trajectory)
INNER = _palette["INNER"]  # current-limit reference line
REF_GREY = _palette["REF_GREY"]  # dashed reference/limit lines
MTPA_COLOR = _palette["MTPA_COLOR"]  # FOC+MTPA point cloud
NAIVE_COLOR = _palette["NAIVE_COLOR"]  # FOC naive (i_d=0) point cloud
ISOLINE_COLOR = _palette["ISOLINE_COLOR"]  # analytic MTPA locus

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    h1, h2, h3, h4, h5 {{ font-family: 'Inter', sans-serif !important; font-weight: 600; letter-spacing: -0.01em; }}

    /* Numbers/ranges/units in mono, per the workbench's data-dense reading convention. */
    .stNumberInput input, .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"],
    div[data-testid="stMetricValue"], .stMetric label, .stMetric div, code {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ---- Sticky technical header ------------------------------------------------------- */
    .df-header {{
        position: sticky; top: 0; z-index: 999;
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
        gap: 0.4rem 1rem;
        padding: 0.6rem 1rem; margin-bottom: 1rem;
        background: {CANVAS};
        border-bottom: 1px solid {BORDER};
    }}
    .df-header-left {{ display: flex; align-items: baseline; gap: 0.75rem; }}
    .df-header-title {{ font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.05rem; color: {INK}; letter-spacing: 0.01em; }}
    .df-header-sub {{ font-family: 'Inter', sans-serif; font-size: 0.78rem; color: {INK_2}; }}
    .df-header-right {{ display: flex; align-items: center; gap: 0.5rem; }}
    .df-badge {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {INK_2};
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px; padding: 3px 10px;
        white-space: nowrap;
    }}
    .df-badge-live {{ color: {ACCENT}; border-color: {ACCENT}; }}

    /* ---- Sidebar: section eyebrows + card headers -------------------------------------- */
    section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
    .df-sidebar-title {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase; color: {INK};
        margin: 0.6rem 0 0.1rem 0;
    }}
    .df-sidebar-hint {{ font-family: 'Inter', sans-serif; font-size: 0.72rem; color: {INK_2}; margin-bottom: 0.4rem; }}
    .df-card-header {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.07em; text-transform: uppercase; color: {ACCENT};
        margin-bottom: 0.3rem;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 10px !important; border-color: {BORDER} !important; }}

    /* ---- Tabs as a segmented control, not a plain text menu ---------------------------- */
    div[data-baseweb="tab-list"] {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 4px; gap: 4px;
    }}
    button[data-baseweb="tab"] {{
        border-radius: 8px !important; font-family: 'Inter', sans-serif; font-weight: 500;
        color: {INK_2};
    }}
    button[data-baseweb="tab"]:hover {{ background: rgba(14, 165, 233, 0.10); color: {INK}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: {CANVAS}; color: {INK}; box-shadow: inset 0 0 0 1px {BORDER};
    }}
    div[data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}
    div[data-baseweb="tab-border"] {{ background-color: transparent !important; }}

    /* ---- Empty-state placeholders (replace the default blue st.info banners) ----------- */
    .df-empty {{
        border: 1px dashed {BORDER}; border-radius: 12px; padding: 4.5rem 1.5rem; text-align: center;
        background-color: rgba(30, 41, 59, 0.35);
        background-image: linear-gradient({BORDER} 1px, transparent 1px), linear-gradient(90deg, {BORDER} 1px, transparent 1px);
        background-size: 28px 28px; background-position: center;
    }}
    .df-empty-icon {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; color: {ACCENT}; margin-bottom: 0.5rem; display: inline-block; animation: df-float 2.4s ease-in-out infinite; }}
    .df-empty-text {{ font-family: 'Inter', sans-serif; font-size: 0.88rem; color: {INK_2}; }}
    .df-empty-text b {{ color: {INK}; font-weight: 600; }}

    /* ---- Motion: precise/technical transitions, not bouncy/playful ones ---------------- */
    /* Custom dark scrollbar -- the default browser one breaks the workbench illusion. */
    * {{ scrollbar-width: thin; scrollbar-color: {BORDER} {CANVAS}; }}
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {CANVAS}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 6px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

    /* Buttons: hover glow + a real press (mousedown) feedback, like a physical panel switch. */
    .stButton button, .stDownloadButton button {{
        transition: transform 0.1s ease, box-shadow 0.15s ease, filter 0.15s ease, border-color 0.15s ease;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{ filter: brightness(1.12); box-shadow: 0 4px 14px -4px {ACCENT}66; }}
    .stButton button:active, .stDownloadButton button:active {{ transform: scale(0.97); }}

    /* Tabs: the existing highlight bar now slides instead of jump-cutting between tabs. */
    div[data-baseweb="tab-highlight"] {{ transition: left 0.25s cubic-bezier(0.2, 0.7, 0.3, 1), width 0.25s cubic-bezier(0.2, 0.7, 0.3, 1) !important; }}
    button[data-baseweb="tab"] {{ transition: background-color 0.15s ease, color 0.15s ease; }}

    /* Live status badge: a small pulsing LED, like real instrument panels use for "armed/ready". */
    .df-badge-live {{ position: relative; padding-left: 18px; }}
    .df-badge-live::before {{
        content: ''; position: absolute; left: 8px; top: 50%; width: 6px; height: 6px; margin-top: -3px;
        border-radius: 50%; background: {ACCENT}; animation: df-pulse 1.8s ease-in-out infinite;
    }}
    @keyframes df-pulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 {ACCENT}66; }}
        70% {{ box-shadow: 0 0 0 6px {ACCENT}00; }}
    }}
    @keyframes df-float {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-4px); }} }}

    /* Landing-page phase cards ONLY (a stable, per-card key -- see _render_landing_page -- keeps
       this off every other bordered container in the app; those re-render on every sidebar
       interaction elsewhere and would replay the entrance animation on each one). */
    @keyframes df-card-in {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    [class*="st-key-landing_card_"] {{
        animation: df-card-in 0.45s cubic-bezier(0.2, 0.7, 0.3, 1) both;
        transition: transform 0.18s cubic-bezier(0.2, 0.7, 0.3, 1), box-shadow 0.18s ease, border-color 0.18s ease;
    }}
    [class*="st-key-landing_card_"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 28px -10px {ACCENT}55, 0 0 0 1px {ACCENT}66;
        border-color: {ACCENT} !important;
    }}
    .st-key-landing_card_B {{ animation-delay: 0.08s; }}
    .st-key-landing_card_IA {{ animation-delay: 0.16s; }}
    .st-key-landing_card_TL {{ animation-delay: 0.24s; }}

    /* The theme-toggle button's own row -- pulled up tight against the header markdown right
       below it (a real st.button can't live INSIDE that header's raw HTML, see the comment at
       its call site) so it reads as one header unit instead of two stacked blocks. */
    .df-theme-toggle-row {{ margin-bottom: -2.75rem; position: relative; z-index: 1000; }}
    .df-theme-toggle-row .stButton button {{
        font-size: 0.7rem; padding: 0.15rem 0.6rem; min-height: 0;
        background: {SURFACE}; border: 1px solid {BORDER}; color: {INK_2};
    }}
    .df-theme-toggle-row .stButton button:hover {{ color: {INK}; border-color: {ACCENT}; }}

    /* ---- Native-widget re-theme: config.toml's [theme] is fixed on dark (it can't itself
       switch at runtime, see module docstring) -- these overrides make the page/sidebar
       background and the most visible native controls follow theme_mode too. Some deep BaseWeb
       internals (e.g. a checkbox's own tick glyph) aren't practically reachable here and may
       still assume dark; flag anything that looks wrong in light mode. ------------------------ */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{
        background-color: {CANVAS} !important;
    }}
    section[data-testid="stSidebar"] {{ background-color: {SURFACE} !important; }}
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, [data-testid="stMarkdownContainer"] {{ color: {INK}; }}
    [data-testid="stAlert"] {{ background-color: {SURFACE}; border: 1px solid {BORDER}; }}
    .stSlider [data-baseweb="slider"] div {{ background-color: {BORDER}; }}
    div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
        background-color: {SURFACE} !important; border-color: {BORDER} !important; color: {INK} !important;
    }}
    div[data-testid="stExpander"] {{ border-color: {BORDER} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.html(
    f"""
    <canvas id="df-scope" style="display:block; width:100%; height:40px; background:{CANVAS};"></canvas>
    <script>
    (function() {{
        const canvas = document.getElementById('df-scope');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        function resize() {{
            canvas.width = canvas.clientWidth * dpr;
            canvas.height = canvas.clientHeight * dpr;
        }}
        resize();
        window.addEventListener('resize', resize);
        let t = 0;
        function draw() {{
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '{BORDER}';
            ctx.lineWidth = 1 * dpr;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
            // A traveling waveform, amplitude-enveloped to zero at both edges (sin(frac*PI)) so
            // it reads as a live oscilloscope sweep, not a plain repeating sine strip.
            ctx.strokeStyle = '{ACCENT}';
            ctx.lineWidth = 1.4 * dpr;
            ctx.beginPath();
            const n = 220;
            for (let i = 0; i <= n; i++) {{
                const frac = i / n;
                const x = frac * w;
                const envelope = Math.sin(frac * Math.PI);
                const y = h / 2 + Math.sin(frac * 26 - t) * (h * 0.32) * envelope;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }}
            ctx.stroke();
            t += 0.05;
            requestAnimationFrame(draw);
        }}
        draw();
    }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# Real st.button (a raw HTML button inside the header's own unsafe_allow_html markdown below
# has no way to run Python -- Streamlit widgets are always separate DOM nodes in the normal
# document flow), right-aligned via st.columns and pulled up against the header underneath it
# by the .df-theme-toggle-row CSS above -- reads as "part of" the header row without literally
# being nested inside its markup.
st.markdown('<div class="df-theme-toggle-row">', unsafe_allow_html=True)
_toggle_col_spacer, _toggle_col_button = st.columns([0.92, 0.08])
with _toggle_col_button:
    if st.button("☀️ Light" if theme_mode == "dark" else "🌙 Dark", key="theme_toggle", help="Switch between light and dark theme", width="stretch"):
        st.session_state["theme_mode"] = "light" if theme_mode == "dark" else "dark"
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="df-header">
        <div class="df-header-left">
            <span class="df-header-title">DRIVEFLOW</span>
            <span class="df-header-sub">Live simulation workbench — DC motor PI + PMSM FOC/MTPA, runs on click, no pre-generated data</span>
        </div>
        <div class="df-header-right">
            <span class="df-badge df-badge-live">● ENGINE READY</span>
            <span class="df-badge">Ts = {_TAU_S * 1e6:.0f} µs</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _diagram_box(fig, cx, cy, w, h, text, color):
    """One labeled rectangle for _render_system_diagram -- a filled shape (the box) plus a
    centered annotation (the text), since Plotly shapes carry no text of their own. Fill is only
    a ~15% tint of `color` over the page's dark background (not a solid fill), so INK (light
    text) reads fine on every box regardless of which semantic `color` it uses."""
    fig.add_shape(
        type="rect", x0=cx - w / 2, y0=cy - h / 2, x1=cx + w / 2, y1=cy + h / 2,
        line=dict(color=color, width=1.5), fillcolor=f"{color}26",
    )
    fig.add_annotation(x=cx, y=cy, text=text, showarrow=False, font=dict(color=INK, size=11), align="center")


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
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=REF_GREY, width=1.3, dash="dot"))
    else:
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1, axref="x", ayref="y", xref="x", yref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.3, arrowcolor=REF_GREY,
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
        "(electrical/MCSA + synthetic vibration). This is domain `dc_motor`'s data source -- see tabs 01-03 "
        "in Fase A."
    ),
    "VSC + trained DPC network (Fase B)": (
        "Also physics, plus a separately pre-trained Direct Power Control network (ported from "
        "DPC4PowerElectronics, not part of this AI layer's own training) tracking a rotating voltage "
        "reference. This is domain `vsc_dpc`'s data source -- see Fase B's tabs."
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
        "`viz/ai_dashboard.py` -- what you actually interact with: pick a domain, generate a sample run or "
        "upload a file, see the **PC-tier** classifier/regressor's live prediction. **Current limitation:** "
        "it does not yet show the RPi5/ESP32 edge predictions above, nor the monitoring agents' badge "
        "status -- both exist as real backend artifacts (registry entries, `.tflite` files, agent classes "
        "with tests) but nothing in the frontend queries them yet."
    ),
}


#: Single source of truth for the diagram's boxes -- (component_key, cx, cy, w, h, color,
#: display_text). component_key must match a key in _DIAGRAM_COMPONENT_DETAILS (checked by
#: tests/test_dashboard_landing.py); display_text is the multi-line label shown IN the box, while
#: component_key is the short name used by the click handler/selectbox below the diagram.
_DIAGRAM_LAYOUT = [
    ("DC motor + PMSM (Fase A)", 2.3, 1.0, 3.8, 1.05, ACCENT, "DC motor + PMSM<br>(Fase A — physics simulation)"),
    ("VSC + trained DPC network (Fase B)", 7.3, 1.0, 3.8, 1.05, NAIVE_COLOR, "VSC + trained DPC network<br>(Fase B — physics + real network)"),
    ("Common data layer", 4.8, 2.6, 6.6, 0.95, INK_2, "Common data layer — Scenario · windowing · dataset<br>(domains never mixed, see design doc Sec. 1)"),
    ("Classifier (CNN)", 3.0, 4.15, 3.2, 0.95, MTPA_COLOR, "Classifier (CNN)<br>tier: PC"),
    ("Regressor (LSTM/GRU/TCN)", 6.6, 4.15, 3.6, 0.95, MTPA_COLOR, "Regressor (LSTM/GRU/TCN)<br>tier: PC"),
    ("Model registry", 4.8, 5.65, 7.2, 0.95, ISOLINE_COLOR, "Model registry — ai/registry.py<br>(domain, tier, block) → promoted artifact"),
    ("Raspberry Pi 5", 3.0, 7.1, 3.6, 0.95, INNER, "Raspberry Pi 5<br>distilled + TFLite float16"),
    ("ESP32", 6.6, 7.1, 3.6, 0.95, INNER, "ESP32<br>distilled + TFLite int8"),
    ("Dashboard — IA tab", 4.8, 8.55, 7.0, 0.95, ACCENT_2, "Dashboard — IA tab<br>classifier/regressor panels + status badges"),
    ("ESP32 watchdog", 10.9, 1.0, 2.7, 0.85, REF_GREY, "ESP32 watchdog<br>hard thresholds, no ML"),
    ("GatewayAgent — RPi5", 10.9, 3.3, 2.7, 0.85, REF_GREY, "GatewayAgent — RPi5<br>hysteresis + debounce"),
    ("ServerAgent — PC", 10.9, 5.65, 2.7, 0.85, REF_GREY, "ServerAgent — PC<br>alert history + confidence drift"),
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


def _render_system_diagram():
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


def _render_about_content():
    """Always visible on the landing page (no click needed) -- what's simulated, inputs/outputs,
    how to read the charts, across all 3 systems this platform covers (Fase A's two plants, Fase
    B's one)."""
    st.markdown("##### System diagram")
    _render_system_diagram()
    st.divider()
    st.markdown("##### About this platform — what's simulated, inputs/outputs, how to read the charts")
    col_dc, col_pmsm, col_vsc = st.columns(3)
    with col_dc:
        st.markdown(
            """
##### System 1 — DC motor (tabs 01-02)
**Plant**: a real permanently-excited DC motor (`DcPermanentlyExcitedMotor`), the same physics/parameters as [gym-electric-motor](https://github.com/upb-lea/gym-electric-motor), driven by a one-quadrant converter.

**Controller**: a native cascaded PI controller. Tracks either a **speed** setpoint (outer speed loop -> inner current loop) or a **torque** setpoint directly (bypasses the speed loop) -- pick which in Control mode.

**Faults**: bearing defects (outer/inner race, ball, cage, or a custom one you define) injected on two independent, physically distinct paths -- torque ripple on the current signal (electrical/MCSA), and impulse trains on synthetic 3-axis vibration (Module B). "Noise" (below that) is separate: generic sensor-style jitter, not a fault mechanism.

**You control**: fault type & severity, control mode & setpoint, run duration, random seed, the motor's own characteristics (resistance/inductance/flux), and how much white noise to add.

**You get out**: `i_a` (current), `rpm`, `torque`, `u_a` (armature voltage -- the controller's actual commanded/applied voltage, always plotted regardless of whether the setpoint above is speed or torque), and 3-axis synthetic vibration -- from a real closed-loop physics simulation run fresh every time you click Generate, never pre-computed or looked up.
            """
        )
    with col_pmsm:
        st.markdown(
            """
##### System 2 — PMSM FOC/MTPA (tab 03)
**Plant**: a different, salient permanent-magnet synchronous motor (unrelated to the DC system on the left -- see its own motor-characteristics panel).

**Controller**: a native dq-frame field-oriented current controller, run under two competing policies at the same current magnitude: **MTPA** (max torque per ampere, the closed-form-optimal current split) vs. **naive** (all current on the torque axis, none on the reluctance-exploiting axis).

**You control**: the motor's own characteristics (pole pairs, inductances, resistance, flux), how many current magnitudes to sweep, and how densely to sample each transient.

**You get out**: simulated (i_d, i_q) trajectories for both policies, plotted against the analytic MTPA curve and the current limit -- showing concretely how much torque MTPA buys you over naive control at the same current.

**No fault model** exists for this motor yet -- tab 03 is a control-law comparison, not a diagnosis scenario.
            """
        )
    with col_vsc:
        st.markdown(
            """
##### System 3 — DPC Voltage Source Converter (Fase B)
**Plant**: a Voltage Source Converter (VSC) -- power electronics, no motor, no rotating machinery, no bearing.

**Controller**: a trained Direct Power Control (DPC) neural network tracking a rotating voltage reference (fixed magnitude, fixed frequency).

**You control**: run duration, seed (reference start phase), and 3 off-distribution robustness probes -- load resistance, reference magnitude, reference frequency -- all defaulting to the exact values the network was trained on.

**You get out**: `v_c` (actual voltage) vs. `v_ref` (commanded), `i_f` (filter current), and closed-loop RMSE -- from the real trained network driving the real identified plant, live.

**Ported from**: [DPC4PowerElectronics](https://github.com/aipoweraau/DPC4PowerElectronics) -- the original MATLAB DPC/VSC implementation (network architecture, identified plant matrices, and training loss all verified line-for-line against this source).
            """
        )
    st.markdown(
        """
---
##### Reading the tabs
**Fase A** -- 01 Corrientes y vibración: the DC system's electrical (i_a, rpm, torque, u_a) and vibration (acc_x/y/z) signals over time. 02 Envolvente (ω, i_a) DC: the same run's (ω, i_a) path against the motor's physical voltage/current limit curves. 03 Plano i_d–i_q PMSM: MTPA vs. naive point clouds against the analytic MTPA locus and current-limit circle.

**Fase B** -- 01 Tracking (t): v_ref vs. v_c and i_f over time. 02 Complex voltage plane: the same run's v_c trajectory against v_ref in the complex plane.

**Nothing on this platform is pre-generated, faked, or looked up from a table.** Every number and every chart is the direct output of a real simulation (physics-based for Fase A, the real trained network for Fase B), run fresh every time you click Generate. There is currently no trained diagnosis/classification model wired into Fase A's view -- that's a separate, later stage of the project.
        """
    )


#: Technical names/descriptions for the two macro-phases this platform covers -- shown on the
#: landing page (_render_landing_page) and reused for the sidebar's "current phase" badge, so the
#: two only ever need to be edited in one place.
_PHASES = {
    "A": dict(
        title="Macro-Fase A — DC Motor Diagnosis",
        summary="Permanently-excited DC motor, PI cascade (speed or torque).",
        description=(
            "A permanently-excited DC motor under a native PI cascade controller (speed or torque setpoint), "
            "with bearing-fault injection on two independent paths -- electrical/MCSA (torque ripple) and "
            "synthetic 3-axis vibration (Module B) -- plus a native PMSM FOC/MTPA control-law demo. Single "
            "simulations, or multi-segment Advanced Flows (a sequence of operating states run back-to-back)."
        ),
    ),
    "B": dict(
        title="Macro-Fase B — DPC Voltage Source Converter",
        summary="Trained Direct Power Control network, a Voltage Source Converter.",
        description=(
            "A trained Direct Power Control (DPC) neural network tracking a rotating voltage reference on a "
            "Voltage Source Converter -- a completely different plant from Fase A: no rotating machinery, no "
            "bearing, no fault model. Tracking accuracy shown in time and in the complex voltage plane."
        ),
    ),
    "IA": dict(
        title="IA — Cross-domain classifiers & regressors",
        summary="Config-driven classifiers/regressors, consuming data already generated (or uploaded).",
        description=(
            "Evaluates a per-domain classifier and/or regressor -- registered separately by domain, never "
            "sharing training data across Fase A/B (docs/design_ai_layer_transversal.md) -- against a sample "
            "simulation run or an uploaded file. A consumer of data, not a live control surface: no sidebar "
            "controls here re-run Fase A/B's own simulations."
        ),
    ),
    "TL": dict(
        title="Transfer Learning — Fine-tune a promoted model",
        summary="Fine-tune an already-promoted PC-tier model against new (simulated + external) data.",
        description=(
            "Loads whichever PC-tier model is currently promoted for a domain, fine-tunes it against a mix "
            "of freshly-simulated and (optionally) uploaded data using one of three strategies (feature "
            "extractor, discriminative LR, adapter), and -- if validation passes -- promotes the result as a "
            "new run, ready for the existing --distill path to push down to Raspberry Pi 5/ESP32."
        ),
    ),
}


def _render_landing_page():
    st.markdown(
        f"""
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; color:{INK_2};">Select a macro-phase</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(_PHASES))
    for col, phase_id in zip(cols, _PHASES):
        phase = _PHASES[phase_id]
        with col:
            # key="landing_card_*" gives this specific container a stable ".st-key-landing_card_*"
            # class (Streamlit's own convention for a keyed container) -- the hover-lift/entrance
            # animation in the injected CSS above targets that class specifically, NOT every
            # st.container(border=True) in the app (sidebar scenario cards rerun on every slider
            # tweak; animating those would replay constantly instead of once per landing view).
            card = st.container(border=True, key=f"landing_card_{phase_id}")
            card.markdown(f"##### {phase['title']}")
            card.markdown(phase["description"])
            if card.button(f"Enter Fase {phase_id} →", key=f"enter_phase_{phase_id}", type="primary", width="stretch"):
                st.session_state["selected_phase"] = phase_id
                st.rerun()

    # Always visible here, no click needed -- once a phase is chosen there's a "← Back to phase
    # selection" button to get back here if this is needed again, so it doesn't need to persist
    # inside every phase's own sidebar too.
    st.divider()
    _render_about_content()


_selected_phase = st.session_state.get("selected_phase")

if _selected_phase is None:
    _render_landing_page()
    st.stop()

_phase_info = _PHASES[_selected_phase]
st.sidebar.markdown(f'<div class="df-badge" style="display:block; text-align:center; margin-bottom:0.5rem;">{_phase_info["title"]}</div>', unsafe_allow_html=True)
if st.sidebar.button("← Back to phase selection", key="back_to_phases"):
    st.session_state["selected_phase"] = None
    st.rerun()
st.sidebar.divider()


def _card_header(container, label: str):
    container.markdown(f'<div class="df-card-header">{label}</div>', unsafe_allow_html=True)


def _empty_state(instruction_html: str):
    """Grid-pattern technical placeholder, replacing the default blue st.info banner for
    "nothing generated yet" states -- purely presentational, same instruction text as before."""
    st.markdown(f'<div class="df-empty"><div class="df-empty-icon">⌁</div><div class="df-empty-text">{instruction_html}</div></div>', unsafe_allow_html=True)


def _base_layout(fig, height=320):
    """No `title` handling here -- see module docstring on why chart titles are Streamlit
    markdown, not Plotly's layout.title."""
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=20, t=44, b=36),
        font=dict(family="Inter, sans-serif", size=12, color=INK),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11, color=INK_2, family="Inter, sans-serif")),
    )
    # Tick labels (the actual numbers) in mono; axis titles inherit the Inter font above -- same
    # "Inter for labels, JetBrains Mono for numbers/units" split as the sidebar's own typography.
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=INK_2, tickfont=dict(family="JetBrains Mono, monospace", size=11, color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=INK_2, tickfont=dict(family="JetBrains Mono, monospace", size=11, color=INK_2))
    return fig


def _plotly_lines(df, rows, ref_cols=None):
    """ref_cols: one entry per row, aligned with `rows` -- None (no reference line), a column name
    (string, looked up in df -- e.g. Fase B's time-varying v_ref_real), a plain number (a constant
    setpoint held for the whole run, e.g. Fase A's single-simulation ω_ref/τ_ref -- broadcast to a
    flat dashed line), or an array already the same length as df (e.g. Advanced Flow's per-segment
    setpoint, which can change control mode/value partway through -- NaN stretches in that array
    render as gaps, which is exactly "no reference active" for a segment controlled the other way)."""
    ref_cols = ref_cols or [None] * len(rows)
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[y for _, y in rows])
    for i, ((col, ylabel), ref_col) in enumerate(zip(rows, ref_cols), start=1):
        fig.add_trace(
            go.Scatter(x=df["timestamp_s"], y=df[col], mode="lines", name=col, line=dict(width=1.6, color=HEALTHY), showlegend=False),
            row=i,
            col=1,
        )
        if ref_col is not None:
            if isinstance(ref_col, str):
                ref_y = df[ref_col]
            elif np.isscalar(ref_col):
                ref_y = [ref_col] * len(df)
            else:
                ref_y = ref_col
            fig.add_trace(
                go.Scatter(x=df["timestamp_s"], y=ref_y, mode="lines", name="reference", line=dict(width=1.2, color=REF_GREY, dash="dash"), showlegend=(i == 1)),
                row=i,
                col=1,
            )
    fig.update_xaxes(title_text="t (s)", row=len(rows), col=1)
    fig.update_annotations(font=dict(family="Inter, sans-serif", size=13, color=INK))
    return _base_layout(fig, height=170 * len(rows) + 40)


def _operating_envelope_figure(df, dc_params):
    r_a, psi_e, u_lim, i_lim, omega_lim = dc_params["r_a"], dc_params["psi_e"], dc_params["u_lim"], dc_params["i_lim"], dc_params["omega_lim"]
    omega = df["rpm"] * 2 * np.pi / 60.0
    i_a = df["current_r"]
    om_range = np.linspace(0, omega_lim, 200)
    i_voltage_limited = np.clip((u_lim - psi_e * om_range) / r_a, 0, i_lim)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=om_range, y=i_voltage_limited, mode="lines", name="voltage-limited envelope", line=dict(color=REF_GREY, dash="dot", width=1.3)))
    fig.add_trace(go.Scatter(x=[0, omega_lim], y=[i_lim] * 2, mode="lines", name=f"current limit ({i_lim:.0f}A)", line=dict(color=INNER, dash="dot", width=1.3)))
    fig.add_trace(go.Scatter(x=omega, y=i_a, mode="lines+markers", name="trajectory", line=dict(color=HEALTHY, width=1.6), marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=[omega.iloc[-1]], y=[i_a.iloc[-1]], mode="markers", name="end", marker=dict(size=10, color=HEALTHY, symbol="circle", line=dict(width=1, color=SURFACE))))
    fig.update_xaxes(title_text="ω (rad/s)", range=[0, omega_lim])
    fig.update_yaxes(title_text="i_a (A)", range=[0, i_lim * 1.1])
    return _base_layout(fig, height=520)


def _pmsm_dq_figure(data, i_lim):
    fig = go.Figure()
    circ_x, circ_y = zip(*data["limit_circle"])
    fig.add_trace(go.Scatter(x=circ_x, y=circ_y, mode="lines", name=f"current limit ({i_lim:.0f}A)", line=dict(color=REF_GREY, dash="dot", width=1.3)))
    curve_x, curve_y = zip(*data["mtpa_curve"])
    fig.add_trace(go.Scatter(x=curve_x, y=curve_y, mode="lines", name="MTPA locus (analytic)", line=dict(color=ISOLINE_COLOR, width=2.2)))
    for key, label, color in [("mtpa", "FOC + MTPA", MTPA_COLOR), ("naive", "FOC naive (i_d=0)", NAIVE_COLOR)]:
        xs, ys = zip(*data[key])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", name=label, marker=dict(size=4, color=color, opacity=0.5)))
    fig.update_xaxes(title_text="i_d (A)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="i_q (A)")
    return _base_layout(fig, height=560)


def _slider_with_custom(container, label, min_value, max_value, value, step=None, format=None, help=None, key=None):
    """A slider for the common range, plus a "Custom value" checkbox that swaps in an unbounded
    number_input -- a slider alone can never go past max_value, and some parameters (e.g. an
    aggressive noise/severity level for stress-testing) legitimately need to. `container` is
    whatever sidebar card (st.container(border=True)) this control belongs to."""
    base_key = key or label
    slider_val = container.slider(label, min_value, max_value, value, step=step, format=format, help=help, key=f"{base_key}_slider")
    use_custom = container.checkbox("Custom value", key=f"{base_key}_custom", help=f'Type a "{label}" value outside {min_value}-{max_value} above.')
    if not use_custom:
        return slider_val
    # Preserve int vs. float: n_i_s/pmsm_subsample downstream (np.linspace's num=, range-like
    # loops) need a real int, not 18.0 -- inferred from the slider's own default `value` type.
    is_int = isinstance(value, int) and not isinstance(value, bool)
    custom_val = container.number_input(
        f"{label} (custom)",
        value=int(slider_val) if is_int else float(slider_val),
        step=step or (1 if is_int else 0.1),
        format=format,
        key=f"{base_key}_custom_input",
    )
    return int(custom_val) if is_int else custom_val


#: Saved motor-characteristic presets (both DC and PMSM), so repeated iteration doesn't mean
#: retyping r_a/l_a/psi_e (or p/l_d/l_q/r_s/psi_p) by hand every time -- a user's own local
#: experimentation state, not source-controlled calibration, so it's gitignored (see .gitignore).
_PROFILES_PATH = Path(__file__).resolve().parents[3] / "configs" / "motor_profiles.json"


def _load_profiles() -> dict:
    if _PROFILES_PATH.exists():
        try:
            return json.loads(_PROFILES_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_profiles(profiles: dict):
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=2, sort_keys=True))


#: Built-in fault types (Fault type selectbox always offers these -- "Custom fault types" below
#: only ever APPENDS to this list, never hides/replaces it, so it's always clear what's built-in
#: vs. user-added).
_BUILTIN_FAULT_TYPES = ["healthy", "outer_race", "inner_race", "ball", "cage"]

#: User-defined fault types (name -> {"order": float}), gitignored for the same reason as
#: motor_profiles.json above -- personal experimentation state, not source-controlled data.
_CUSTOM_FAULTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "custom_fault_types.json"


def _load_custom_faults() -> dict:
    if _CUSTOM_FAULTS_PATH.exists():
        try:
            return json.loads(_CUSTOM_FAULTS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_custom_faults(custom_faults: dict):
    _CUSTOM_FAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CUSTOM_FAULTS_PATH.write_text(json.dumps(custom_faults, indent=2, sort_keys=True))


def _resolve_builtin_order(fault_type: str) -> float:
    return bf.fault_order(fault_type, bf.KAT_DATACENTER_6203_GEOMETRY)


def _custom_fault_manager(container):
    """Renders the save/delete UI for user-defined fault types, right below the built-in Fault
    type selectbox. A saved custom fault type is a NAMED list of one or more characteristic
    orders (see Scenario.fault_order_override's docstring for what "order" means -- a multiple of
    shaft speed, not raw data or a trained model) -- pick one or more EXISTING fault types (built-
    in or already-saved custom) to merge, and/or type a manual order, then save under one new
    name. That name then behaves like any other single Fault type entry above: selecting it once
    injects every one of its component orders together, with no separate combining step at
    generate time (Scenario/runner.py already support N simultaneous orders per run, see
    BearingFaultLoad.extra_faults -- this just resolves a saved name into that list up front).
    """
    custom_faults = _load_custom_faults()

    header_col, help_col = container.columns([4, 1])
    header_col.markdown('<div class="df-card-header">Custom fault types</div>', unsafe_allow_html=True)
    with help_col.popover("❓", help="How do I pick an order?"):
        st.markdown(
            """
**Order** = how many times the fault event repeats per shaft revolution
(`fault_frequency_hz / shaft_frequency_hz`). That single number is all this needs -- no
geometry, dataset, or trained model.

| Fault | Order to use |
|---|---|
| Mass unbalance | `1.0` |
| Shaft misalignment | `2.0` (sometimes a 1.0 component too) |
| Gear-mesh fault | the gear's tooth count on that shaft (e.g. `24.0`) |
| A different bearing's known BPFO/BPFI/BSF | `known_fault_freq_hz / shaft_freq_hz` -- or give the bearing's geometry (ball count, diameters, contact angle) and it can be computed the same way the built-in types are |
| A peak seen in a real FFT spectrum | `peak_freq_hz / shaft_freq_hz`, both read at the same run's speed |

**Combining**: pick existing fault types below to merge their orders into one new named fault
type -- e.g. outer_race + a custom gear_mesh order together, saved as "outer_gear_combo". Select
it later like any other single fault type; every component order injects simultaneously.

Severity (amplitude) isn't set here -- use the **Fault severity** sliders above, same as the built-in types. Amplitude is always constant here (no load-zone modulation, unlike inner_race/ball).
            """
        )
    container.caption(
        "Save a new fault type: pick existing ones to combine and/or type a manual order (a multiple of shaft "
        "speed). Tap ❓ above for examples."
    )

    new_fault_name = container.text_input("Name", key="custom_fault_name", placeholder="e.g. outer_gear_combo")
    component_options = _BUILTIN_FAULT_TYPES[1:] + sorted(custom_faults)  # exclude "healthy"
    selected_components = container.multiselect(
        "Combine these (optional)",
        component_options,
        key="custom_fault_components",
        help="Existing fault types (built-in or already-saved custom) to merge into the new one -- each contributes its own order(s).",
    )
    col_toggle, col_order = container.columns([1, 2])
    add_manual = col_toggle.checkbox("+ manual order", key="custom_fault_add_manual", value=True, help="Also add a literal order you type in, on top of anything selected above.")
    new_fault_order = col_order.number_input("Manual order", key="custom_fault_order_input", min_value=0.01, value=3.5, step=0.1, format="%.2f", label_visibility="collapsed", disabled=not add_manual)

    name_conflicts_builtin = new_fault_name.strip() in _BUILTIN_FAULT_TYPES
    has_content = bool(selected_components) or add_manual
    if container.button(
        "💾 Save fault type",
        key="custom_fault_save",
        disabled=not new_fault_name.strip() or name_conflicts_builtin or not has_content,
        help=("That name is already a built-in fault type -- pick another." if name_conflicts_builtin else "Pick at least one component or add a manual order." if not has_content else "Save this fault type."),
    ):
        orders = []
        for comp in selected_components:
            orders.extend(custom_faults[comp]["orders"] if comp in custom_faults else [_resolve_builtin_order(comp)])
        if add_manual:
            orders.append(new_fault_order)
        custom_faults[new_fault_name.strip()] = dict(orders=orders)
        _save_custom_faults(custom_faults)
        st.toast(f"Saved '{new_fault_name.strip()}' -- {len(orders)} order(s): {', '.join(f'{o:.2f}' for o in orders)}.")
        st.rerun()

    if not custom_faults:
        container.caption("No custom fault types yet -- the built-ins above (healthy/outer_race/inner_race/ball/cage) are always available regardless.")
        return

    summary = ", ".join(f"{n} ({len(v['orders'])} order{'s' if len(v['orders']) != 1 else ''})" for n, v in sorted(custom_faults.items()))
    container.caption(f"Saved: {summary}")
    col_select, col_delete = container.columns([3, 1])
    selected = col_select.selectbox("Delete a custom fault type", sorted(custom_faults), key="custom_fault_delete_select", label_visibility="collapsed")
    if col_delete.button(
        "🗑️ Delete",
        key="custom_fault_delete",
        help=f"Delete '{selected}'. Any saved scenario history referencing it keeps working (its orders are baked into that history entry already) -- only future runs lose access to it, including as a component of other combos.",
    ):
        del custom_faults[selected]
        _save_custom_faults(custom_faults)
        st.rerun()


#: {group: {characteristic_name: number_input's key=}} -- the single source of truth for which
#: widget each saved field maps to, used both by _profile_manager (to know what "Save" reads) and
#: by _apply_pending_profile_load (to know what "Load" must write into session_state).
_PROFILE_FIELD_KEYS = {
    "dc": dict(r_a="dc_r_a", l_a="dc_l_a", psi_e="dc_psi_e"),
    "pmsm": dict(p="pmsm_p", l_d="pmsm_l_d", l_q="pmsm_l_q", r_s="pmsm_r_s", psi_p="pmsm_psi_p"),
}


def _apply_pending_profile_load():
    """Must be called BEFORE any motor-characteristic number_input is instantiated this script
    pass. Streamlit forbids writing session_state[key] for a widget that has already rendered in
    the same run ("cannot be modified after the widget with key ... is instantiated") -- but the
    "Load" button lives inside the expander, AFTER those widgets. So Load doesn't write
    session_state directly; it stashes (group, name) here and reruns, and this function (called
    near the top of the script, before tab1/tab3's widgets exist for this pass) applies it."""
    pending = st.session_state.pop("_pending_profile_load", None)
    if pending is None:
        return
    group, name = pending
    values = _load_profiles().get(group, {}).get(name)
    if values is None:
        return
    for field, value in values.items():
        key = _PROFILE_FIELD_KEYS.get(group, {}).get(field)
        if key is not None:
            st.session_state[key] = value


def _profile_manager(container, group: str, current_values: dict):
    """Renders a compact save/load/delete UI for named motor-characteristic presets, persisted to
    configs/motor_profiles.json (survives server restarts, unlike the in-memory undo history).

    Args:
        container: Where to render (an st.expander or similar) -- placed right below the
            characteristic number_inputs it saves/overwrites.
        group: "dc" or "pmsm" -- a separate namespace within the shared JSON file, and a key into
            _PROFILE_FIELD_KEYS.
        current_values: {characteristic_name: current_value}, e.g. {"r_a": r_a, "l_a": l_a, ...}
            -- what "Save" writes under the chosen name.
    """
    all_profiles = _load_profiles()
    group_profiles = all_profiles.setdefault(group, {})

    container.markdown('<div class="df-card-header">Motor profiles</div>', unsafe_allow_html=True)
    col_name, col_save = container.columns([3, 1])
    new_name = col_name.text_input("Profile name", key=f"{group}_profile_name", placeholder="e.g. worn-bearing spare motor", label_visibility="collapsed")
    if col_save.button("💾 Save", key=f"{group}_profile_save", disabled=not new_name.strip(), help="Save the current values above under this name."):
        group_profiles[new_name.strip()] = dict(current_values)
        _save_profiles(all_profiles)
        st.toast(f"Saved profile '{new_name.strip()}'.")
        st.rerun()

    if not group_profiles:
        container.caption("No saved profiles yet -- name one above and click Save.")
        return

    col_select, col_load, col_delete = container.columns([3, 1, 1])
    selected = col_select.selectbox("Load profile", sorted(group_profiles), key=f"{group}_profile_select", label_visibility="collapsed")
    if col_load.button("Load", key=f"{group}_profile_load", help=f"Overwrite the fields above with '{selected}'."):
        st.session_state["_pending_profile_load"] = (group, selected)
        st.rerun()
    if col_delete.button("🗑️", key=f"{group}_profile_delete", help=f"Delete '{selected}'."):
        del group_profiles[selected]
        _save_profiles(all_profiles)
        st.rerun()


_apply_pending_profile_load()


def _render_single_simulation():
    # ---------------------------------------------------------------------------
    # Sidebar: one scenario config for the DC-motor tabs (1 & 2, same underlying run), plus a
    # separate, independent config for the PMSM tab (3) -- different plant entirely. Motor physics
    # parameters are edited in each tab's own expander (below), not here, so they sit next to the
    # characteristics they change. Grouped into bordered "cards" (st.container(border=True)) by
    # logical role, matching an industrial-workbench control panel rather than one long flat list.
    # ---------------------------------------------------------------------------
    st.sidebar.markdown('<div class="df-sidebar-title">DC motor scenario</div><div class="df-sidebar-hint">Drives tabs 01-02</div>', unsafe_allow_html=True)

    _custom_faults = _load_custom_faults()

    card_ref = st.sidebar.container(border=True)
    _card_header(card_ref, "Reference & fault")
    fault_label = card_ref.selectbox(
        "Fault type",
        _BUILTIN_FAULT_TYPES + sorted(_custom_faults),
        help="Bearing fault to inject on both the electrical/MCSA path (current_r) and the mechanical/vibration path (acc_x/y/z). 'healthy' means no fault: acc_x/y/z are then pure background noise (see the Noise card below), independent of speed/torque by design -- see docs/patch3_mejora_modulo_B.md. Entries after 'cage' are custom fault types added below.",
    )
    fault_type = None if fault_label == "healthy" else fault_label
    # A custom fault type (built below) may itself be a saved COMBINATION of several characteristic
    # orders -- _custom_faults[name]["orders"] is always a list (length 1 for a plain single-order
    # custom fault). Split into (primary order, the rest as extra_faults) so this reuses the exact
    # same multi-fault machinery Scenario/runner.py already has (see BearingFaultLoad.extra_faults) --
    # selecting ONE saved name is enough to inject all of its component orders together.
    if fault_label in _custom_faults:
        _orders = _custom_faults[fault_label]["orders"]
        fault_order_override = _orders[0]
        extra_faults = [(fault_label, o) for o in _orders[1:]]
    else:
        fault_order_override = None
        extra_faults = []

    control_mode = card_ref.radio(
        "Control mode",
        ["Speed (ω_ref)", "Torque (τ_ref)"],
        horizontal=True,
        help=(
            "PICascadeController either tracks a speed setpoint (the original cascade: speed-outer / "
            "current-inner PI loops) or a torque setpoint directly (PICascadeController.control_torque() "
            "-- bypasses the outer speed loop, feeds the same inner current loop i_ref = τ_ref/ψ_e "
            "directly). Only one reference is active per run."
        ),
    )
    if control_mode == "Speed (ω_ref)":
        omega_ref = _slider_with_custom(
            card_ref,
            "Speed setpoint ω_ref (rad/s)",
            10.0,
            350.0,
            300.0,  # DcPermanentlyExcitedMotor's own nominal omega (GEM's real spec) -- settles to
            # current/torque within ~94% of GEM's nominal 97A/16Nm at duration=1.0s.
            help=(
                "The controller (PICascadeController) is a speed-outer / current-inner cascade: SPEED is "
                "the only externally commanded reference in this mode. Current and torque emerge from "
                "tracking it. Default (300 rad/s) is the motor's real nominal speed from GEM's own spec, "
                "not an arbitrary number."
            ),
        )
        torque_ref = None
        card_ref.caption("ω_ref=300 rad/s is the motor's real nominal speed (GEM spec); 1.0s is how long it takes to settle there.")
    else:
        torque_ref = _slider_with_custom(
            card_ref,
            "Torque setpoint τ_ref (Nm)",
            0.0,
            38.0,  # DcPermanentlyExcitedMotor's own torque LIMIT (GEM's real spec; nominal is 16.0Nm).
            16.0,
            help=(
                "The outer speed loop is bypassed entirely: i_ref = τ_ref/ψ_e feeds the inner current "
                "loop directly (same loop/gains as speed mode). ω is free-running here -- there is no "
                "speed reference to track, so it settles wherever the load lets it. Default (16 Nm) is "
                "the motor's real nominal torque from GEM's own spec."
            ),
        )
        omega_ref = 300.0  # unused by Scenario in this mode (torque_ref_nm takes over) -- kept as a
        # stable placeholder so _dc_config_now/history bookkeeping below has a value either way.
        card_ref.caption("τ_ref=16 Nm is the motor's real nominal torque (GEM spec); ω is free-running -- no speed loop active in this mode.")
    duration_s = _slider_with_custom(card_ref, "Duration (s)", 0.05, 2.0, 1.0, help="How long the scenario runs. The default settling time is ~1.0s for either reference above.")
    seed = card_ref.number_input("Seed", value=0, step=1, help="Seeds SCMLSystem's own randomness plus the white-noise generators below -- the same seed always reproduces the exact same run, including any noise added on top.")

    card_severity = st.sidebar.container(border=True)
    _card_header(card_severity, "Fault severity")
    electrical_severity = _slider_with_custom(card_severity, "Electrical severity (Nm, MCSA path)", 0.0, 20.0, 8.0, help="Torque-ripple amplitude (Nm) injected by BearingFaultLoad on the electrical/MCSA path -- a physical fault mechanism, only active when Fault type above isn't 'healthy'. Independent of Mechanical severity below (see Scenario's docstring for why).")
    default_mech = CALIBRATED_MECHANICAL_SEVERITY.get(fault_type, 0.0)
    mechanical_severity = (
        _slider_with_custom(
            card_severity,
            "Mechanical severity (Module B)",
            0.0,
            0.2,
            float(default_mech),
            format="%.3f",
            help="Module B's own fault-impulse amplitude (own units, not Nm). Defaults to the per-fault-type value calibrated against real Paderborn separability (AUC), not a round number.",
        )
        if fault_type is not None
        else None
    )
    if fault_type is None:
        card_severity.caption("Mechanical severity is hidden -- no fault selected above.")

    card_custom_fault = st.sidebar.container(border=True)
    _custom_fault_manager(card_custom_fault)

    card_noise = st.sidebar.container(border=True)
    _card_header(card_noise, "Noise")
    card_noise.caption("Generic white-noise added AFTER the simulation, on top of whatever the physics above produced -- distinct from the fault severities, which are physical excitation mechanisms.")
    electrical_noise_pct = _slider_with_custom(
        card_noise, "Electrical noise (%)", 0.0, 20.0, 0.0, help="Adds white Gaussian noise to current_r, sized as this percentage of current_r's own std dev over the run. 0% (default) = no noise, output matches the raw simulation exactly."
    )
    mechanical_noise_pct = _slider_with_custom(
        card_noise, "Vibration noise (%)", 0.0, 20.0, 0.0, help="Adds white Gaussian noise to acc_x/acc_y/acc_z independently, each sized as a percentage of that axis's own std dev over the run. 0% (default) = no noise."
    )

    st.sidebar.markdown('<div class="df-sidebar-title">PMSM FOC/MTPA scenario</div><div class="df-sidebar-hint">Drives tab 03 -- a different plant entirely</div>', unsafe_allow_html=True)
    card_pmsm = st.sidebar.container(border=True)
    _card_header(card_pmsm, "Point-cloud density")
    card_pmsm.caption("Salient PMSM, native FOC -- not part of the DC-motor scenario above. See control/classical/pmsm_foc.py.")
    n_i_s = _slider_with_custom(card_pmsm, "Current-magnitude steps", 3, 25, 18, help="More steps = more points along the MTPA curve. Trajectories converge fast (~30ms), so most of the point-cloud density comes from testing more current magnitudes, not longer runs.")
    pmsm_subsample = _slider_with_custom(card_pmsm, "Trajectory sample density", 1, 15, 3, help="Lower = more points kept per current step's transient (denser cloud, slightly slower to render).")

    # ---------------------------------------------------------------------------
    # Main area: exactly the 3 tabs from the artifact, restyled as a segmented control (CSS above)
    # with small technical icons prefixed onto each label.
    # ---------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["〰️ 01 Corrientes y vibración", "📈 02 Envolvente (ω, i_a) DC", "🧭 03 Plano i_d–i_q PMSM"])

    _DEFAULT_DC = DcPermanentlyExcitedMotor().motor_parameter
    _DEFAULT_DC_LIMITS = DcPermanentlyExcitedMotor().limits

    #: How many past configurations are kept for the "back/forward" comparison controls, per tab.
    #: Bounded on purpose (each entry holds a full result DataFrame/point-cloud) -- explicitly a
    #: memory/history-depth tradeoff, not a number that could grow unbounded.
    MAX_HISTORY = 4


    def _push_history(state_key: str, entry: dict):
        """Appends to a capped history list in session_state and points the "current" index at the
        newest entry. Used so the back/forward buttons below have something bounded to browse."""
        history = st.session_state.setdefault(state_key, [])
        history.append(entry)
        del history[:-MAX_HISTORY]
        st.session_state[f"{state_key}_idx"] = len(history) - 1


    def _history_nav(state_key: str, describe):
        """Renders '< Previous / config N of M / Next >' controls for a capped history list, and
        returns the currently-selected entry (or None if there's no history yet)."""
        history = st.session_state.get(state_key, [])
        if not history:
            return None
        idx = st.session_state.get(f"{state_key}_idx", len(history) - 1)
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        if col_prev.button("◀ Previous", key=f"{state_key}_prev", disabled=idx <= 0):
            st.session_state[f"{state_key}_idx"] = idx - 1
            st.rerun()
        if col_next.button("Next ▶", key=f"{state_key}_next", disabled=idx >= len(history) - 1):
            st.session_state[f"{state_key}_idx"] = idx + 1
            st.rerun()
        col_info.caption(f"Configuration {idx + 1}/{len(history)} -- {describe(history[idx])}")
        return history[idx]


    with tab1:
        with st.expander("Motor characteristics (DcPermanentlyExcitedMotor) -- editable", expanded=False):
            st.caption("Overrides GEM's motor_parameter for this run -- see sim/motors/dc_permanently_excited_motor.py for the stock defaults.")
            c1, c2, c3 = st.columns(3)
            r_a = c1.number_input(
                "Armature resistance r_a (Ω)",
                min_value=0.001,
                value=float(_DEFAULT_DC["r_a"]),
                format="%.4f",
                key="dc_r_a",
                help="PICascadeController's current-loop PI gains retune against r_a/l_a at construction (magnitude-optimum), so steady-state current tracking is nearly invariant to r_a alone -- see psi_e below for a change that does show up.",
            )
            l_a = c2.number_input(
                "Armature inductance l_a (H)",
                min_value=1e-6,
                value=float(_DEFAULT_DC["l_a"]),
                format="%.6f",
                key="dc_l_a",
                help="Also absorbed into the current-loop's self-tuning gains (same as r_a) -- mainly shapes loop bandwidth/transient response, not steady-state tracking.",
            )
            psi_e = c3.number_input(
                "Magnetic flux ψ_e (Wb)",
                min_value=0.01,
                value=float(_DEFAULT_DC["psi_e"]),
                format="%.4f",
                key="dc_psi_e",
                help="The torque constant -- unlike r_a/l_a, PICascadeController's current-loop gains don't retune against this, so changing it directly changes achieved torque/speed for the same current.",
            )
            _profile_manager(st, group="dc", current_values=dict(r_a=r_a, l_a=l_a, psi_e=psi_e))
        motor_overrides = dict(r_a=r_a, l_a=l_a, psi_e=psi_e)

        # History sync runs BEFORE the disabled-state computation below -- df_a/dc_config_used must
        # already reflect the (possibly just-navigated-to) current entry before we decide whether the
        # "Generate" button should be disabled, or the disabled check lags a full render pass behind
        # (it would still see the pre-sync/pre-push state) -- a real bug found by testing the actual
        # button state right after a click, not just the chart.
        def _describe_dc_entry(e):
            ref = f"τ_ref={e['torque_ref']:.1f}Nm" if e.get("torque_ref") is not None else f"ω_ref={e['omega_ref']:.0f}"
            noise = f", noise elec={e['elec_noise']:.0f}%/mech={e['mech_noise']:.0f}%" if e.get("elec_noise") or e.get("mech_noise") else ""
            return f"r_a={e['r_a']:.4f}Ω, l_a={e['l_a']:.6f}H, ψ_e={e['psi_e']:.4f}Wb, {ref}, {e['fault']}{noise}"

        _dc_entry = _history_nav("dc_history", _describe_dc_entry)
        if _dc_entry is not None:
            st.session_state["df_a"] = _dc_entry["df"]
            st.session_state["dc_params_used"] = _dc_entry["dc_params_used"]
            st.session_state["dc_config_used"] = _dc_entry["config"]

        # Computed BEFORE the button so its enabled/disabled state reflects whether anything
        # (sidebar scenario params OR the motor characteristics above) actually changed since the
        # last run -- disabled means "this exact config already produced what's showing below."
        _dc_config_now = (
            fault_type,
            fault_order_override,
            tuple(extra_faults),
            control_mode,
            omega_ref,
            torque_ref,
            duration_s,
            int(seed),
            electrical_severity,
            mechanical_severity,
            r_a,
            l_a,
            psi_e,
            electrical_noise_pct,
            mechanical_noise_pct,
        )
        _dc_has_result = "df_a" in st.session_state
        _dc_stale = _dc_has_result and st.session_state["dc_config_used"] != _dc_config_now
        _dc_button_disabled = _dc_has_result and not _dc_stale

        if st.sidebar.button("Generate DC scenario", type="primary", disabled=_dc_button_disabled, help="Disabled: nothing changed since the last run below." if _dc_button_disabled else None):
            with st.spinner("Simulating..."):
                scenario = Scenario(
                    scenario_id="dashboard_run",
                    fault_type=fault_type,
                    omega_ref_rad_s=omega_ref,
                    duration_s=duration_s,
                    seed=int(seed),
                    electrical_severity=electrical_severity,
                    mechanical_severity=mechanical_severity,
                    motor_parameter_overrides=motor_overrides,
                    electrical_noise_pct=electrical_noise_pct,
                    mechanical_noise_pct=mechanical_noise_pct,
                    torque_ref_nm=torque_ref,
                    fault_order_override=fault_order_override,
                    extra_faults=extra_faults,
                )
                records = run_scenario(scenario)
            df_new = pd.DataFrame.from_records(records)
            dc_params_used_new = dict(r_a=r_a, psi_e=psi_e, u_lim=float(_DEFAULT_DC_LIMITS["u"]), i_lim=float(_DEFAULT_DC_LIMITS["i"]), omega_lim=float(_DEFAULT_DC_LIMITS["omega"]))
            _push_history(
                "dc_history",
                {
                    "df": df_new,
                    "dc_params_used": dc_params_used_new,
                    "config": _dc_config_now,
                    "r_a": r_a,
                    "l_a": l_a,
                    "psi_e": psi_e,
                    "omega_ref": omega_ref,
                    "torque_ref": torque_ref,
                    "fault": fault_label,
                    "elec_noise": electrical_noise_pct,
                    "mech_noise": mechanical_noise_pct,
                },
            )
            st.session_state["dc_config_used"] = _dc_config_now
            st.rerun()  # immediately re-disable the button now that this config has been run

        if "df_a" in st.session_state:
            df = st.session_state["df_a"]
            if _dc_stale:
                st.warning("Sidebar/motor parameters changed since this result was generated -- click 'Generate DC scenario' to refresh (nothing re-runs automatically).")
            st.subheader(f"Result -- {df['label'].iloc[0]}")
            # Positional unpacking matching _dc_config_now's layout above -- only control_mode/
            # omega_ref/torque_ref (indices 3-5) are needed here, `*_` absorbs the rest so this
            # doesn't break if that tuple's shape changes elsewhere.
            _, _, _, _cm_used, _omega_ref_used, _torque_ref_used, *_ = st.session_state["dc_config_used"]
            if _cm_used == "Speed (ω_ref)":
                _elec_ref_cols = [None, _omega_ref_used * 60.0 / (2 * np.pi), None, None]
            else:
                _elec_ref_cols = [None, None, _torque_ref_used, None]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Electrical / mechanical")
                st.plotly_chart(_plotly_lines(df, [("current_r", "i_a (A)"), ("rpm", "rpm"), ("torque_nm", "torque (Nm)"), ("voltage_v", "u_a (V)")], ref_cols=_elec_ref_cols), width="stretch")
            with col2:
                st.markdown("##### Synthetic vibration (Module B)")
                st.plotly_chart(_plotly_lines(df, [("acc_x", "acc_x"), ("acc_y", "acc_y"), ("acc_z", "acc_z")]), width="stretch")
            if fault_type is not None or extra_faults:
                f = df.iloc[-1]
                st.caption(f"Fault frequencies: BPFO={f['bpfo_hz']:.1f}Hz  BPFI={f['bpfi_hz']:.1f}Hz  BSF={f['bsf_hz']:.1f}Hz  FTF={f['ftf_hz']:.1f}Hz")
            st.download_button("Download CSV", df.to_csv(index=False), file_name="driveflow_scenario_a.csv")
        else:
            _empty_state("Configure the DC motor scenario in the sidebar and click <b>Generate DC scenario</b>.")

    with tab2:
        if "df_a" in st.session_state:
            if _dc_stale:
                st.warning("Sidebar/motor parameters changed since this result was generated -- click 'Generate DC scenario' to refresh (nothing re-runs automatically).")
            st.markdown("##### Operating envelope (ω, i_a)")
            st.plotly_chart(_operating_envelope_figure(st.session_state["df_a"], st.session_state["dc_params_used"]), width="stretch")
        else:
            _empty_state("Configure the DC motor scenario in the sidebar and click <b>Generate DC scenario</b>.")

    with tab3:
        with st.expander("Motor characteristics (PermanentMagnetSynchronousMotor) -- editable", expanded=False):
            st.caption("Overrides pmsm_foc.py's MOTOR_PARAMS for this run -- see that file for GEM's stock defaults.")
            c1, c2, c3 = st.columns(3)
            p = c1.number_input("Pole pairs p", min_value=1, value=int(PMSM_DEFAULTS["p"]), step=1, key="pmsm_p", help="Scales both torque (τ = 1.5·p·(ψ_p + (l_d-l_q)·i_d)·i_q) and electrical frequency for a given mechanical speed.")
            l_d = c2.number_input("d-axis inductance l_d (H)", min_value=1e-6, value=float(PMSM_DEFAULTS["l_d"]), format="%.6f", key="pmsm_l_d", help="d-axis inductance -- together with l_q, sets how salient the motor is (see l_q's tooltip).")
            l_q = c3.number_input("q-axis inductance l_q (H)", min_value=1e-6, value=float(PMSM_DEFAULTS["l_q"]), format="%.6f", key="pmsm_l_q", help="MTPA only exists as a real curve (not just i_d=0) when l_d != l_q -- a salient motor.")
            c4, c5 = st.columns(2)
            r_s = c4.number_input(
                "Stator resistance r_s (Ω)",
                min_value=0.001,
                value=float(PMSM_DEFAULTS["r_s"]),
                format="%.4f",
                key="pmsm_r_s",
                help="DqCurrentController's PI gains retune against r_s/l_d/l_q at construction (magnitude-optimum), so steady-state dq tracking is largely invariant to r_s alone -- mainly affects transient shape and losses.",
            )
            psi_p = c5.number_input(
                "PM flux ψ_p (Wb)",
                min_value=0.001,
                value=float(PMSM_DEFAULTS["psi_p"]),
                format="%.4f",
                key="pmsm_psi_p",
                help="Permanent-magnet flux linkage -- the dominant torque-constant term; directly changes achieved torque for a given i_q, unlike r_s.",
            )
            _profile_manager(st, group="pmsm", current_values=dict(p=p, l_d=l_d, l_q=l_q, r_s=r_s, psi_p=psi_p))
        pmsm_mp = dict(p=p, l_d=l_d, l_q=l_q, r_s=r_s, psi_p=psi_p)

        # See the matching comment in tab1: history sync must run before the disabled-state
        # computation, or the button lags a full render pass behind the actual session_state.
        _pmsm_entry = _history_nav(
            "pmsm_history",
            lambda e: f"p={e['p']}, l_d={e['l_d']:.6f}H, l_q={e['l_q']:.6f}H, r_s={e['r_s']:.4f}Ω, ψ_p={e['psi_p']:.4f}Wb",
        )
        if _pmsm_entry is not None:
            st.session_state["pmsm_data"] = _pmsm_entry["data"]
            st.session_state["pmsm_config_used"] = _pmsm_entry["config"]

        _pmsm_config_now = (p, l_d, l_q, r_s, psi_p, n_i_s, pmsm_subsample)
        _pmsm_has_result = "pmsm_data" in st.session_state
        _pmsm_stale = _pmsm_has_result and st.session_state.get("pmsm_config_used") != _pmsm_config_now
        _pmsm_button_disabled = _pmsm_has_result and not _pmsm_stale

        if st.sidebar.button("Generate PMSM cloud", type="primary", disabled=_pmsm_button_disabled, help="Disabled: nothing changed since the last run below." if _pmsm_button_disabled else None):
            with st.spinner("Simulating current steps..."):
                i_s_values = list(np.linspace(40, 390, n_i_s))
                data_new = generate_mtpa_vs_naive_cloud(i_s_values=i_s_values, mp=pmsm_mp, subsample=pmsm_subsample)
            _push_history(
                "pmsm_history",
                {"data": data_new, "config": _pmsm_config_now, "p": p, "l_d": l_d, "l_q": l_q, "r_s": r_s, "psi_p": psi_p},
            )
            st.session_state["pmsm_config_used"] = _pmsm_config_now
            st.rerun()

        if "pmsm_data" in st.session_state:
            data = st.session_state["pmsm_data"]
            if _pmsm_stale:
                st.warning("Sidebar/motor parameters changed since this result was generated -- click 'Generate PMSM cloud' to refresh (nothing re-runs automatically).")
            st.markdown("##### Plano i_d-i_q (PMSM, FOC+MTPA vs. naive)")
            st.plotly_chart(_pmsm_dq_figure(data, i_lim=400.0), width="stretch")
            st.caption(
                "At equal current magnitude, MTPA delivers more torque than naive i_d=0 by exploiting the motor's "
                "reluctance (l_d≠l_q) -- see control/classical/pmsm_foc.py for the closed-form MTPA locus, verified "
                "against GEM's own PermanentMagnetSynchronousMotor._torque_limit()."
            )
        else:
            _empty_state("Configure the PMSM scenario in the sidebar and click <b>Generate PMSM cloud</b>.")


def _segment_to_scenario(seg: dict, motor_overrides: dict, seed: int, idx: int, custom_faults: dict) -> Scenario:
    """Resolves one queued Advanced Flow segment (a plain dict from st.session_state) into a
    Scenario -- same fault-name -> (fault_order_override, extra_faults) resolution
    _render_single_simulation uses for its own Fault type selectbox (see there for why: a custom
    fault type may itself be a saved combination of several orders)."""
    fault_label = seg["fault_label"]
    fault_type = None if fault_label == "healthy" else fault_label
    if fault_label in custom_faults:
        orders = custom_faults[fault_label]["orders"]
        fault_order_override = orders[0]
        extra_faults = [(fault_label, o) for o in orders[1:]]
    else:
        fault_order_override = None
        extra_faults = []

    kwargs = dict(
        scenario_id=f"flow_segment_{idx}",
        fault_type=fault_type,
        duration_s=seg["duration_s"],
        seed=seed,
        electrical_severity=seg["electrical_severity"],
        mechanical_severity=seg["mechanical_severity"],
        motor_parameter_overrides=motor_overrides,
        electrical_noise_pct=seg["electrical_noise_pct"],
        mechanical_noise_pct=seg["mechanical_noise_pct"],
        fault_order_override=fault_order_override,
        extra_faults=extra_faults,
    )
    if seg["control_mode"] == "Speed (ω_ref)":
        kwargs["omega_ref_rad_s"] = seg["setpoint"]
    else:
        kwargs["torque_ref_nm"] = seg["setpoint"]
    return Scenario(**kwargs)


def _describe_segment(seg: dict) -> str:
    ref = f"ω_ref={seg['setpoint']:.0f}rad/s" if seg["control_mode"] == "Speed (ω_ref)" else f"τ_ref={seg['setpoint']:.1f}Nm"
    return f"{seg['duration_s']:.2f}s @ {ref}, {seg['fault_label']}"


def _flow_figure(df, rows, ref_cols=None):
    """Same multi-row line layout as _plotly_lines, plus a dotted vertical marker at every
    segment boundary (skipping t=0, the flow's own start) so the sequence of operating states is
    visible directly on the chart, not just in the segment table below it."""
    fig = _plotly_lines(df, rows, ref_cols=ref_cols)
    boundaries = df.loc[df["segment_index"].diff() != 0, "timestamp_s"]
    for t in boundaries.iloc[1:]:
        fig.add_vline(x=float(t), line_width=1, line_dash="dot", line_color=REF_GREY, row="all", col="all")
    return fig


#: How many past flow RUNS (not segments) are kept for side-by-side comparison -- each holds a
#: full concatenated DataFrame across all its segments, so bounded for the same memory reason as
#: MAX_HISTORY above. Unlike MAX_HISTORY (bounded undo, one shown at a time), these are all shown
#: at once, stacked, so the cap is a little more generous.
MAX_FLOW_RUNS = 6


def _parse_float_list(raw: str) -> list:
    """"100, 200, 300.5" -> [100.0, 200.0, 300.5] -- silently skips tokens that aren't a number,
    so a trailing comma or stray space doesn't hard-fail the whole sequence builder."""
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _render_advanced_flow():
    st.sidebar.markdown('<div class="df-sidebar-title">Advanced Flow</div><div class="df-sidebar-hint">A sequence of operating states, run back-to-back</div>', unsafe_allow_html=True)
    st.sidebar.caption(
        "Each segment is its own independent simulation (fresh start, not a continuation of the previous "
        "segment's exact instantaneous state) -- 'flow' means a sequence of different operating states shown "
        "on one timeline, not one continuous physical trajectory threaded across fault/reference changes."
    )

    default_dc = DcPermanentlyExcitedMotor().motor_parameter
    default_dc_limits = DcPermanentlyExcitedMotor().limits
    custom_faults = _load_custom_faults()

    card_motor = st.sidebar.container(border=True)
    _card_header(card_motor, "Motor characteristics (shared across the whole flow)")
    c1, c2, c3 = card_motor.columns(3)
    r_a = c1.number_input("r_a (Ω)", min_value=0.001, value=float(default_dc["r_a"]), format="%.4f", key="flow_r_a")
    l_a = c2.number_input("l_a (H)", min_value=1e-6, value=float(default_dc["l_a"]), format="%.6f", key="flow_l_a")
    psi_e = c3.number_input("ψ_e (Wb)", min_value=0.01, value=float(default_dc["psi_e"]), format="%.4f", key="flow_psi_e")
    motor_overrides = dict(r_a=r_a, l_a=l_a, psi_e=psi_e)
    seed = card_motor.number_input("Seed (shared across segments)", value=0, step=1, key="flow_seed")

    card_add = st.sidebar.container(border=True)
    _card_header(card_add, "Add a segment")
    card_add.caption(
        "This is the 'control step' for one segment: pick Speed or Torque mode, then the single setpoint value "
        "that segment holds constant for its whole duration -- e.g. Speed mode + 300 rad/s means 'track 300 "
        "rad/s for Duration seconds.' Add several segments (below, or via the sequence builder further down) "
        "with different setpoints to see the transitions between them."
    )
    control_mode = card_add.radio("Control mode", ["Speed (ω_ref)", "Torque (τ_ref)"], horizontal=True, key="flow_add_control_mode")
    if control_mode == "Speed (ω_ref)":
        setpoint = _slider_with_custom(card_add, "Setpoint ω_ref (rad/s)", 10.0, 350.0, 300.0, key="flow_add_speed")
    else:
        setpoint = _slider_with_custom(card_add, "Setpoint τ_ref (Nm)", 0.0, 38.0, 16.0, key="flow_add_torque")
    duration_s = _slider_with_custom(card_add, "Duration (s)", 0.05, 2.0, 0.5, key="flow_add_duration")
    fault_label = card_add.selectbox("Fault type", _BUILTIN_FAULT_TYPES + sorted(custom_faults), key="flow_add_fault", help="Manage custom fault types from Single simulation mode -- they're shared across both modes.")
    electrical_severity = _slider_with_custom(card_add, "Electrical severity (Nm)", 0.0, 20.0, 8.0, key="flow_add_elec_sev")
    default_mech = CALIBRATED_MECHANICAL_SEVERITY.get(None if fault_label == "healthy" else fault_label, 0.0)
    mechanical_severity = _slider_with_custom(card_add, "Mechanical severity", 0.0, 0.2, float(default_mech), format="%.3f", key="flow_add_mech_sev")
    electrical_noise_pct = _slider_with_custom(card_add, "Electrical noise (%)", 0.0, 20.0, 0.0, key="flow_add_elec_noise")
    mechanical_noise_pct = _slider_with_custom(card_add, "Vibration noise (%)", 0.0, 20.0, 0.0, key="flow_add_mech_noise")

    def _base_segment_kwargs():
        return dict(
            control_mode=control_mode,
            setpoint=setpoint,
            duration_s=duration_s,
            fault_label=fault_label,
            electrical_severity=electrical_severity,
            mechanical_severity=mechanical_severity,
            electrical_noise_pct=electrical_noise_pct,
            mechanical_noise_pct=mechanical_noise_pct,
        )

    if card_add.button("+ Add segment", key="flow_add_button"):
        segments = st.session_state.setdefault("flow_segments", [])
        segments.append(_base_segment_kwargs())
        st.rerun()

    card_seq = st.sidebar.container(border=True)
    _card_header(card_seq, "Add a sequence (vary one field)")
    card_seq.caption(
        "Bulk-adds several segments at once, holding everything above constant except the one field you vary "
        "here -- e.g. step through healthy → outer_race → inner_race to see the transitions between fault "
        "types, or sweep several setpoints in a row, without refilling the form above each time."
    )
    vary_by = card_seq.radio(
        "Vary",
        ["Fault type", "Setpoint"],
        horizontal=True,
        key="flow_seq_vary",
        help=(
            "**Fault type**: one segment per fault you pick below, each reusing the SAME Control mode/Setpoint/"
            "Duration/severities/noise set in 'Add segment' above -- only fault_type changes between segments. "
            "Once added, open any segment below in the Queue to tweak its own setpoint if you want it to differ.\n\n"
            "**Setpoint**: one segment per number you type below, each reusing the CURRENT Control mode and "
            "Fault type/severities/noise set in 'Add segment' above -- only the setpoint (ω_ref or τ_ref, "
            "whichever Control mode is active) changes between segments."
        ),
    )
    if vary_by == "Fault type":
        seq_faults = card_seq.multiselect(
            "Fault types, in order",
            _BUILTIN_FAULT_TYPES + sorted(custom_faults),
            key="flow_seq_faults",
            help="One segment per selection, in the order you pick them -- each uses the Control mode/Setpoint/Duration/severities/noise set above (editable per-segment afterwards, in the Queue below).",
        )
        if card_seq.button("+ Add sequence", key="flow_seq_add_fault", disabled=len(seq_faults) < 2, help="Pick at least 2 fault types to build a sequence." if len(seq_faults) < 2 else None):
            segments = st.session_state.setdefault("flow_segments", [])
            for fl in seq_faults:
                seg = _base_segment_kwargs()
                seg["fault_label"] = fl
                segments.append(seg)
            st.rerun()
    else:
        seq_setpoints_raw = card_seq.text_input(
            "Setpoints, comma-separated",
            key="flow_seq_setpoints",
            placeholder="e.g. 100, 200, 300",
            help=f"One segment per value, in order -- each uses the current Control mode ({control_mode}) and the Duration/Fault type/severities/noise set above.",
        )
        seq_setpoints = _parse_float_list(seq_setpoints_raw)
        if card_seq.button("+ Add sequence", key="flow_seq_add_setpoint", disabled=len(seq_setpoints) < 2, help="Enter at least 2 comma-separated numbers to build a sequence." if len(seq_setpoints) < 2 else None):
            segments = st.session_state.setdefault("flow_segments", [])
            for sp in seq_setpoints:
                seg = _base_segment_kwargs()
                seg["setpoint"] = sp
                segments.append(seg)
            st.rerun()

    segments = st.session_state.get("flow_segments", [])
    card_queue = st.sidebar.container(border=True)
    _card_header(card_queue, f"Queue ({len(segments)} segment{'s' if len(segments) != 1 else ''})")
    if not segments:
        card_queue.caption("Empty -- add at least one segment above.")
    for i, seg in enumerate(segments):
        col_desc, col_up, col_down, col_del = card_queue.columns([5, 1, 1, 1])
        col_desc.caption(f"{i}. {_describe_segment(seg)}")
        if col_up.button("▲", key=f"flow_seg_up_{i}", disabled=i == 0, help="Move earlier"):
            segments[i - 1], segments[i] = segments[i], segments[i - 1]
            st.rerun()
        if col_down.button("▼", key=f"flow_seg_down_{i}", disabled=i == len(segments) - 1, help="Move later"):
            segments[i + 1], segments[i] = segments[i], segments[i + 1]
            st.rerun()
        if col_del.button("🗑️", key=f"flow_seg_del_{i}", help="Remove"):
            del segments[i]
            st.rerun()
        seg_expander = card_queue.expander(f"✏️ Edit segment {i}'s setpoint", expanded=False)
        # Keyed by id(seg), not the positional i -- i shifts under a segment when ▲/▼ reorders the
        # list (same dict object, new index), which would otherwise show a stale widget value left
        # over from whatever segment used to sit at that index.
        _seg_key = f"flow_seg_setpoint_{id(seg)}"
        if seg["control_mode"] == "Speed (ω_ref)":
            seg["setpoint"] = _slider_with_custom(seg_expander, "ω_ref (rad/s)", 10.0, 350.0, float(seg["setpoint"]), key=_seg_key)
        else:
            seg["setpoint"] = _slider_with_custom(seg_expander, "τ_ref (Nm)", 0.0, 38.0, float(seg["setpoint"]), key=_seg_key)

    col_run, col_clear = st.sidebar.columns(2)
    if col_run.button("▶ Run flow", key="flow_run_button", type="primary", disabled=not segments):
        with st.spinner(f"Simulating {len(segments)} segment(s)..."):
            scenarios = [_segment_to_scenario(seg, motor_overrides, int(seed), i, custom_faults) for i, seg in enumerate(segments)]
            records = run_flow(scenarios)
        runs = st.session_state.setdefault("flow_runs", [])
        run_id = st.session_state.get("_flow_run_counter", 0) + 1
        st.session_state["_flow_run_counter"] = run_id
        runs.append(
            dict(
                id=run_id,
                name=f"Simulation {run_id}",
                df=pd.DataFrame.from_records(records),
                dc_params_used=dict(r_a=r_a, psi_e=psi_e, u_lim=float(default_dc_limits["u"]), i_lim=float(default_dc_limits["i"]), omega_lim=float(default_dc_limits["omega"])),
                segments=[dict(s) for s in segments],
            )
        )
        del runs[: -MAX_FLOW_RUNS]
    if col_clear.button("Clear queue", key="flow_clear_button", disabled=not segments):
        st.session_state["flow_segments"] = []
        st.rerun()

    runs = st.session_state.get("flow_runs", [])
    if not runs:
        _empty_state("Queue one or more segments in the sidebar, then click <b>▶ Run flow</b>.")
        return

    st.caption(f"{len(runs)} simulation{'s' if len(runs) != 1 else ''} run so far (newest first, capped at {MAX_FLOW_RUNS}) -- rename any of them below, or remove the ones you don't need to compare anymore.")
    for entry in reversed(runs):
        st.divider()
        name_key = f"flow_run_name_{entry['id']}"
        name = st.text_input("Simulation name", value=entry["name"], key=name_key, label_visibility="collapsed")
        entry["name"] = name
        df = entry["df"]
        st.subheader(name)
        st.caption(
            f"{len(entry['segments'])} segment{'s' if len(entry['segments']) != 1 else ''}, {df['timestamp_s'].iloc[-1]:.2f}s total -- "
            + " → ".join(f"[{i}] {_describe_segment(s)}" for i, s in enumerate(entry["segments"]))
        )
        # Per-segment reference arrays -- NaN outside a segment controlled that way, so the dashed
        # line only appears while that reference was actually active (see _plotly_lines' docstring).
        _omega_ref_arr = np.full(len(df), np.nan)
        _torque_ref_arr = np.full(len(df), np.nan)
        for _i, _seg in enumerate(entry["segments"]):
            _mask = (df["segment_index"] == _i).to_numpy()
            if _seg["control_mode"] == "Speed (ω_ref)":
                _omega_ref_arr[_mask] = _seg["setpoint"] * 60.0 / (2 * np.pi)
            else:
                _torque_ref_arr[_mask] = _seg["setpoint"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Electrical / mechanical")
            st.plotly_chart(_flow_figure(df, [("current_r", "i_a (A)"), ("rpm", "rpm"), ("torque_nm", "torque (Nm)"), ("voltage_v", "u_a (V)")], ref_cols=[None, _omega_ref_arr, _torque_ref_arr, None]), width="stretch", key=f"flow_chart_elec_{entry['id']}")
        with col2:
            st.markdown("##### Synthetic vibration (Module B)")
            st.plotly_chart(_flow_figure(df, [("acc_x", "acc_x"), ("acc_y", "acc_y"), ("acc_z", "acc_z")]), width="stretch", key=f"flow_chart_vib_{entry['id']}")
        st.caption("Dotted vertical lines mark segment boundaries.")
        col_dl, col_rm = st.columns([1, 5])
        col_dl.download_button("Download CSV", df.to_csv(index=False), file_name=f"driveflow_flow_{entry['id']}.csv", key=f"flow_dl_{entry['id']}")
        if col_rm.button("🗑️ Remove this simulation", key=f"flow_remove_{entry['id']}"):
            runs.remove(entry)
            st.rerun()


def _dpc_figure(df):
    """Complex voltage plane: the DPC-commanded converter voltage vc (real, imag) traced over the
    run, against the rotating reference vref it's supposed to track -- the VSC/DPC equivalent of
    Fase A's operating-envelope plot (tab 02), same idea (trajectory vs. a reference curve),
    different plant/units entirely."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["v_ref_real"], y=df["v_ref_imag"], mode="lines", name="v_ref (commanded)", line=dict(color=REF_GREY, dash="dot", width=1.3)))
    fig.add_trace(go.Scatter(x=df["vc_real"], y=df["vc_imag"], mode="lines", name="v_c (actual)", line=dict(color=HEALTHY, width=1.6)))
    fig.add_trace(go.Scatter(x=[df["vc_real"].iloc[-1]], y=[df["vc_imag"].iloc[-1]], mode="markers", name="end", marker=dict(size=10, color=HEALTHY, symbol="circle", line=dict(width=1, color=SURFACE))))
    fig.update_xaxes(title_text="v_real (V)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="v_imag (V)")
    return _base_layout(fig, height=520)


#: horizon step (1-4, since step 0 IS vref_alpha/vref_beta) each auto-fill column corresponds to.
_DPC_AUTOFILL_STEP = {"vref_alphaph": 1, "vref_betaph": 1, "vref_alphaph3": 2, "vref_betaph3": 2,
                       "vref_alphaph4": 3, "vref_betaph4": 3, "vref_alphaph5": 4, "vref_betaph5": 4}

#: (column name, physical meaning, units, requirement) -- documents DPC_COLUMNS for the
#: upload-format table below.
_DPC_COLUMN_SPEC = [
    ("if_alpha", "Filter current, real (α) axis -- measured now", "A", "Required"),
    ("if_beta", "Filter current, imag (β) axis -- measured now", "A", "Required"),
    ("vc_alpha", "Capacitor voltage, real (α) axis -- measured now", "V", "Required"),
    ("vc_beta", "Capacitor voltage, imag (β) axis -- measured now", "V", "Required"),
    ("vref_alpha", "Reference voltage, real axis -- horizon step 1 (now)", "V", "Required"),
    ("vref_beta", "Reference voltage, imag axis -- horizon step 1 (now)", "V", "Required"),
    ("r", "Load resistance -- measured now", "Ω", "Required"),
    ("vref_alphaph", "Reference voltage, real axis -- horizon step 2", "V", "Optional -- auto-filled"),
    ("vref_betaph", "Reference voltage, imag axis -- horizon step 2", "V", "Optional -- auto-filled"),
    ("vref_alphaph3", "Reference voltage, real axis -- horizon step 3", "V", "Optional -- auto-filled"),
    ("vref_betaph3", "Reference voltage, imag axis -- horizon step 3", "V", "Optional -- auto-filled"),
    ("vref_alphaph4", "Reference voltage, real axis -- horizon step 4", "V", "Optional -- auto-filled"),
    ("vref_betaph4", "Reference voltage, imag axis -- horizon step 4", "V", "Optional -- auto-filled"),
    ("vref_alphaph5", "Reference voltage, real axis -- horizon step 5", "V", "Optional -- auto-filled"),
    ("vref_betaph5", "Reference voltage, imag axis -- horizon step 5", "V", "Optional -- auto-filled"),
]


def _dpc_format_table_styler(spec_df: pd.DataFrame):
    """Colors each row by requirement -- green tint (MTPA_COLOR) for Required, orange tint
    (ISOLINE_COLOR) for Optional -- reusing the app's existing semantic-ish palette instead of
    inventing new hexes just for this table."""
    def _row_style(row):
        tint = MTPA_COLOR if row["requirement"] == "Required" else ISOLINE_COLOR
        return [f"background-color: {tint}26; color: {INK}"] * len(row)
    return spec_df.style.apply(_row_style, axis=1)


def _autofill_horizon_columns(df_up: pd.DataFrame, missing_cols: list[str], omega_rad_s: float) -> pd.DataFrame:
    """Fills the given (missing) horizon-reference columns by extrapolating each row's own
    vref_alpha/vref_beta forward under a rotating-reference assumption -- the exact same
    magnitude*cos/sin(phase0 + omega*tau*k) formula RotatingReference.at_step uses, vectorized
    over rows instead of called once per step. Only as good as that assumption: a real reference
    that isn't a constant-magnitude rotation at `omega_rad_s` will make these steps wrong even
    though steps actually present in the upload are untouched."""
    magnitude = np.sqrt(df_up["vref_alpha"] ** 2 + df_up["vref_beta"] ** 2)
    phase0 = np.arctan2(df_up["vref_beta"], df_up["vref_alpha"])
    df_up = df_up.copy()
    for col in missing_cols:
        k = _DPC_AUTOFILL_STEP[col]
        theta = phase0 + omega_rad_s * _TAU_S * k
        df_up[col] = magnitude * (np.cos(theta) if col.startswith("vref_alpha") else np.sin(theta))
    return df_up


@st.cache_resource
def _load_dpc_model():
    model = build_dpc_network()
    model(np.zeros((1, len(DPC_COLUMNS)), dtype=np.float32))  # build before loading weights
    model.load_weights(_DPC_WEIGHTS_PATH)
    return model


def _dpc_upload_template_df(n_rows: int = 3) -> pd.DataFrame:
    """A genuine, physically-consistent example -- generated from the real RotatingReference class
    at i_f=v_c=0 and the training-default R/magnitude/frequency, not hand-typed placeholder
    numbers. Rows are consecutive steps of one made-up run purely so the reference columns show
    a real rotating pattern; an uploaded dataset does NOT need to be a time series -- each row is
    evaluated independently (see _render_dpc_upload_eval's caption)."""
    reference = RotatingReference(tau=_TAU_S)
    rows = []
    for k in range(n_rows):
        refs = reference.horizon(k, horizon=DPC_HORIZON)
        rows.append(
            {
                "if_alpha": 0.0, "if_beta": 0.0, "vc_alpha": 0.0, "vc_beta": 0.0,
                "vref_alpha": refs[0][0], "vref_beta": refs[0][1], "r": float(_VSC_R_OHM),
                "vref_alphaph": refs[1][0], "vref_betaph": refs[1][1],
                "vref_alphaph3": refs[2][0], "vref_betaph3": refs[2][1],
                "vref_alphaph4": refs[3][0], "vref_betaph4": refs[3][1],
                "vref_alphaph5": refs[4][0], "vref_betaph5": refs[4][1],
            }
        )
    return pd.DataFrame(rows, columns=DPC_COLUMNS)


def _render_dpc_upload_eval():
    st.markdown("##### Evaluate the trained DPC network on your own dataset")
    st.caption(
        "Open-loop evaluation, same method as experiments/evaluate_dpc.py: each row is one "
        "independent (measured state, 5-step reference horizon) snapshot -- NOT a time series, "
        "rows don't need to be consecutive steps of a real run. For each row the network predicts "
        "a 5-step voltage command and this simulates the identified plant forward from that row's "
        "own state to score it -- exactly how Data4train.mat's own holdout split is scored."
    )

    with st.expander(f"Required format -- {len(DPC_COLUMNS)} columns ({len(DPC_REQUIRED_COLUMNS)} required + {len(DPC_AUTOFILL_COLUMNS)} optional)", expanded=False):
        st.dataframe(
            _dpc_format_table_styler(pd.DataFrame(_DPC_COLUMN_SPEC, columns=["column", "meaning", "units", "requirement"])),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"**Minimum to run at all: the {len(DPC_REQUIRED_COLUMNS)} green 'Required' columns** -- the "
            f"current measured state (i_f, v_c), R, and the reference's current value. The "
            f"{len(DPC_AUTOFILL_COLUMNS)} orange 'Optional' columns are the reference at horizon steps "
            "2-5 -- supply them for an exact evaluation, or omit them and they'll be auto-filled after "
            "upload by extrapolating vref_alpha/vref_beta forward under a rotating-reference assumption "
            "(you'll get to confirm the assumed frequency)."
        )
        st.caption(
            "CSV: a header row with these column names (any column order is fine -- matched by name, not "
            "position). JSON: either a list of objects with these keys, or an object of equal-length arrays "
            "keyed by these names. Extra columns are ignored."
        )
        template_df = _dpc_upload_template_df()
        col_csv, col_json, col_min = st.columns(3)
        col_csv.download_button(
            "Download full template (CSV)", template_df.to_csv(index=False),
            file_name="dpc_dataset_template.csv", key="dpc_template_csv",
        )
        col_json.download_button(
            "Download full template (JSON)", template_df.to_json(orient="records", indent=2),
            file_name="dpc_dataset_template.json", key="dpc_template_json",
        )
        col_min.download_button(
            "Download minimal template (CSV, 7 cols)", template_df[DPC_REQUIRED_COLUMNS].to_csv(index=False),
            file_name="dpc_dataset_template_minimal.csv", key="dpc_template_minimal_csv",
            help="Only the required columns -- the rest get auto-filled after upload.",
        )

    uploaded = st.file_uploader("Upload dataset", type=["csv", "json"], key="dpc_upload")
    if uploaded is None:
        return

    try:
        if uploaded.name.lower().endswith(".json"):
            df_up = pd.DataFrame(json.load(uploaded))
        else:
            df_up = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse '{uploaded.name}' as {'JSON' if uploaded.name.lower().endswith('.json') else 'CSV'}: {e}")
        return

    df_clean, missing_horizon_cols, messages = validate_dpc_upload(df_up)
    for level, text in messages:
        getattr(st, level)(text)
    if df_clean is None:
        return
    df_up = df_clean

    MAX_ROWS = 5000
    if len(df_up) > MAX_ROWS:
        st.warning(f"File has {len(df_up)} rows -- evaluating the first {MAX_ROWS} only.")
        df_up = df_up.iloc[:MAX_ROWS]

    if missing_horizon_cols:
        st.info(
            f"{len(missing_horizon_cols)} of the {len(DPC_AUTOFILL_COLUMNS)} optional horizon column(s) "
            f"weren't in your upload ({', '.join(missing_horizon_cols)}) -- auto-filling them by "
            "extrapolating each row's own vref_alpha/vref_beta forward assuming a rotating reference at "
            "the frequency below. For an exact evaluation, supply the real values instead."
        )
        assumed_omega = _slider_with_custom(
            st, "Assumed reference frequency ω (rad/s)", 50.0, 700.0, float(GRID_OMEGA_RAD_S),
            format="%.2f", key="dpc_upload_assumed_omega",
        )
        df_up = _autofill_horizon_columns(df_up, missing_horizon_cols, assumed_omega)

    df_up = df_up[DPC_COLUMNS]
    x_batch = df_up.to_numpy(dtype=np.float32)
    model = _load_dpc_model()
    with st.spinner(f"Running the trained DPC network on {len(df_up)} row(s)..."):
        v_o = model(x_batch, training=False)
        sim = simulate_horizon(x_batch, v_o)
    v_ref_real, v_ref_imag = sim["v_ref_real"].numpy(), sim["v_ref_imag"].numpy()
    vc_real, vc_imag = sim["vc_real"].numpy(), sim["vc_imag"].numpy()
    err = np.sqrt((v_ref_real - vc_real) ** 2 + (v_ref_imag - vc_imag) ** 2)  # (horizon, n_rows)

    r_mean, mag_mean = float(df_up["r"].mean()), float(np.sqrt(df_up["vref_alpha"] ** 2 + df_up["vref_beta"] ** 2).mean())
    on_distribution = abs(r_mean - float(_VSC_R_OHM)) < 0.5 and abs(mag_mean - float(REFERENCE_MAGNITUDE_V)) < 2.5
    st.caption(
        f"{len(df_up)} row(s) scored -- mean R={r_mean:.3f}Ω, mean |v_ref|={mag_mean:.1f}V "
        f"{'(close to training defaults)' if on_distribution else '(⚠ off-distribution vs. training defaults R≈8.01Ω, |v_ref|=50.0V -- expect worse tracking)'}"
    )

    tol_v = 0.05 * mag_mean if mag_mean > 1e-6 else 2.5
    success_rate = 100.0 * float(np.mean(err < tol_v))
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall RMSE", f"{float(np.sqrt(np.mean(err ** 2))):.2f} V")
    m2.metric("Overall MAE", f"{float(np.mean(err)):.2f} V")
    m3.metric(f"Success rate (<{tol_v:.1f}V)", f"{success_rate:.1f}%")

    rmse_per_step = np.sqrt(np.mean(err ** 2, axis=1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"step {k + 1}" for k in range(DPC_HORIZON)], y=rmse_per_step, marker_color=HEALTHY))
    fig.update_yaxes(title_text="RMSE (V)")
    st.plotly_chart(_base_layout(fig, height=340), width="stretch")
    st.caption("RMSE per horizon step -- later steps predict further into the future from the same measured row, so they're usually noisier.")

    pred_rows = [
        {
            "row": i, "horizon_step": k + 1,
            "v_ref_real": float(v_ref_real[k, i]), "v_ref_imag": float(v_ref_imag[k, i]),
            "vc_pred_real": float(vc_real[k, i]), "vc_pred_imag": float(vc_imag[k, i]),
            "error_v": float(err[k, i]),
        }
        for i in range(len(df_up))
        for k in range(DPC_HORIZON)
    ]
    st.download_button(
        "Download predictions (CSV)", pd.DataFrame(pred_rows).to_csv(index=False),
        file_name="dpc_upload_predictions.csv", key="dpc_pred_download",
    )


def _render_fase_b():
    st.sidebar.markdown('<div class="df-sidebar-title">DPC scenario</div><div class="df-sidebar-hint">Voltage Source Converter, trained DPC network</div>', unsafe_allow_html=True)
    card_ref = st.sidebar.container(border=True)
    _card_header(card_ref, "Reference")
    card_ref.caption(
        "This plant/controller pair is deterministic (no simulated randomness to seed) -- Seed instead picks "
        "the rotating reference's starting phase, so different seeds still give distinguishable, reproducible "
        "runs. No fault model or motor characteristics exist for this plant (no motor at all -- a VSC is power "
        "electronics, see 'About this platform') -- but the reference's own magnitude/frequency below are real, "
        "editable parameters."
    )
    duration_s = _slider_with_custom(card_ref, "Duration (s)", 0.02, 1.0, 0.1, key="dpc_duration")
    seed = card_ref.number_input("Seed (reference start phase)", value=0, step=1, key="dpc_seed")
    card_ref.caption(
        "Magnitude and frequency below default to Data4train.mat's own values (verified on its holdout split) "
        "-- same off-distribution caveat as Load resistance below: the network has never seen anything else."
    )
    reference_magnitude_v = _slider_with_custom(card_ref, "Reference magnitude |v_ref| (V)", 10.0, 150.0, float(REFERENCE_MAGNITUDE_V), key="dpc_magnitude")
    reference_omega_rad_s = _slider_with_custom(card_ref, "Reference frequency ω (rad/s)", 50.0, 700.0, float(GRID_OMEGA_RAD_S), format="%.2f", key="dpc_omega")

    card_electrical = st.sidebar.container(border=True)
    _card_header(card_electrical, "Electrical")
    card_electrical.caption(
        "Load resistance R is the one real physical parameter this plant has -- it's fed BOTH to the simulated "
        "converter (i_load = v_c/R) and directly into the trained DPC network's own inputs (see "
        "control/dpc/controller.py). The default below is the constant value found in every row of the "
        "network's own training data (Data4train.mat) -- it has never seen any other R. Moving this slider is "
        "a genuine robustness probe (does tracking degrade when the real load doesn't match training?), not a "
        f"validated operating point. The slider floors at {MIN_STABLE_LOAD_RESISTANCE_OHM:.1f}Ω because below "
        "~3.37Ω the plant itself (Adf/Bdf's i_load=v_c/R feedback) is open-loop unstable -- no controller, "
        "trained on this or any other data, changes that (see sim/vsc_system.py's load_feedback_spectral_radius "
        "and tests/test_dpc_robustness_grid.py)."
    )
    load_resistance_ohm = _slider_with_custom(
        card_electrical, "Load resistance R (Ω)", MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, float(_VSC_R_OHM), format="%.4f", key="dpc_r_ohm"
    )
    if load_resistance_ohm < MIN_STABLE_LOAD_RESISTANCE_OHM:
        card_electrical.warning(
            f"R = {load_resistance_ohm:.4f}Ω is below the plant's open-loop stability floor (~3.37Ω, see caption "
            "above) -- the closed loop is expected to diverge to NaN within ~2000 steps regardless of the "
            "controller. Kept reachable via 'Custom value' for reproducing this known finding, not as a "
            "supported operating point."
        )

    dpc_config_now = (duration_s, int(seed), load_resistance_ohm, reference_magnitude_v, reference_omega_rad_s)
    dpc_has_result = "df_b" in st.session_state
    dpc_stale = dpc_has_result and st.session_state.get("dpc_config_used") != dpc_config_now
    dpc_button_disabled = dpc_has_result and not dpc_stale

    if st.sidebar.button("Generate DPC scenario", type="primary", disabled=dpc_button_disabled, help="Disabled: nothing changed since the last run below." if dpc_button_disabled else None):
        with st.spinner("Simulating..."):
            scenario = Scenario(
                scenario_id="dashboard_dpc_run",
                controller_type="DPC",
                plant_config_id="vsc_dpc_v1",
                duration_s=duration_s,
                seed=int(seed),
                load_resistance_ohm=load_resistance_ohm,
                reference_magnitude_v=reference_magnitude_v,
                reference_omega_rad_s=reference_omega_rad_s,
            )
            records = run_scenario(scenario)
        st.session_state["df_b"] = pd.DataFrame.from_records(records)
        st.session_state["dpc_config_used"] = dpc_config_now
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["〰️ 01 Tracking (t)", "🧭 02 Complex voltage plane", "📤 03 Evaluate your dataset"])

    with tab3:
        _render_dpc_upload_eval()

    if "df_b" not in st.session_state:
        with tab1:
            _empty_state("Configure the DPC scenario in the sidebar and click <b>Generate DPC scenario</b>.")
        with tab2:
            _empty_state("Configure the DPC scenario in the sidebar and click <b>Generate DPC scenario</b>.")
        return

    df = st.session_state["df_b"]
    rmse = float(np.sqrt(np.mean((df["vc_real"] - df["v_ref_real"]) ** 2 + (df["vc_imag"] - df["v_ref_imag"]) ** 2)))

    with tab1:
        if dpc_stale:
            st.warning("Sidebar parameters changed since this result was generated -- click 'Generate DPC scenario' to refresh (nothing re-runs automatically).")
        st.subheader("Result -- DPC tracking")
        _, _, _r_used, _mag_used, _omega_used = st.session_state["dpc_config_used"]
        _on_distribution = abs(_r_used - float(_VSC_R_OHM)) < 1e-9 and abs(_mag_used - float(REFERENCE_MAGNITUDE_V)) < 1e-9 and abs(_omega_used - float(GRID_OMEGA_RAD_S)) < 1e-9
        _note = " (training defaults)" if _on_distribution else " (⚠ off-distribution -- network never trained on this combination)"
        st.caption(f"Closed-loop RMSE: {rmse:.3f} V over {df['timestamp_s'].iloc[-1]:.3f}s -- R={_r_used:.4f}Ω, |v_ref|={_mag_used:.1f}V, ω={_omega_used:.2f}rad/s{_note}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Voltage (real/imag) -- reference vs. actual")
            st.plotly_chart(_plotly_lines(df, [("vc_real", "v_real (V)"), ("vc_imag", "v_imag (V)")], ref_cols=["v_ref_real", "v_ref_imag"]), width="stretch")
        with col2:
            st.markdown("##### Filter current (real/imag)")
            st.plotly_chart(_plotly_lines(df, [("i_f_real", "i_f_real (A)"), ("i_f_imag", "i_f_imag (A)")]), width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="driveflow_scenario_b.csv")

    with tab2:
        if dpc_stale:
            st.warning("Sidebar parameters changed since this result was generated -- click 'Generate DPC scenario' to refresh (nothing re-runs automatically).")
        st.markdown("##### Complex voltage plane (v_c vs. v_ref)")
        st.plotly_chart(_dpc_figure(df), width="stretch")
        st.caption("The commanded voltage v_c should trace the same rotating path as v_ref -- deviation is closed-loop tracking error, summarized above as RMSE.")


if _selected_phase == "A":
    # Top-level mode switch within Fase A: "Single simulation" is everything this app had before
    # Advanced Flow existed (one operating state per Generate click); "Advanced Flow" runs an
    # ordered SEQUENCE of operating states back-to-back (datagen/runner.py's run_flow()) -- kept
    # as a separate mode, not merged into the single-sim UI, specifically so the common case
    # doesn't get more cluttered with segment-sequencing controls most sessions won't touch.
    app_mode = st.sidebar.radio("Mode", ["Single simulation", "Advanced Flow"], horizontal=True, key="app_mode")
    st.sidebar.divider()
    if app_mode == "Single simulation":
        _render_single_simulation()
    else:
        _render_advanced_flow()
elif _selected_phase == "B":
    _render_fase_b()
elif _selected_phase == "IA":
    _render_fase_ia()
else:
    _render_fase_tl()
