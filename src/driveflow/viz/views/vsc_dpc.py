"""VSC / DPC page (legacy Fase B): a Voltage Source Converter driven by the trained Direct Power
Control network, tracking a rotating voltage reference -- no motor, no bearing, no fault model.
Tabs: tracking over time, complex voltage plane, and open-loop evaluation of the trained network on
an uploaded dataset (same method as experiments/evaluate_dpc.py)."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.control.dpc import COLUMNS as DPC_COLUMNS
from driveflow.control.dpc import HORIZON as DPC_HORIZON
from driveflow.control.dpc import build_dpc_network, simulate_horizon
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V, RotatingReference
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH, _VSC_R_OHM
from driveflow.sim.vsc_params import UPSTREAM_CONVERTER_LIMIT_V, UPSTREAM_DC_LINK_V
from driveflow.datagen.runner import TAU as _TAU_S
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM
from driveflow.viz.adapters.session import config_hash, record_run
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.metric_cards import kpi_row
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.components.signal_chart import time_series_figure
from driveflow.viz.dpc_upload_validation import DPC_AUTOFILL_COLUMNS, DPC_REQUIRED_COLUMNS, validate_dpc_upload
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure
from driveflow.viz.theme.tokens import DARK

#: Upload size limit (AUDIT AI-12) -- rows beyond MAX_ROWS are not evaluated anyway.
MAX_UPLOAD_MB = 50

def _dpc_figure(df):
    """Complex voltage plane: the DPC-commanded converter voltage vc (real, imag) traced over the
    run, against the rotating reference vref it's supposed to track -- the VSC/DPC equivalent of
    Fase A's operating-envelope plot (tab 02), same idea (trajectory vs. a reference curve),
    different plant/units entirely."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["v_ref_real"], y=df["v_ref_imag"], mode="lines", name="v_ref (commanded)", line=dict(color=series_color("reference"), dash="dot", width=1.3)))
    fig.add_trace(go.Scatter(x=df["vc_real"], y=df["vc_imag"], mode="lines", name="v_c (actual)", line=dict(color=series_color("electrical"), width=1.6)))
    fig.add_trace(go.Scatter(x=[df["vc_real"].iloc[-1]], y=[df["vc_imag"].iloc[-1]], mode="markers", name="end", marker=dict(size=10, color=series_color("electrical"), symbol="circle")))
    style_figure(fig, height=520, x_title="v_real (V)", y_title="v_imag (V)")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


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
    """Tints each row by requirement (healthy token = Required, warning token = Optional) and
    leaves the text colour to the active theme."""
    def _row_style(row):
        tint = DARK["healthy"] if row["requirement"] == "Required" else DARK["warning"]
        return [f"background-color: {tint}26"] * len(row)
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


#: Steps of start-up transient (from rest) left out of the converter-voltage check.
_SETTLE_STEPS = 50


def _converter_limit_notice(df: pd.DataFrame, p: dict) -> None:
    """S-09: say when the run needed more converter voltage than the original hardware can give."""
    if "v_o_real" not in df:
        return
    settled = df.iloc[_SETTLE_STEPS:]
    peak = float(np.max(np.abs(settled[["v_o_real", "v_o_imag"]].to_numpy()))) if len(settled) else 0.0
    if p.get("vsc_voltage_limit_v") is not None:
        share = float(settled["v_o_saturated"].mean()) if len(settled) else 0.0
        msg = f"Converter voltage limited to ±{p['vsc_voltage_limit_v']:.0f} V per axis: the limit clipped the command in {share:.0%} of the settled steps."
        (st.warning if share > 0 else st.caption)(msg)
    elif peak > UPSTREAM_CONVERTER_LIMIT_V:
        st.warning(f"This run needed up to {peak:.0f} V per axis of converter voltage once settled. The original converter (V_dc = {UPSTREAM_DC_LINK_V:.0f} V) clamps at ±{UPSTREAM_CONVERTER_LIMIT_V:.0f} V, so it could not track this reference; enable the voltage limit to see that behavior (S-09).", icon=":material/bolt:")


@st.cache_resource(show_spinner=False)
def _load_dpc_model():
    # Inference only -- this network is never trained from the dashboard, so a shared cached
    # instance is safe (contrast with transfer learning, AUDIT AI-04).
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

    uploaded = st.file_uploader("Upload dataset", type=["csv", "json"], key="dpc_upload", help=f"CSV or JSON, at most {MAX_UPLOAD_MB} MB. Parsed as data only -- nothing in the file is executed.")
    if uploaded is None:
        return
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"'{uploaded.name}' is {uploaded.size / 1e6:.1f} MB -- the limit is {MAX_UPLOAD_MB} MB.")
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
        assumed_omega = slider_with_custom(
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
    fig.add_trace(go.Bar(x=[f"step {k + 1}" for k in range(DPC_HORIZON)], y=rmse_per_step, marker_color=series_color("electrical")))
    style_figure(fig, height=340, x_title="Horizon step", y_title="RMSE (V)", legend=False)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
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




def render() -> None:
    page_header(
        "Simulation / VSC",
        "Direct Power Control of a Voltage Source Converter",
        "A trained DPC network tracking a rotating voltage reference on an identified VSC plant. Power electronics only: no rotating machinery, no bearing, no fault model.",
        [source_badge_html("simulated"), badge_html("TRAINED NETWORKS: DPC v3 near 8 Ω / v4 elsewhere (selected by operating point)", "training", "◆", f"{_DPC_NOMINAL_WEIGHTS_PATH.name} + {_DPC_WEIGHTS_PATH.name}")],
    )
    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl:
        with st.container(border=True):
            st.markdown("**Reference**")
            duration_s = slider_with_custom(st, "Duration (s)", 0.02, 1.0, 0.1, key="dpc_duration")
            seed = st.number_input("Seed (reference start phase)", value=0, step=1, key="dpc_seed", help="The plant/controller pair is deterministic; the seed picks the reference's starting phase.")
            magnitude = slider_with_custom(st, "Reference magnitude |v_ref| (V)", 10.0, 150.0, float(REFERENCE_MAGNITUDE_V), key="dpc_magnitude")
            omega = slider_with_custom(st, "Reference frequency ω (rad/s)", 50.0, 700.0, float(GRID_OMEGA_RAD_S), format="%.2f", key="dpc_omega")
            st.caption(f"Training defaults: |v_ref| = {REFERENCE_MAGNITUDE_V:.0f} V, ω = {GRID_OMEGA_RAD_S:.2f} rad/s — any other value is an off-distribution probe.")
        with st.container(border=True):
            st.markdown("**Electrical**")
            load_r = slider_with_custom(st, "Load resistance R (Ω)", MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, float(_VSC_R_OHM), format="%.4f", key="dpc_r_ohm",
                                        help=f"Default {_VSC_R_OHM} Ω is the constant R of the network's training data. Below ~{MIN_STABLE_LOAD_RESISTANCE_OHM:.2f} Ω the plant itself is open-loop unstable (sim/vsc_system.py).")
            if load_r < MIN_STABLE_LOAD_RESISTANCE_OHM:
                st.warning(f"R = {load_r:.4f} Ω is below the plant's open-loop stability floor (~{MIN_STABLE_LOAD_RESISTANCE_OHM:.2f} Ω): divergence is expected regardless of the controller.")
            limit_on = st.toggle(f"Limit converter voltage to ±{UPSTREAM_CONVERTER_LIMIT_V:.0f} V per axis", value=False, key="dpc_voltage_limit",
                                 help=f"The original converter (DPC4PowerElectronics, V_dc = {UPSTREAM_DC_LINK_V:.0f} V) clamps its voltage command at ±{UPSTREAM_CONVERTER_LIMIT_V:.0f} V per α/β axis. Off (default): SPADE's averaged plant has no limit, so high references are tracked although the real converter could not deliver them (S-09).")
        params = {"duration_s": float(duration_s), "seed": int(seed), "load_resistance_ohm": float(load_r), "reference_magnitude_v": float(magnitude), "reference_omega_rad_s": float(omega)}
        if limit_on:
            params["vsc_voltage_limit_v"] = UPSTREAM_CONVERTER_LIMIT_V
        current = st.session_state.get("dpc_result")
        unchanged = current is not None and current["run"]["config_hash"] == config_hash(params)
        clicked = st.button("Generate DPC scenario", type="primary", key="dpc_run_button", width="stretch", disabled=unchanged, icon=":material/play_arrow:")
    if clicked:
        with view, st.spinner("Simulating…"):
            records = run_scenario(Scenario(scenario_id="dashboard_dpc_run", controller_type="DPC", plant_config_id="vsc_dpc_v1", **params))
        df = pd.DataFrame.from_records(records)
        rmse = float(np.sqrt(np.mean((df["vc_real"] - df["v_ref_real"]) ** 2 + (df["vc_imag"] - df["v_ref_imag"]) ** 2)))
        st.session_state["dpc_result"] = {"df": df, "params": params, "rmse": rmse, "run": record_run("vsc", params, {"rmse_v": round(rmse, 3)})}
        st.rerun()

    with view:
        tab_t, tab_plane, tab_eval = st.tabs(["Tracking (time)", "Complex voltage plane", "Evaluate your dataset"])
        with tab_eval:
            _render_dpc_upload_eval()
        result = st.session_state.get("dpc_result")
        if result is None:
            for t in (tab_t, tab_plane):
                with t:
                    st.info("Configure the scenario on the left and click **Generate DPC scenario**.", icon=":material/info:")
            return
        df, p, run = result["df"], result["params"], result["run"]
        on_dist = abs(p["load_resistance_ohm"] - float(_VSC_R_OHM)) < 1e-9 and abs(p["reference_magnitude_v"] - float(REFERENCE_MAGNITUDE_V)) < 1e-9 and abs(p["reference_omega_rad_s"] - float(GRID_OMEGA_RAD_S)) < 1e-9
        with tab_t:
            if run["config_hash"] != config_hash(params):
                st.warning(f"Controls changed since run {run['run_id']} — charts still show that run.", icon=":material/history:")
            kpi_row([
                {"label": "Closed-loop RMSE", "value": f"{result['rmse']:.3f}", "unit": "V", "sub": "|v_c − v_ref| over the run", "tone": "primary"},
                {"label": "Load resistance", "value": f"{p['load_resistance_ohm']:.3f}", "unit": "Ω", "sub": "training default" if abs(p["load_resistance_ohm"] - float(_VSC_R_OHM)) < 1e-9 else "off-distribution", "tone": "healthy" if abs(p["load_resistance_ohm"] - float(_VSC_R_OHM)) < 1e-9 else "warning"},
                {"label": "Reference", "value": f"{p['reference_magnitude_v']:.0f}", "unit": "V", "sub": f"ω = {p['reference_omega_rad_s']:.2f} rad/s", "tone": "training"},
                {"label": "Operating point", "value": "on-distribution" if on_dist else "off-distribution", "sub": "vs. the network's training data", "tone": "healthy" if on_dist else "warning"},
            ])
            _converter_limit_notice(df, p)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Voltage — reference vs. actual")
                st.plotly_chart(time_series_figure(df, [("vc_real", "v_real (V)"), ("vc_imag", "v_imag (V)")], ref_cols=["v_ref_real", "v_ref_imag"]), width="stretch", config=PLOTLY_CONFIG, key="dpc_ts_v")
            with c2:
                st.markdown("##### Filter current")
                st.plotly_chart(time_series_figure(df, [("i_f_real", "i_f_real (A)"), ("i_f_imag", "i_f_imag (A)")], role="vibration"), width="stretch", config=PLOTLY_CONFIG, key="dpc_ts_i")
            c1, c2 = st.columns([3, 1])
            with c1:
                provenance_table({"Run ID": run["run_id"], "Data origin": "Simulation: identified VSC plant + trained DPC network", "Network weights": f"{_DPC_NOMINAL_WEIGHTS_PATH.name} (7.25-8.75 Ω, ≤ 62.5 V; or ≤ 10 V at 5-12 Ω) / {_DPC_WEIGHTS_PATH.name} (elsewhere)", "Seed": p["seed"], "Configuration hash": run["config_hash"], "Timestamp (UTC)": run["utc"]})
            with c2:
                st.download_button("Export run (CSV)", df.to_csv(index=False), file_name=f"spade_{run['run_id']}.csv", key="dpc_export_csv", width="stretch", icon=":material/download:")
        with tab_plane:
            st.markdown("##### Complex voltage plane (v_c vs. v_ref)")
            st.plotly_chart(_dpc_figure(df), width="stretch", config=PLOTLY_CONFIG, key="dpc_plane")
            st.caption("v_c should trace the same rotating path as v_ref; the deviation is the closed-loop tracking error summarised as RMSE.")
