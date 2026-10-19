"""AI Diagnostics page (legacy IA tab; docs/design_ai_layer_transversal.md Sec. 5.1).

Consumes data -- a short simulated sample run or an uploaded file -- and shows what the promoted
PC-tier classifier / forecaster of the chosen domain outputs for it. It is not a control surface.

What changed vs. the legacy tab: outputs are labelled as AI suggestions with the model's identity
and training-time test metrics; uploads go through the explicit INFERENCE validator and are capped
in size; the data source (simulated vs. uploaded file + content hash) is always shown.
"""

import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai.audit_findings import load_audit_findings
from driveflow.ai.registry import RegistryError, resolve
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.models.common.windowing import CANDIDATE_CHANNELS, VSC_DPC_CANDIDATE_CHANNELS
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM
from driveflow.ai.calibration import UNCERTAIN, decide
from driveflow.viz.adapters.ai_adapter import normalize_window, promoted_model, run_metrics
from driveflow.viz.adapters.session import record_run
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.components.presets import BUILTIN_FAULT_TYPES
from driveflow.viz.dpc_upload_validation import validate_inference_upload
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure

DOMAIN_LABELS = {"dc_motor": "DC motor (dc_motor)", "vsc_dpc": "VSC / DPC (vsc_dpc)"}
TIER = "pc"
CANDIDATES = {"dc_motor": CANDIDATE_CHANNELS, "vsc_dpc": VSC_DPC_CANDIDATE_CHANNELS}
DC_DEFAULT_OMEGA_REF = 300.0  # GEM nominal speed, same default as the DC Motor page
EDGE_TIERS = {"rpi5": "Raspberry Pi 5 (TFLite FP16)", "esp32": "ESP32 (TFLite INT8)"}
MAX_UPLOAD_MB = 50
_PALETTE = ["#38BDF8", "#F59E0B", "#A78BFA", "#F472B6", "#22C55E", "#94A3B8"]


def generate_sample_dataframe(domain: str, **kwargs) -> pd.DataFrame:
    if domain == "dc_motor":
        return pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="ai_tab_sample_dc", **kwargs)))
    return pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="ai_tab_sample_vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", **kwargs)))


def channel_upload_template_df(domain: str, n_rows: int = 5) -> pd.DataFrame:
    """A real short simulation, restricted to channels it actually populates (an all-NaN column
    would make the example fail its own validator)."""
    df = generate_sample_dataframe(domain, duration_s=0.05, seed=0)
    usable = [c for c in CANDIDATES[domain] if c in df.columns and df[c].notna().all()]
    return df[usable].head(n_rows).reset_index(drop=True)


def _sample_controls(domain: str) -> dict:
    duration = slider_with_custom(st, "Duration (s)", 0.05, 1.0, 0.3, key="ia_duration")
    seed = st.number_input("Seed", value=0, step=1, key="ia_seed")
    kwargs = {"duration_s": duration, "seed": int(seed)}
    if domain == "dc_motor":
        label = st.selectbox("Fault condition", BUILTIN_FAULT_TYPES, key="ia_fault_type")
        fault = None if label == "healthy" else label
        kwargs["fault_type"] = fault
        kwargs["omega_ref_rad_s"] = slider_with_custom(st, "Speed setpoint ω_ref (rad/s)", 10.0, 350.0, DC_DEFAULT_OMEGA_REF, key="ia_omega_ref")
        if fault is not None:
            kwargs["electrical_severity"] = slider_with_custom(st, "Electrical severity (N·m)", 0.0, 20.0, 8.0, key="ia_elec_severity")
            kwargs["mechanical_severity"] = slider_with_custom(st, "Mechanical severity", 0.0, 0.2, float(CALIBRATED_MECHANICAL_SEVERITY.get(fault, 0.0)), format="%.3f", key="ia_mech_severity")
    else:
        kwargs["load_resistance_ohm"] = slider_with_custom(st, "Load resistance R (Ω)", MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, float(_VSC_R_OHM), format="%.4f", key="ia_load_r", help=f"Floored at the plant's open-loop stability limit ({MIN_STABLE_LOAD_RESISTANCE_OHM:.2f} Ω).")
        kwargs["reference_magnitude_v"] = slider_with_custom(st, "Reference magnitude |v_ref| (V)", 10.0, 150.0, float(REFERENCE_MAGNITUDE_V), key="ia_ref_mag")
        kwargs["reference_omega_rad_s"] = slider_with_custom(st, "Reference frequency ω (rad/s)", 50.0, 700.0, float(GRID_OMEGA_RAD_S), format="%.2f", key="ia_ref_omega")
    return kwargs


def _upload_help(domain: str) -> None:
    channels = CANDIDATES[domain]
    with st.expander("Required format", expanded=False):
        st.caption(f"At least one of these {len(channels)} column(s) (any of them, not all): `{'`, `'.join(channels)}`.")
        st.caption("CSV with a header row, or JSON (list of objects / object of equal-length arrays). Extra columns are ignored. The model reads the LAST rows in file order. For TRAINING data (labels, source_file, timestamp_s) use the Transfer Learning page.")
        tpl = channel_upload_template_df(domain)
        c1, c2 = st.columns(2)
        c1.download_button("Example (CSV)", tpl.to_csv(index=False), file_name=f"ia_{domain}_example.csv", key=f"ia_template_csv_{domain}")
        c2.download_button("Example (JSON)", tpl.to_json(orient="records", indent=2), file_name=f"ia_{domain}_example.json", key=f"ia_template_json_{domain}")


def _edge_downloads(domain: str) -> None:
    with st.expander("Edge artifacts (Raspberry Pi 5 / ESP32)", expanded=False):
        st.caption("Fixed, already-trained artifacts — not re-generated from the sample-run controls. Export only: running on real hardware is NOT verified (see Model Registry).")
        found = False
        for block in ("classifier", "regressor"):
            for tier, label in EDGE_TIERS.items():
                try:
                    run_dir = resolve(domain, tier, block)
                except RegistryError:
                    continue
                tfl = run_dir / "model.tflite"
                if not tfl.exists():
                    st.caption(f"{block.capitalize()} — {label}: promoted, not exported.")
                    continue
                found = True
                m = run_metrics(run_dir)
                head = f"macro-F1 {m['test_macro_f1']:.3f}" if m.get("test_macro_f1") is not None else (f"RMSE {m['test_rmse']:.3f} (normalized)" if m.get("test_rmse") is not None else "no test metric")
                data = tfl.read_bytes()
                st.download_button(f"{block.capitalize()} — {label}", data, file_name=f"{domain}_{tier}_{block}.tflite", key=f"ia_edge_dl_{domain}_{tier}_{block}")
                st.caption(f"{len(data) / 1024:.1f} KB · sha256 {hashlib.sha256(data).hexdigest()[:12]} · {head} · {run_dir.parent.name}/{run_dir.name}")
        if not found:
            st.caption(f"Nothing exported for {DOMAIN_LABELS[domain]} at the rpi5/esp32 tiers.")


def _last_window(df: pd.DataFrame, channels: list, n: int):
    missing = [c for c in channels if c not in df.columns or df[c].isna().all()]
    if missing:
        st.warning(f"Missing channel(s) this model needs: {', '.join(missing)} — cannot evaluate.")
        return None
    if len(df) < n:
        st.warning(f"Need at least {n} rows for one window — only {len(df)} available.")
        return None
    return df[channels].iloc[-n:].to_numpy(dtype=np.float32)


def _classifier_panel(domain: str, df: pd.DataFrame) -> None:
    st.markdown("##### Classifier")
    try:
        model, config, metrics, run_dir = promoted_model(domain, TIER, "classifier")
    except RegistryError:
        st.info(f"No classifier is registered yet for {DOMAIN_LABELS[domain]}.")
        return
    window = _last_window(df, metrics["channels"], config.input_window)
    if window is None:
        return
    wn, _, _ = normalize_window(window, metrics)
    classes = metrics["classes"]
    decision, probs = decide(model(wn[np.newaxis, ...], training=False).numpy()[0], classes, metrics)
    k = int(np.argmax(probs))
    calibrated = "calibration" in metrics
    st.html('<div class="spade-root spade-badges">' + badge_html("AI SUGGESTION — REQUIRES VERIFICATION", "warning", "▲") + badge_html(f"{run_dir.parent.name}/{run_dir.name}", "training") + "</div>")
    st.metric("Predicted class", decision, f"{probs[k] * 100:.1f}% confidence ({'temperature-calibrated' if calibrated else 'uncalibrated softmax'})", delta_color="off")
    if decision == UNCERTAIN:
        st.info(f"Top class **{classes[k]}** is below the abstention threshold ({metrics['calibration']['abstain_below']:.2f}) fitted on the validation split, so no class is suggested.", icon=":material/help:")
    fig = go.Figure(go.Bar(x=classes, y=probs, marker_color=[series_color("candidate") if i == k else series_color("base") for i in range(len(classes))]))
    style_figure(fig, height=260, x_title="Class", y_title="Probability", legend=False)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"ia_clf_probs_{domain}")
    for f in load_audit_findings(run_dir):
        st.warning(f"**Audit finding:** {f.get('summary')}  \nEvidence: `{f.get('evidence')}`", icon=":material/warning:")
    st.caption(f"Tier: {config.tier}. Channels: {', '.join(metrics['channels'])}. Recorded test macro-F1 {metrics.get('test_macro_f1')} / accuracy {metrics.get('test_accuracy')} on n={metrics.get('n_test')} simulated windows (grouped-by-run split). " + (f"Confidence is temperature-calibrated on the validation split (T = {metrics['calibration']['temperature']:.2f}); calibration fitted in-distribution does not make out-of-distribution inputs safe." if calibrated else "No calibration or abstention for this model."))


def _regressor_panel(domain: str, df: pd.DataFrame) -> None:
    st.markdown("##### Regressor (forecaster)")
    try:
        model, config, metrics, run_dir = promoted_model(domain, TIER, "regressor")
    except RegistryError:
        st.info(f"No regressor is registered yet for {DOMAIN_LABELS[domain]}.")
        return
    channels = metrics["channels"]
    window = _last_window(df, channels, config.input_window)
    if window is None:
        return
    wn, mu, sigma = normalize_window(window, metrics)
    forecast = model(wn[np.newaxis, ...], training=False).numpy()[0] * sigma + mu
    st.html('<div class="spade-root spade-badges">' + badge_html("MODEL FORECAST", "training", "◆") + badge_html(f"{run_dir.parent.name}/{run_dir.name}", "training") + "</div>")
    fig = go.Figure()
    n_ctx = min(len(window), 3 * config.horizon)
    t_ctx = np.arange(-n_ctx, 0)
    t_fut = np.arange(0, config.horizon)
    for i, ch in enumerate(channels):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(x=t_ctx, y=window[-n_ctx:, i], mode="lines", name=f"{ch} (input)", line=dict(color=color, width=1.2), opacity=0.55))
        fig.add_trace(go.Scatter(x=t_fut, y=forecast[:, i], mode="lines", name=f"{ch} (forecast)", line=dict(color=color, width=2, dash="dash")))
    fig.add_vline(x=0, line_dash="dot", line_color=series_color("reference"))
    style_figure(fig, height=340, x_title="Step relative to forecast start (1 step = 0.1 ms)", y_title="Value (physical units)")
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"ia_reg_{domain}")
    st.caption(f"Tier: {config.tier}. Horizon: {config.horizon} steps. Test RMSE (normalized) at training time: {metrics.get('test_rmse')} on n={metrics.get('n_test')} windows.")


def render() -> None:
    page_header(
        "Intelligence / AI diagnostics",
        "Promoted models on your data",
        "Run the promoted PC-tier classifier / forecaster of a domain on a simulated sample or an uploaded file. Outputs are model suggestions, not certified diagnoses.",
        [badge_html("PC TIER", "training", "◆")],
    )
    ctrl, view = st.columns([1, 3], gap="medium")
    df, source_rows = None, None
    with ctrl:
        domain = st.selectbox("Domain", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="ia_domain")
        source = st.radio("Data source", ["Generate a sample run", "Upload a file"], key="ia_source")
        if source == "Generate a sample run":
            with st.container(border=True):
                kwargs = _sample_controls(domain)
                clicked = st.button("Generate", type="primary", key="ia_generate", width="stretch", icon=":material/play_arrow:")
            if clicked:
                with st.spinner("Simulating…"):
                    st.session_state["ia_df"] = generate_sample_dataframe(domain, **kwargs)
                st.session_state["ia_df_domain"] = domain
                st.session_state["ia_df_config"] = kwargs
                st.session_state["ia_df_run"] = record_run("ai_sample", {"domain": domain, **kwargs})
            if st.session_state.get("ia_df_domain") == domain:
                df = st.session_state.get("ia_df")
                if df is not None and st.session_state.get("ia_df_config") != kwargs:
                    st.caption("⚠ Parameters changed since this run — click Generate to refresh.")
            source_rows = {"Data origin": "Simulated sample (driveflow.datagen.run_scenario)", "Run ID": (st.session_state.get("ia_df_run") or {}).get("run_id")}
        else:
            _upload_help(domain)
            uploaded = st.file_uploader("Upload dataset", type=["csv", "json"], key="ia_upload", help=f"At most {MAX_UPLOAD_MB} MB. Parsed as data only.")
            if uploaded is not None:
                raw = uploaded.getvalue()
                if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"File is {len(raw) / 1e6:.1f} MB — the limit is {MAX_UPLOAD_MB} MB.")
                else:
                    try:
                        df_raw = pd.read_json(uploaded) if uploaded.name.lower().endswith(".json") else pd.read_csv(uploaded)
                    except Exception as e:
                        st.error(f"Could not parse '{uploaded.name}': {e}")
                        df_raw = None
                    if df_raw is not None:
                        df, _, messages = validate_inference_upload(df_raw, domain)
                        for level, text in messages:
                            getattr(st, level)(text)
                        source_rows = {"Data origin": f"Uploaded file: {uploaded.name}", "SHA-256": hashlib.sha256(raw).hexdigest(), "Validation": "inference contract (channels only)"}
        _edge_downloads(domain)

    with view:
        if df is None:
            st.info("Choose a data source on the left — generate a short sample run, or upload a file.", icon=":material/info:")
            return
        st.html('<div class="spade-root spade-badges">' + (source_badge_html("simulated") if source == "Generate a sample run" else source_badge_html("external", "Uploaded file -- its origin is not verified by SPADE")) + "</div>")
        st.caption(f"{len(df)} row(s) available for evaluation.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            _classifier_panel(domain, df)
        with c2:
            _regressor_panel(domain, df)
        if source_rows:
            provenance_table(source_rows, "Input provenance")
