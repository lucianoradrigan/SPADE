"""AI Diagnostics page (legacy IA tab; docs/design_ai_layer_transversal.md Sec. 5.1).

Consumes data -- a short simulated sample run or an uploaded file -- and shows what the promoted
PC-tier classifier / forecaster of the chosen domain outputs for it. It is not a control surface.

Two tabs: "Evaluate a model" (the promoted classifier, or any other classifier run on this machine,
on a sample run whose options match the simulation pages, or on an uploaded file) and "Train a
model" (views/ai_training.py: several fault classes over a custom simulated grid, as a worker job).

What changed vs. the legacy tab: outputs are labelled as AI suggestions with the model's identity
and training-time test metrics; uploads go through the explicit INFERENCE validator and are capped
in size; the data source (simulated vs. uploaded file + content hash) is always shown.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai.audit_findings import SYNTHETIC_LABEL, is_synthetic_class, load_audit_findings
from driveflow.ai.registry import RegistryError, resolve
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.models.common.windowing import CANDIDATE_CHANNELS, VSC_DPC_CANDIDATE_CHANNELS
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM
from driveflow.ai import condition
from driveflow.viz.adapters import ai_adapter as da_ai
from driveflow.viz.adapters.ai_adapter import normalize_window, promoted_model, run_metrics
from driveflow.viz.adapters.session import record_run
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.ai.custom_training import list_classifier_runs
from driveflow.sim.vsc_params import UPSTREAM_CONVERTER_LIMIT_V
from driveflow.viz.adapters import simulation_adapter as sim
from driveflow.viz.components.presets import BUILTIN_FAULT_TYPES, load_custom_faults
from driveflow.viz.views.ai_training import render_train_tab
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
    kwargs.pop("controller_type", None)
    return pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="ai_tab_sample_vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", **kwargs)))


def channel_upload_template_df(domain: str, n_rows: int = 5) -> pd.DataFrame:
    """A real short simulation, restricted to channels it actually populates (an all-NaN column
    would make the example fail its own validator)."""
    df = generate_sample_dataframe(domain, duration_s=0.05, seed=0)
    usable = [c for c in CANDIDATES[domain] if c in df.columns and df[c].notna().all()]
    return df[usable].head(n_rows).reset_index(drop=True)


def _sample_controls(domain: str) -> dict:
    """Scenario kwargs for the sample run, with the same options as the DC Motor / VSC pages."""
    if domain == "dc_motor":
        custom_faults = load_custom_faults()
        controller = st.selectbox("Controller", ["PI", "MPC"], key="ia_controller")
        label = st.selectbox("Fault condition", BUILTIN_FAULT_TYPES + sorted(custom_faults), key="ia_fault_type",
                             help="Entries after 'cage' are your saved custom fault types (DC Motor → Parameters).")
        fault, override, extras = sim.resolve_fault(label, custom_faults)
        mode = st.radio("Control mode", ["Speed (ω_ref)", "Torque (τ_ref)"], horizontal=True, key="ia_control_mode")
        kwargs = {"controller_type": controller, "fault_type": fault}
        if mode.startswith("Speed"):
            kwargs["omega_ref_rad_s"] = slider_with_custom(st, "Speed setpoint ω_ref (rad/s)", 10.0, 350.0, DC_DEFAULT_OMEGA_REF, key="ia_omega_ref")
        else:
            kwargs["torque_ref_nm"] = slider_with_custom(st, "Torque setpoint τ_ref (N·m)", 0.0, 38.0, 16.0, key="ia_torque_ref")
        if fault is not None:
            kwargs["electrical_severity"] = slider_with_custom(st, "Electrical severity (N·m)", 0.0, 20.0, 8.0, key="ia_elec_severity")
            kwargs["mechanical_severity"] = slider_with_custom(st, "Mechanical severity", 0.0, 0.2, float(CALIBRATED_MECHANICAL_SEVERITY.get(fault, 0.02)), format="%.3f", key="ia_mech_severity")
            if override is not None:
                kwargs["fault_order_override"] = override
            if extras:
                kwargs["extra_faults"] = extras
        kwargs["duration_s"] = slider_with_custom(st, "Duration (s)", 0.05, 2.0, 0.6, key="ia_duration")
        kwargs["seed"] = int(st.number_input("Seed", value=0, step=1, key="ia_seed"))
        with st.expander("Noise and plant parameters"):
            kwargs["electrical_noise_pct"] = slider_with_custom(st, "Electrical noise (%)", 0.0, 100.0, 0.0, key="ia_elec_noise")
            kwargs["mechanical_noise_pct"] = slider_with_custom(st, "Vibration noise (%)", 0.0, 100.0, 0.0, key="ia_mech_noise")
            d = sim.dc_defaults()["params"]
            motor = {"r_a": st.number_input("r_a (Ω)", 0.001, 10.0, float(d["r_a"]), format="%.4f", key="ia_r_a"),
                     "l_a": st.number_input("l_a (H)", 1e-6, 1.0, float(d["l_a"]), format="%.6f", key="ia_l_a"),
                     "psi_e": st.number_input("ψ_e (Wb)", 0.01, 5.0, float(d["psi_e"]), format="%.4f", key="ia_psi_e"),
                     "j_rotor": st.number_input("J rotor (kg·m²)", 1e-4, 1.0, float(d.get("j_rotor", 0.025)), format="%.4f", key="ia_j_rotor")}
            motor = {k: v for k, v in motor.items() if abs(v - float(d.get(k, v))) > 1e-12}
            if motor:
                kwargs["motor_parameter_overrides"] = motor
            if controller == "PI":
                b = st.slider("PI setpoint weight b", 0.0, 1.0, 1.0, 0.05, key="ia_pi_b")
                if b != 1.0:
                    kwargs["pi_setpoint_weight"] = b
        return kwargs
    kwargs = {"duration_s": slider_with_custom(st, "Duration (s)", 0.05, 1.0, 0.3, key="ia_duration"), "seed": int(st.number_input("Seed", value=0, step=1, key="ia_seed"))}
    kwargs["load_resistance_ohm"] = slider_with_custom(st, "Load resistance R (Ω)", MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, float(_VSC_R_OHM), format="%.4f", key="ia_load_r", help=f"Floored at the plant's open-loop stability limit ({MIN_STABLE_LOAD_RESISTANCE_OHM:.2f} Ω).")
    kwargs["reference_magnitude_v"] = slider_with_custom(st, "Reference magnitude |v_ref| (V)", 10.0, 150.0, float(REFERENCE_MAGNITUDE_V), key="ia_ref_mag")
    kwargs["reference_omega_rad_s"] = slider_with_custom(st, "Reference frequency ω (rad/s)", 50.0, 700.0, float(GRID_OMEGA_RAD_S), format="%.2f", key="ia_ref_omega")
    with st.expander("Converter limit and capacitor degradation"):
        if st.toggle(f"Limit converter voltage to ±{UPSTREAM_CONVERTER_LIMIT_V:.0f} V per axis", value=False, key="ia_vsc_limit"):
            kwargs["vsc_voltage_limit_v"] = float(UPSTREAM_CONVERTER_LIMIT_V)
        cap = st.slider("Effective capacitance (fraction of nominal)", 0.05, 1.0, 1.0, key="ia_vsc_cap")
        esr = st.slider("Series ESR (Ω)", 0.0, 2.0, 0.0, key="ia_vsc_esr")
        if cap != 1.0:
            kwargs["vsc_capacitor_scale"] = cap
        if esr != 0.0:
            kwargs["vsc_esr_ohm"] = esr
        st.caption("Capacitor degradation is synthetic-only (sim/vsc_faults.py).")
    return kwargs


def _model_choice(domain: str):
    """None = the promoted classifier; otherwise a run directory on this machine."""
    runs = list_classifier_runs(domain)
    try:
        promoted = resolve(domain, TIER, "classifier").resolve()
    except RegistryError:
        promoted = None
    options = [None] + [str(r["run_dir"]) for r in runs if r["run_dir"].resolve() != promoted]
    labels = {None: "Promoted model (registry)"}
    labels.update({str(r["run_dir"]): f"{r['label']} · {r['origin']} · {len(r['classes'] or [])} classes" + (f" · F1 {r['macro_f1']:.2f}" if r.get("macro_f1") is not None else "")
                   for r in runs})
    requested = st.session_state.pop("ia_model_request", None)
    if requested in options:
        st.session_state["ia_model"] = requested
    if st.session_state.get("ia_model") not in options:
        st.session_state.pop("ia_model", None)
    return st.selectbox("Classifier", options, format_func=lambda o: labels.get(o, o), key="ia_model",
                        help="The promoted model, or any other classifier run on this machine: custom-trained (Train a model tab), transfer-learning candidates, earlier runs.")


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


def _state_box(a) -> None:
    """The one condition box (spec Sec. 8.2): each state has its own label, colour and icon."""
    body = f"**{a.label}.** {a.explanation}"
    if a.state == condition.INPUT_INVALID:
        st.error(body + "  \n" + "  \n".join(a.problems), icon=":material/block:")
    elif a.state == condition.STATE_UNKNOWN:
        st.error(body + f"  \nOOD score **{a.ood_score:.2f}** > threshold **{a.ood_threshold:.2f}** ({a.ood_method}). "
                 f"The top class ({a.top_class}) is not suggested. Not a confirmed diagnosis.", icon=":material/report:")
    elif a.state == condition.STATE_UNCERTAIN:
        st.warning(body + f"  \nTop class {a.top_class} at {a.confidence:.0%} is below the abstention threshold ({a.abstain_below:.2f}) fitted on the validation split, so no class is suggested.",
                   icon=":material/help:")
    elif a.ood_available:
        st.success(body, icon=":material/check_circle:")
    else:
        # Without an OOD check "known" only means "confident": not shown in green.
        st.info(f"**{a.label}** (confidence check only). The top class passed the confidence check; whether the input resembles the training conditions was not checked.",
                icon=":material/info:")


def _ood_section(a, info: dict) -> None:
    if not a.ood_available:
        st.caption(f"**OOD detection unavailable** — {a.ood_reason} The state *Unknown condition* cannot be reported for this model, and no OOD score is shown.")
    with st.expander("OOD method and validation", icon=":material/science:"):
        if not a.ood_available:
            st.caption("No validated OOD detector for this model. Detectors are fitted by ai/ood.fit_ood (task A1): statistics on the run's training windows, threshold on its validation windows.")
            return
        ood = info["metrics"]["ood"]
        st.caption(f"Method: **{ood['method']}** (ai/ood.py). Threshold {ood[ood['method']]['threshold']:.3f}: the {ood.get('target_coverage', 0.95):.0%} quantile of the score on in-distribution validation windows. No out-of-distribution data was used to fit it.")
        rep = da_ai.ood_validation_report(info["run_dir"], info["metrics"])
        if rep is None:
            st.caption("No evaluation report found for this run's detector in docs/audit/ai/.")
            return
        rows = [{"Evaluation set": "in-distribution test", "n": None, "Flagged unknown": rep["id_false_unknown_rate"], "AUROC vs ID test": None}]
        rows += [{"Evaluation set": k, "n": v["n"], "Flagged unknown": v["unknown_rate"], "AUROC vs ID test": v["auroc"]} for k, v in rep["ood_sets"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Flagged unknown": st.column_config.NumberColumn(format="percent"), "AUROC vs ID test": st.column_config.NumberColumn(format="%.3f")})
        st.caption(f"Source: `{rep['path']}`. Simulated windows only. A low flagged share on an OOD set means many of those inputs are NOT caught.")


def _agent_section(domain: str, df: pd.DataFrame, data_key: str, operating_point: dict | None) -> None:
    st.markdown("###### Secondary simulated-agent hypothesis")
    key = f"ia_agent_result_{domain}"
    stored = st.session_state.get(key)
    if st.button("Run simulated-agent hypothesis", key=f"ia_agent_{domain}", icon=":material/psychology:",
                 help="Compares the input against simulations of each fault hypothesis (several physics runs, ~10–20 s)."):
        with st.spinner("Simulating the agent's hypotheses…"):
            try:
                stored = {"data_key": data_key, "result": da_ai.agent_hypothesis(domain, df, operating_point)}
            except ValueError as exc:
                stored = {"data_key": data_key, "error": str(exc)}
        st.session_state[key] = stored
    if stored is None or stored["data_key"] != data_key:
        st.caption("Not run for this input. A secondary hint only: it never changes the classifier state above.")
        return
    if "error" in stored:
        st.caption(f"Unavailable: {stored['error']}")
        return
    r = stored["result"]
    st.caption(f"Closest simulated hypothesis: **{r['closest_hypothesis']}** · anomaly score {r['score']:.2f} (heuristic in [0, 1], not a calibrated probability) · "
               f"features {', '.join(r['features'])} · operating point {r['operating_point'] or 'n/a'}. Not a confirmed diagnosis.")


def _classifier_panel(domain: str, df: pd.DataFrame, data_key: str = "", operating_point: dict | None = None, provenance: dict | None = None, run_dir=None) -> None:
    st.markdown("##### Classifier")
    try:
        out = da_ai.assess_last_window(domain, df, TIER, run_dir=run_dir)
    except RegistryError as exc:
        if run_dir is not None:
            st.error(f"Could not load the selected classifier: {exc}", icon=":material/error:")
            return
        out = None
    if out is None:
        st.info(f"No classifier is registered yet for {DOMAIN_LABELS[domain]}.")
        return
    a, _, info = out
    metrics, run_dir = info["metrics"], info["run_dir"]
    st.html('<div class="spade-root spade-badges">' + badge_html("AI SUGGESTION — REQUIRES VERIFICATION", "warning", "▲") + badge_html(info["model_id"], "training")
            + ("" if info.get("promoted") else badge_html("CANDIDATE — NOT PROMOTED", "warning", "▲")) + "</div>")
    _state_box(a)
    if a.state != condition.INPUT_INVALID:
        shown = a.suggested_class or {condition.STATE_UNKNOWN: "Unknown condition", condition.STATE_UNCERTAIN: "No class suggested"}.get(a.state)
        st.metric("Classifier output", shown, f"top class {a.top_class}: {a.confidence * 100:.1f}% ({'temperature-calibrated' if a.calibrated else 'uncalibrated softmax'})", delta_color="off")
        synthetic = set(metrics.get("synthetic_only_classes") or []) | {c for c in a.classes if is_synthetic_class(c)}
        flagged = a.suggested_class if a.suggested_class in synthetic else (a.top_class if a.top_class in synthetic else None)
        if flagged:
            st.warning(f"**{SYNTHETIC_LABEL}** ({'suggested class' if flagged == a.suggested_class else 'top class, not suggested'}: {flagged}). "
                       "This class was trained on simulated faults whose amplitude is a placeholder, not calibrated against measured bearings (A2). Treat it as a hypothesis to check, not a diagnosis.",
                       icon=":material/science:")
        k = a.classes.index(a.top_class)
        highlight = a.state == condition.KNOWN
        fig = go.Figure(go.Bar(x=list(a.classes), y=list(a.probs), marker_color=[series_color("candidate") if (i == k and highlight) else series_color("base") for i in range(len(a.classes))]))
        style_figure(fig, height=240, x_title="Class", y_title="Probability", legend=False)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"ia_clf_probs_{domain}")
    _ood_section(a, info)
    if a.state != condition.INPUT_INVALID:
        _agent_section(domain, df, data_key, operating_point)
    agent = st.session_state.get(f"ia_agent_result_{domain}")
    evidence = da_ai.condition_evidence(a, info["model_id"], provenance, agent["result"] if agent and agent.get("data_key") == data_key and "result" in agent else None)
    st.download_button("Export evidence (JSON)", json.dumps(evidence, indent=2, default=str), file_name=f"spade_ai_condition_{domain}.json", key=f"ia_evidence_{domain}", icon=":material/download:")
    for f in load_audit_findings(run_dir):
        st.warning(f"**Audit finding:** {f.get('summary')}  \nEvidence: `{f.get('evidence')}`", icon=":material/warning:")
    st.caption(f"Tier: {info['tier']}. Channels: {', '.join(info['channels'])}. Recorded test macro-F1 {metrics.get('test_macro_f1')} / accuracy {metrics.get('test_accuracy')} on n={metrics.get('n_test')} simulated windows (grouped-by-run split). " + (f"Confidence is temperature-calibrated on the validation split (T = {metrics['calibration']['temperature']:.2f}); calibration fitted in-distribution does not make out-of-distribution inputs safe." if a.calibrated else "No calibration or abstention for this model."))


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


def _residual_panel(domain: str, df: pd.DataFrame) -> None:
    """Forecast-residual detector (task E14): forecast vs. measured signal over the horizon, the
    score against its healthy-validation threshold. It suggests, it does not classify."""
    st.markdown("##### Forecast residual")
    r = da_ai.residual_assessment(domain, df, TIER)
    if not r["available"]:
        st.info(r["reason"], icon=":material/info:")
        st.caption("The detector is a forecaster trained only on healthy simulated runs; it appears here once a dc_motor regressor with a residual detector is promoted.")
        return
    above = r["score"] > r["threshold"]
    st.html('<div class="spade-root spade-badges">' + badge_html("AI SUGGESTION — REQUIRES VERIFICATION", "warning", "▲") + source_badge_html("simulated") + badge_html(r["model_id"], "training") + "</div>")
    msg = (f"Residual score **{r['score']:.2f}** vs. threshold **{r['threshold']:.2f}** (the {r['quantile']:.0%} quantile on healthy validation windows). "
           f"{r['share_above']:.0%} of the run's {r['n_windows']} window(s) are above it.")
    if above:
        st.warning(f"**Above threshold:** {da_ai.RESIDUAL_TEXT}. {msg}", icon=":material/troubleshoot:")
    else:
        st.info(f"**Below threshold.** {msg}", icon=":material/check:")
    fig = go.Figure()
    w, h = r["input_window"], r["horizon"]
    t_ctx, t_fut = np.arange(-min(w, 4 * h), 0), np.arange(0, h)
    for i, ch in enumerate(r["target_channels"]):
        j = r["channels"].index(ch)
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(x=t_ctx, y=r["window"][w - len(t_ctx):w, j], mode="lines", name=f"{ch} (input)", line=dict(color=color, width=1.2), opacity=0.55))
        fig.add_trace(go.Scatter(x=t_fut, y=r["window"][w:, j], mode="lines", name=f"{ch} (measured)", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=t_fut, y=r["forecast"][:, j], mode="lines", name=f"{ch} (forecast)", line=dict(color=color, width=2, dash="dash")))
    fig.add_vline(x=0, line_dash="dot", line_color=series_color("reference"))
    style_figure(fig, height=340, x_title="Step relative to forecast start (1 step = 0.1 ms)", y_title="Value (physical units)")
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"ia_residual_{domain}")
    st.caption(f"A high residual means {da_ai.RESIDUAL_TEXT}. Trained and validated on simulated windows only; it does not raise an alert by itself.")


def render() -> None:
    page_header(
        "Intelligence / AI diagnostics",
        "Evaluate and train diagnosis models",
        "Run the promoted PC-tier classifier / forecaster of a domain (or another classifier run) on a simulated sample or an uploaded file, or train a new classifier on a custom simulated grid. Outputs are model suggestions, not certified diagnoses.",
        [badge_html("PC TIER", "training", "◆")],
    )
    domain = st.selectbox("Domain", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="ia_domain")
    eval_tab, train_tab = st.tabs(["Evaluate a model", "Train a model"])
    with train_tab:
        render_train_tab(domain)
    with eval_tab:
        _evaluate(domain)


def _evaluate(domain: str) -> None:
    ctrl, view = st.columns([1, 3], gap="medium")
    df, source_rows, data_key, operating_point = None, None, "", None
    with ctrl:
        model_run = _model_choice(domain)
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
            data_key = f"run:{source_rows['Run ID']}"
            # The setpoint of a generated run is known, so the agent's hypotheses use it (AI-N09).
            cfg = st.session_state.get("ia_df_config") or {}
            operating_point = {"omega_ref_rad_s": float(cfg["omega_ref_rad_s"])} if domain == "dc_motor" and "omega_ref_rad_s" in cfg else ({} if domain == "vsc_dpc" else None)
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
                        data_key = f"file:{source_rows['SHA-256']}"
        _edge_downloads(domain)

    with view:
        if df is None:
            st.info("Choose a data source on the left — generate a short sample run, or upload a file.", icon=":material/info:")
            return
        st.html('<div class="spade-root spade-badges">' + (source_badge_html("simulated") if source == "Generate a sample run" else source_badge_html("external", "Uploaded file -- its origin is not verified by SPADE")) + "</div>")
        st.caption(f"{len(df)} row(s) available for evaluation.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            _classifier_panel(domain, df, data_key, operating_point, source_rows, run_dir=model_run)
        with c2:
            if domain == "dc_motor":
                _residual_panel(domain, df)
            else:
                _regressor_panel(domain, df)
        if source_rows:
            provenance_table(source_rows, "Input provenance")
