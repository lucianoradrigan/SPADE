"""AI Diagnostics → Train a model: define a simulated condition grid with the same options as the
simulation pages, pick several fault classes, and train a classifier on it as a worker job
(driveflow.ai.custom_training via the ``classifier_training`` job kind).

The trained model is a candidate: it can be selected in the Evaluate tab and used as the starting
point on the Transfer Learning page. Promotion is not offered here.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai import custom_training as ct
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V
from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM
from driveflow.viz.adapters import jobs_adapter as ja
from driveflow.viz.adapters import simulation_adapter as sim
from driveflow.viz.adapters import training_adapter as tra
from driveflow.viz.components.job_card import job_card
from driveflow.viz.components.page_header import badge_html, provenance_table, source_badge_html
from driveflow.viz.components.presets import BUILTIN_FAULT_TYPES, load_custom_faults
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, style_figure

SEVERITY_CHOICES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
WINDOW_CHOICES = [256, 512, 1024, 2048, 4096]


def _levels(label: str, lo: float, hi: float, default: tuple, n_default: int, key: str, fmt: str = "%.1f", help: str | None = None) -> list:
    c1, c2 = st.columns([3, 1])
    a, b = c1.slider(label, lo, hi, default, format=fmt, key=f"{key}_range", help=help)
    n = c2.number_input("Levels", 1, 15, n_default, key=f"{key}_n")
    return [float(v) for v in (np.linspace(a, b, int(n)) if n > 1 and b > a else [a])]


def _dc_form() -> dict:
    custom_faults = load_custom_faults()
    defaults = sim.dc_defaults()["params"]
    with st.container(border=True):
        st.markdown("**1 · Classes (fault modes)**")
        labels = st.multiselect("Classes the model must tell apart", BUILTIN_FAULT_TYPES + sorted(custom_faults), default=["healthy", "outer_race", "inner_race"],
                                key="tr_dc_classes", help="Every selected fault becomes one class; 'healthy' is the fault-free class. Saved custom fault types (DC Motor → Parameters) are listed after 'cage'.")
        custom_mech = {}
        for lab in labels:
            if lab in custom_faults:
                custom_mech[lab] = st.number_input(f"Mechanical severity for '{lab}' (Module B units)", 0.0, 0.2, 0.02, 0.005, format="%.3f", key=f"tr_dc_mech_{lab}",
                                                   help="User-defined faults have no calibrated amplitude; built-in ones use CALIBRATED_MECHANICAL_SEVERITY.")
        builtin = [lab for lab in labels if lab in CALIBRATED_MECHANICAL_SEVERITY]
        if builtin:
            st.caption("Calibrated mechanical severity: " + ", ".join(f"{lab} {CALIBRATED_MECHANICAL_SEVERITY[lab]:g}" for lab in builtin)
                       + ". Ball, cage and custom faults are synthetic-only (uncalibrated amplitude).")
    with st.container(border=True):
        st.markdown("**2 · Operating conditions**")
        controllers = st.multiselect("Controllers", ["PI", "MPC"], default=["PI"], key="tr_dc_controllers")
        mode = st.radio("Control mode", ["Speed (ω_ref)", "Torque (τ_ref)"], horizontal=True, key="tr_dc_mode")
        if mode.startswith("Speed"):
            setpoints = _levels("Speed setpoints ω_ref (rad/s)", 10.0, 350.0, (100.0, 200.0), 3, "tr_dc_speed")
        else:
            setpoints = _levels("Torque setpoints τ_ref (N·m)", 0.0, 38.0, (8.0, 24.0), 3, "tr_dc_torque")
        scales = st.multiselect("Fault severity scales (× calibrated severity)", SEVERITY_CHOICES, default=[0.5, 1.0, 2.0], key="tr_dc_scales")
        elec = st.slider("Electrical severity at scale 1 (N·m, MCSA path)", 0.0, 20.0, ct.NOMINAL_ELECTRICAL_SEVERITY, key="tr_dc_elec")
        pi_b = st.slider("PI setpoint weight b", 0.0, 1.0, 1.0, 0.05, key="tr_dc_pi_b", help="Applies to PI runs only; 1 = the default controller.") if "PI" in controllers else 1.0
    with st.container(border=True):
        st.markdown("**3 · Robustness: noise and plant drift**")
        noise = st.slider("Max sensor noise (% of each channel's std)", 0.0, 100.0, 50.0, key="tr_dc_noise", help="Per run, electrical and vibration noise are drawn independently in [0, max].")
        clean = st.slider("Share of noise-free runs", 0.0, 1.0, 1 / 3, key="tr_dc_clean")
        c1, c2, c3 = st.columns(3)
        r_a = c1.number_input("r_a (Ω)", 0.001, 10.0, float(defaults["r_a"]), format="%.4f", key="tr_dc_r_a")
        l_a = c2.number_input("l_a (H)", 1e-6, 1.0, float(defaults["l_a"]), format="%.6f", key="tr_dc_l_a")
        psi = c3.number_input("ψ_e (Wb)", 0.01, 5.0, float(defaults["psi_e"]), format="%.4f", key="tr_dc_psi_e")
        drift = {}
        if st.toggle("Vary ψ_e per run", value=True, key="tr_dc_drift_psi"):
            drift["psi_e"] = list(st.slider("ψ_e factor", 0.5, 2.0, (0.9, 1.1), key="tr_dc_drift_psi_r"))
        if st.toggle("Vary rotor inertia J per run", value=True, key="tr_dc_drift_j"):
            drift["j_rotor"] = list(st.slider("J factor", 0.5, 2.0, (0.8, 1.5), key="tr_dc_drift_j_r"))
        if st.toggle("Vary r_a per run", value=False, key="tr_dc_drift_ra"):
            drift["r_a"] = list(st.slider("r_a factor", 0.5, 2.0, (0.9, 1.2), key="tr_dc_drift_ra_r"))
    motor = {k: v for k, v in {"r_a": r_a, "l_a": l_a, "psi_e": psi}.items() if abs(v - float(defaults[k])) > 1e-12}
    return {"domain": "dc_motor", "classes": tra.dc_classes(labels, custom_faults, custom_mech),
            "operating": {"controllers": controllers, "control_mode": "speed" if mode.startswith("Speed") else "torque", "setpoints": setpoints, "severity_scales": scales,
                          "electrical_severity_nm": elec, "noise_max_pct": noise, "noise_free_share": clean, "motor": motor or None, "drift": drift,
                          "pi_setpoint_weight": None if pi_b == 1.0 else pi_b}}


def _vsc_form() -> dict:
    with st.container(border=True):
        st.markdown("**1 · Classes**")
        labels = st.multiselect("Classes", [ct.HEALTHY, ct.VSC_DEGRADED], default=[ct.HEALTHY, ct.VSC_DEGRADED], key="tr_vsc_classes")
        classes = [{"label": ct.HEALTHY}] if ct.HEALTHY in labels else []
        if ct.VSC_DEGRADED in labels:
            cap = st.slider("Effective capacitance (fraction of nominal 15 µF)", 0.05, 1.0, (0.5, 0.8), key="tr_vsc_cap")
            esr = st.slider("Series ESR (Ω)", 0.0, 2.0, (0.0, 0.2), key="tr_vsc_esr")
            classes.append({"label": ct.VSC_DEGRADED, "capacitor_scale": list(cap), "esr_ohm": list(esr)})
            st.caption("Capacitor degradation is SYNTHETIC-ONLY: no measured reference exists for this fault (sim/vsc_faults.py).")
    with st.container(border=True):
        st.markdown("**2 · Operating conditions**")
        r = _levels("Load resistance R (Ω)", float(MIN_STABLE_LOAD_RESISTANCE_OHM), 20.0, (6.0, 12.0), 3, "tr_vsc_r", fmt="%.2f",
                    help=f"The DPC network was trained at R = {_VSC_R_OHM:.4f} Ω only; other values are robustness probes.")
        mag = _levels("Reference magnitude |v_ref| (V)", 10.0, 150.0, (float(REFERENCE_MAGNITUDE_V), float(REFERENCE_MAGNITUDE_V)), 1, "tr_vsc_mag")
        omg = _levels("Reference frequency ω (rad/s)", 50.0, 700.0, (float(GRID_OMEGA_RAD_S), float(GRID_OMEGA_RAD_S)), 1, "tr_vsc_omega", fmt="%.2f")
        limit = st.toggle("Limit converter voltage (±134 V per axis)", value=False, key="tr_vsc_limit")
    return {"domain": "vsc_dpc", "classes": classes, "operating": {"load_resistance_ohm": r, "reference_magnitude_v": mag, "reference_omega_rad_s": omg, "voltage_limit": limit}}


def _runs_and_model(domain: str) -> tuple[dict, dict]:
    d = ct.default_request(domain)
    with st.container(border=True):
        st.markdown("**4 · Runs**")
        c1, c2, c3, c4 = st.columns(4)
        dur = c1.number_input("Duration per run (s)", 0.05, 10.0, float(d["operating"]["duration_s"]), 0.05, key=f"tr_{domain}_dur")
        skip = c2.number_input("Discard start-up (s)", 0.0, 5.0, float(d["operating"]["skip_start_s"]), 0.05, key=f"tr_{domain}_skip",
                               help="Rows before this time are dropped before windowing (the speed loop settles in ≈0.45 s).")
        reps = c3.number_input("Runs per cell", 1, 50, int(d["operating"]["runs_per_cell"]), key=f"tr_{domain}_reps", help="Repetitions of every (class × operating point) cell with a new seed.")
        seed = c4.number_input("Grid seed", 0, 2**31 - 1, 0, key=f"tr_{domain}_seed", help="Seeds the per-run noise and drift draws and the simulation seeds.")
    with st.container(border=True):
        st.markdown("**5 · Model**")
        name = st.text_input("Model name", d["name"], key=f"tr_{domain}_name", help="Letters, digits, '_' and '-'. Runs go to configs/classifiers/custom/<name>/<date>_runNN/.")
        c1, c2 = st.columns(2)
        preset = c1.selectbox("Architecture", list(ct.PRESETS), index=list(ct.PRESETS).index(d["model"]["preset"]), format_func=ct.PRESETS.get, key=f"tr_{domain}_preset")
        window = c2.select_slider("Window (samples at 10 kHz)", WINDOW_CHOICES, value=d["model"]["input_window"], key=f"tr_{domain}_window")
        allowed = list(ct.DC_CHANNELS) + ["current_s", "current_t"] if domain == "dc_motor" else list(ct.VSC_CHANNELS)
        channels = st.multiselect("Input channels", allowed, default=d["model"]["channels"], key=f"tr_{domain}_channels",
                                  help="e.g. only current_r for a current-signature (MCSA) model, only acc_* for a vibration model.")
        c1, c2, c3 = st.columns(3)
        epochs = c1.number_input("Epochs", 1, 200, d["model"]["epochs"], key=f"tr_{domain}_epochs")
        batch = c2.select_slider("Batch size", [8, 16, 32, 64, 128], value=32, key=f"tr_{domain}_batch")
        cap = c3.number_input("Max windows per class", 50, 20000, d["model"]["cap_per_class"], 50, key=f"tr_{domain}_cap")
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Validation share (runs)", 0.05, 0.4, 0.2, 0.05, key=f"tr_{domain}_val")
        test = c2.number_input("Test share (runs)", 0.05, 0.4, 0.2, 0.05, key=f"tr_{domain}_test")
        tseed = c3.number_input("Training seed", 0, 2**31 - 1, 42, key=f"tr_{domain}_tseed")
        keep_rpm = st.checkbox("Keep the rpm level (speed reference, AI-N07)", value=domain == "dc_motor", key=f"tr_{domain}_rpm", disabled="rpm" not in channels)
        fit_ood = st.checkbox("Fit an OOD detector (Mahalanobis, threshold on validation)", value=True, key=f"tr_{domain}_ood")
    operating = {"duration_s": float(dur), "skip_start_s": float(skip), "runs_per_cell": int(reps), "seed": int(seed)}
    model = {"preset": preset, "input_window": int(window), "channels": channels, "epochs": int(epochs), "batch_size": int(batch), "cap_per_class": int(cap),
             "val_frac": float(val), "test_frac": float(test), "seed": int(tseed), "keep_rpm_level": bool(keep_rpm), "fit_ood": bool(fit_ood)}
    return {"name": name, "operating": operating}, model


def _confusion_figure(cm: list, classes: list) -> go.Figure:
    z = np.asarray(cm, dtype=float)
    share = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=share, x=classes, y=classes, colorscale="Blues", zmin=0, zmax=1, text=z.astype(int), texttemplate="%{text}",
                               hovertemplate="true %{y} → predicted %{x}: %{text} windows (%{z:.0%} of the row)<extra></extra>", showscale=False))
    style_figure(fig, height=80 + 50 * len(classes), x_title="Predicted class", y_title="True class", legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def _results(domain: str) -> None:
    st.markdown("##### Trained models")
    for j in tra.active_jobs():
        st.caption(f":material/schedule: Job `{j['job_id']}`: {j['state']}" + (f" ({j['done']}/{j['total']}, {j['phase']})" if j["total"] else ""))
    rs = tra.runs(domain)
    if not rs:
        st.info("No training run yet for this domain. Configure the grid on the left and select Start training.", icon=":material/info:")
        return
    run = next(r for r in rs if r.run_id == st.selectbox("Training run", [r.run_id for r in rs], key=f"tr_{domain}_run",
                                                         format_func=lambda i: f"{i} · {next((r.params.get('request') or {}).get('name') for r in rs if r.run_id == i)} · "
                                                                               f"{next(r.status for r in rs if r.run_id == i)}"))
    try:
        job = ja.service().get(run.job_id) if run.job_id else None
    except Exception:
        job = None
    if job is not None and run.status != "completed":
        job_card(job, key=f"tr_job_{run.run_id}")
    run_dir = tra.run_dir_of(run)
    if run_dir is None:
        st.warning(f"Run {run.run_id} ended as {run.status} without a trained model. {run.failure_reason or ''}", icon=":material/error:")
        failed = [c for c in run.conditions if c.get("status") == "failed"]
        if failed:
            st.dataframe(pd.DataFrame([{"Condition": c.get("condition"), "Error": c.get("error")} for c in failed]), hide_index=True, width="stretch")
        return
    m = tra.model_metrics(run_dir)
    st.html('<div class="spade-root spade-badges">' + source_badge_html("simulated") + badge_html("CANDIDATE — NOT PROMOTED", "warning", "▲") + "</div>")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test macro-F1", f"{m['test_macro_f1']:.3f}" if m.get("test_macro_f1") is not None else "—")
    c2.metric("Test accuracy", f"{m['test_accuracy']:.3f}" if m.get("test_accuracy") is not None else "—")
    c3.metric("Test windows", m.get("n_test"))
    c4.metric("OOD detector", "fitted" if "ood" in m else "none")
    left, right = st.columns(2, gap="medium")
    with left:
        if m.get("test_confusion_matrix"):
            st.plotly_chart(_confusion_figure(m["test_confusion_matrix"], m["classes"]), width="stretch", config=PLOTLY_CONFIG, key=f"tr_cm_{run.run_id}")
    with right:
        st.dataframe(pd.DataFrame([{"Class": c, "Test recall": r, "Synthetic-only": c in (m.get("synthetic_only_classes") or [])} for c, r in m["test_per_class_recall"].items()]),
                     hide_index=True, width="stretch", column_config={"Test recall": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Test windows come from runs never seen in training or validation (split by run). Simulated data only.")
    rel = run_dir.relative_to(ct.REPO_ROOT).as_posix() if run_dir.is_relative_to(ct.REPO_ROOT) else str(run_dir)
    if st.button("Evaluate this model", key=f"tr_use_{run.run_id}", icon=":material/science:", help="Selects it in the Evaluate tab."):
        st.session_state["ia_model_request"] = str(run_dir)
        st.success("Selected. Open the **Evaluate a model** tab.", icon=":material/check:")
    provenance_table({"Run ID": run.run_id, "Model directory": rel, "Classes": ", ".join(m["classes"]), "Channels": ", ".join(m["channels"]),
                      "Configuration hash": run.config_hash, "Status": m.get("status"), "Promotion": "not promoted — use experiments/promote_run.py after review"}, "Model provenance")


def render_train_tab(domain: str) -> None:
    st.caption("Simulate a condition grid with several fault classes and train a classifier on it. The job runs in the worker; you can keep using SPADE meanwhile. "
               "The result is a candidate model: evaluate it here or adapt it on the Transfer Learning page. Nothing is promoted.")
    form_col, out_col = st.columns([1, 1.25], gap="medium")
    with form_col:
        req = _dc_form() if domain == "dc_motor" else _vsc_form()
        extra, model = _runs_and_model(domain)
        req["name"] = extra["name"]
        req["operating"].update(extra["operating"])
        req["model"] = model
        with st.container(border=True):
            st.markdown("**6 · Start**")
            try:
                s = ct.summary(req)
            except ct.RequestError as exc:
                st.error(str(exc), icon=":material/error:")
                s = None
            workers = st.slider("Simulation processes", 1, max(1, os.cpu_count() or 2), tra.default_workers(), key=f"tr_{domain}_workers")
            if s is not None:
                eta = s["estimated_single_process_s"]
                st.caption(f"**{s['n_runs']} runs** ({', '.join(f'{k}: {v}' for k, v in s['runs_per_class'].items())}), {s['simulated_s']:.1f} s simulated. "
                           f"Rough simulation time ≈ {eta / 60:.1f} min, plus training (measured on the development laptop; "
                           "more processes did not shorten it there).")
            if not ja.live_workers():
                st.caption(":material/warning: No worker is running: the job will wait in the queue (Experiment Runs → Start local worker).")
            if st.button("Start training", type="primary", key=f"tr_{domain}_start", icon=":material/model_training:", disabled=s is None, width="stretch"):
                try:
                    job = tra.submit(req, workers)
                    st.success(f"Queued as job `{job.job_id}`.", icon=":material/schedule:")
                except Exception as exc:
                    st.error(f"Not submitted: {exc}", icon=":material/error:")
            st.button("Refresh", key=f"tr_{domain}_refresh", icon=":material/refresh:")
    with out_col:
        _results(domain)
