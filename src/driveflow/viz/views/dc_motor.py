"""DC Motor page (guide Sec. 3.4, figures/gui/concept_v2/02_dc_motor_diagnostics.png).

Left 25 %: scenario controls; right 75 %: KPIs, then Time signals / Spectrum (FFT) / AI diagnosis /
Operating envelope / Parameters tabs, then provenance + export. Same physics as the legacy Fase A
single-simulation mode (run_scenario on dc_perm_ex_v1), plus the PI/MPC controller choice the
backend already supported but the old UI never exposed. Advanced Flow (segment sequences) is the
second mode of this page.

Honesty rules applied here:
- The fault you inject is shown as "Injected condition (simulation input)" -- ground truth of the
  simulation, never presented as a diagnosis.
- The classifier output is an "AI suggestion" with its model ID and test metrics, flagged
  "requires verification" -- not an alarm, not a certified diagnosis.
- The motor widget is UNKNOWN here: no monitoring agent has evaluated this run (the Monitoring
  page does that); it spins with this run's simulated speed.
- Every result carries its run_id and frozen parameters; changing a control afterwards marks the
  result as stale instead of relabelling it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.datagen import run_flow
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.monitoring.motor_status import derive_motor_view
from driveflow.viz.adapters import simulation_adapter as sim
from driveflow.viz.adapters.ai_adapter import classify_last_window
from driveflow.viz.adapters.session import config_hash, record_run
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.metric_cards import kpi_row
from driveflow.viz.components.motor_status import render_motor_status
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.components.presets import BUILTIN_FAULT_TYPES, apply_pending_profile_load, custom_fault_manager, load_custom_faults, profile_manager
from driveflow.viz.components.signal_chart import spectrum_figure, time_series_figure
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure

MAX_HISTORY = 4
MAX_FLOW_RUNS = 6
_ELEC_ROWS = [("current_r", "i_a (A)"), ("rpm", "Speed (rpm)"), ("torque_nm", "Torque (N·m)"), ("voltage_v", "u_a (V)")]
_VIB_ROWS = [("acc_x", "acc_x (synthetic)"), ("acc_y", "acc_y (synthetic)"), ("acc_z", "acc_z (synthetic)")]
_SPECTRUM_CHANNELS = {"current_r": ("Armature current i_a", "A", "electrical"), "acc_x": ("Vibration acc_x (calibrated axis)", "synthetic unit", "vibration"), "acc_y": ("Vibration acc_y (assumed axis)", "synthetic unit", "vibration"), "acc_z": ("Vibration acc_z (assumed axis)", "synthetic unit", "vibration"), "torque_nm": ("Shaft torque", "N·m", "mechanical")}


def _envelope_figure(df: pd.DataFrame, p: dict) -> go.Figure:
    omega = df["rpm"] * 2 * np.pi / 60.0
    om = np.linspace(0, p["omega_lim"], 200)
    i_v = np.clip((p["u_lim"] - p["psi_e"] * om) / p["r_a"], 0, p["i_lim"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=om, y=i_v, mode="lines", name="voltage-limited envelope", line=dict(color=series_color("reference"), dash="dot", width=1.3)))
    fig.add_trace(go.Scatter(x=[0, p["omega_lim"]], y=[p["i_lim"]] * 2, mode="lines", name=f"current limit ({p['i_lim']:.0f} A)", line=dict(color=series_color("secondary"), dash="dot", width=1.3)))
    fig.add_trace(go.Scatter(x=omega, y=df["current_r"], mode="lines", name="trajectory", line=dict(color=series_color("electrical"), width=1.6)))
    fig.add_trace(go.Scatter(x=[omega.iloc[-1]], y=[df["current_r"].iloc[-1]], mode="markers", name="end", marker=dict(size=10, color=series_color("electrical"))))
    style_figure(fig, height=480, x_title="ω (rad/s)", y_title="i_a (A)")
    fig.update_xaxes(range=[0, p["omega_lim"]])
    fig.update_yaxes(range=[0, p["i_lim"] * 1.1])
    return fig


def _controls(col, custom_faults: dict, motor: dict) -> dict:
    with col:
        with st.container(border=True):
            st.markdown("**Scenario configuration**")
            controller = st.selectbox("Controller", ["PI", "MPC"], key="dc_controller", help="PI: cascaded speed/current PI. MPC: linear model-predictive controller (control/mpc/controller.py). Both drive the same DC plant.")
            fault_label = st.selectbox("Fault condition", BUILTIN_FAULT_TYPES + sorted(custom_faults), key="dc_fault_type", help="Bearing fault injected on the electrical (MCSA torque ripple) and synthetic-vibration paths. Entries after 'cage' are your saved custom fault types (Parameters tab).")
            mode = st.radio("Control mode", ["Speed (ω_ref)", "Torque (τ_ref)"], horizontal=True, key="dc_control_mode")
            if mode.startswith("Speed"):
                setpoint = slider_with_custom(st, "Speed setpoint ω_ref (rad/s)", 10.0, 350.0, 300.0, key="dc_omega_ref", help="300 rad/s = the motor's nominal speed (GEM spec).")
            else:
                setpoint = slider_with_custom(st, "Torque setpoint τ_ref (N·m)", 0.0, 38.0, 16.0, key="dc_torque_ref", help="16 N·m nominal, 38 N·m limit (GEM spec). Speed is free-running in this mode.")
        with st.container(border=True):
            st.markdown("**Fault severity**")
            fault_type = None if fault_label == "healthy" else fault_label
            elec = slider_with_custom(st, "Electrical severity (N·m, MCSA path)", 0.0, 20.0, 8.0, key="dc_elec_sev")
            mech = slider_with_custom(st, "Mechanical severity (Module B units)", 0.0, 0.2, float(CALIBRATED_MECHANICAL_SEVERITY.get(fault_type, 0.0)), format="%.3f", key="dc_mech_sev") if fault_type is not None else None
            if fault_type is None:
                st.caption("No fault selected — severities have no effect.")
        with st.container(border=True):
            st.markdown("**Experiment setup**")
            duration = slider_with_custom(st, "Duration (s)", 0.05, 2.0, 1.0, key="dc_duration", help="≈1 s to settle at the nominal setpoint.")
            seed = st.number_input("Random seed", value=0, step=1, key="dc_seed", help="Same seed + parameters = the same run, including added noise.")
            with st.expander("Post-simulation noise"):
                e_noise = slider_with_custom(st, "Electrical noise (%)", 0.0, 20.0, 0.0, key="dc_elec_noise")
                m_noise = slider_with_custom(st, "Vibration noise (%)", 0.0, 20.0, 0.0, key="dc_mech_noise")
    return {
        "controller": controller, "fault_label": fault_label, "control_mode": "speed" if mode.startswith("Speed") else "torque", "setpoint": float(setpoint),
        "duration_s": float(duration), "seed": int(seed), "electrical_severity": float(elec), "mechanical_severity": None if mech is None else float(mech),
        "electrical_noise_pct": float(e_noise), "mechanical_noise_pct": float(m_noise), "motor": motor,
    }


def _history_nav() -> dict | None:
    history = st.session_state.get("dc_history", [])
    if not history:
        return None
    idx = min(st.session_state.get("dc_history_idx", len(history) - 1), len(history) - 1)
    c1, c2, c3 = st.columns([1, 4, 1])
    if c1.button("◀ Previous", key="dc_history_prev", disabled=idx <= 0):
        st.session_state["dc_history_idx"] = idx - 1
        st.rerun()
    if c3.button("Next ▶", key="dc_history_next", disabled=idx >= len(history) - 1):
        st.session_state["dc_history_idx"] = idx + 1
        st.rerun()
    entry = history[idx]
    c2.caption(f"Result {idx + 1}/{len(history)} · {entry['run']['run_id']} · {entry['run']['utc']}")
    return entry


def _single() -> None:
    apply_pending_profile_load()
    custom_faults = load_custom_faults()
    defaults = sim.dc_defaults()
    params_col, view_col = st.columns([1, 3], gap="medium")
    with view_col:
        kpi_slot = st.container()
        tabs = st.tabs(["Time signals", "Spectrum (FFT)", "AI diagnosis", "Operating envelope", "Parameters"])
        footer = st.container()

    # Parameters tab first: its widget values feed the run configuration below.
    with tabs[4]:
        st.markdown("##### Motor characteristics (DcPermanentlyExcitedMotor)")
        st.caption("Overrides GEM's motor_parameter for the next run. Stock values: sim/motors/dc_permanently_excited_motor.py.")
        c1, c2, c3 = st.columns(3)
        r_a = c1.number_input("Armature resistance r_a (Ω)", min_value=0.001, value=float(defaults["params"]["r_a"]), format="%.4f", key="dc_r_a")
        l_a = c2.number_input("Armature inductance l_a (H)", min_value=1e-6, value=float(defaults["params"]["l_a"]), format="%.6f", key="dc_l_a")
        psi_e = c3.number_input("Magnetic flux ψ_e (Wb)", min_value=0.01, value=float(defaults["params"]["psi_e"]), format="%.4f", key="dc_psi_e")
        profile_manager(st, group="dc", current_values=dict(r_a=r_a, l_a=l_a, psi_e=psi_e))
        st.divider()
        custom_fault_manager(st.container(border=True))
    motor = dict(r_a=r_a, l_a=l_a, psi_e=psi_e)

    params = _controls(params_col, custom_faults, motor)
    current = st.session_state.get("dc_result")
    unchanged = current is not None and current["run"]["config_hash"] == config_hash(params)
    with params_col:
        clicked = st.button("Run simulation", type="primary", key="dc_run_button", width="stretch", disabled=unchanged, icon=":material/play_arrow:", help="Disabled: this exact configuration produced the result shown." if unchanged else None)
        st.caption("All parameters apply to the next run only.")
    if clicked:
        with params_col, st.spinner("Simulating…"):
            df = sim.run_dc(params, custom_faults)
            ai = classify_last_window("dc_motor", df)
        rpm = sim.steady_state(df, "rpm")
        run = record_run("dc", params, {"injected": params["fault_label"], "rpm": None if rpm is None else round(rpm, 1), "ai_suggestion": ai["predicted"] if ai else None})
        entry = {"df": df, "params": params, "run": run, "ai": ai, "envelope": dict(r_a=r_a, psi_e=psi_e, u_lim=float(defaults["limits"]["u"]), i_lim=float(defaults["limits"]["i"]), omega_lim=float(defaults["limits"]["omega"]))}
        history = st.session_state.setdefault("dc_history", [])
        history.append(entry)
        del history[:-MAX_HISTORY]
        st.session_state["dc_history_idx"] = len(history) - 1
        st.session_state["dc_result"] = entry
        st.rerun()

    with view_col, kpi_slot:
        entry = _history_nav()
        if entry is not None:
            st.session_state["dc_result"] = entry
    if entry is None:
        with kpi_slot:
            kpi_row([{"label": l, "value": "—", "sub": "run a simulation", "tone": "unknown"} for l in ("Speed", "Armature current", "Shaft torque", "AI suggestion")])
        for t in tabs[:4]:
            with t:
                st.info("Configure the scenario on the left and click **Run simulation**.", icon=":material/info:")
        return

    df, p, run, ai = entry["df"], entry["params"], entry["run"], entry["ai"]
    stale = run["config_hash"] != config_hash(params)
    rpm, cur, torque = sim.steady_state(df, "rpm"), sim.steady_state(df, "current_r"), sim.steady_state(df, "torque_nm")
    target_rpm = p["setpoint"] * 60 / (2 * np.pi) if p["control_mode"] == "speed" else None
    with kpi_slot:
        if stale:
            st.warning("Controls changed since this result was produced — it still shows run " + run["run_id"] + "'s parameters. Click Run simulation to refresh.", icon=":material/history:")
        kpi_row([
            {"label": "Speed (steady state)", "value": "—" if rpm is None else f"{rpm:,.0f}", "unit": "rpm", "sub": f"target {target_rpm:,.0f} rpm" if target_rpm else "torque mode — speed free-running", "tone": "primary"},
            {"label": "Armature current", "value": "—" if cur is None else f"{cur:.1f}", "unit": "A", "sub": f"nominal {sim.dc_defaults()['nominal']['i']} A", "tone": "training"},
            {"label": "Shaft torque", "value": "—" if torque is None else f"{torque:.2f}", "unit": "N·m", "sub": f"nominal {sim.dc_defaults()['nominal']['torque']} N·m", "tone": "healthy"},
            {"label": "AI suggestion", "value": ai["predicted"] if ai else "n/a", "sub": (f"p={ai['confidence']:.2f} · requires verification" if ai else "no promoted classifier / not enough rows"), "tone": "warning" if ai else "unknown"},
        ])
        st.html('<div class="spade-root spade-badges">' + badge_html(f"Injected condition (simulation input): {p['fault_label']}", "unknown", "◆") + source_badge_html("simulated") + "</div>")

    with tabs[0]:
        ref = [None, target_rpm, p["setpoint"] if p["control_mode"] == "torque" else None, None]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Electrical / mechanical")
            st.plotly_chart(time_series_figure(df, _ELEC_ROWS, ref_cols=ref, role="electrical"), width="stretch", config=PLOTLY_CONFIG, key="dc_ts_elec")
        with c2:
            st.markdown("##### Vibration (synthetic, Module B)")
            st.plotly_chart(time_series_figure(df, _VIB_ROWS, role="vibration"), width="stretch", config=PLOTLY_CONFIG, key="dc_ts_vib")
        st.caption(f"Sampling: {sim.FS_HZ:,.0f} Hz ({len(df):,} samples). Vibration provenance: {sim.VIBRATION_PROVENANCE}.")

    with tabs[1]:
        c1, c2, c3 = st.columns([2, 2, 1])
        channel = c1.selectbox("Channel", list(_SPECTRUM_CHANNELS), format_func=lambda c: _SPECTRUM_CHANNELS[c][0], key="dc_fft_channel")
        skip = c2.slider("Discard start-up transient (fraction of record)", 0.0, 0.8, 0.2, 0.05, key="dc_fft_skip")
        log_y = c3.toggle("Log amplitude", value=True, key="dc_fft_log")
        name, unit, role = _SPECTRUM_CHANNELS[channel]
        markers = sim.fault_markers(df)
        try:
            fig, peak = spectrum_figure(df[channel].to_numpy(), sim.FS_HZ, name, unit, role=role, markers=markers, skip_fraction=skip, log_y=log_y, fmax_hz=2_500.0)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key="dc_fft_chart")
            rows = {"Dominant component (computed)": f"{peak[0]:.1f} Hz, {peak[1]:.4g} {unit}" if peak[0] else "—", "Frequency resolution": f"{sim.FS_HZ / (len(df) * (1 - skip)):.2f} Hz"}
            rows.update({f"{k} (simulator)": f"{v:.1f} Hz" for k, v in markers.items()})
            provenance_table(rows, "Spectrum summary")
            st.caption("Hann-windowed single-sided amplitude spectrum of this run (driveflow.analysis.spectrum). Dotted lines: shaft frequency and the bearing characteristic frequencies computed by the simulator for this run's speed — markers, not detections.")
        except ValueError as exc:
            st.warning(f"Spectrum unavailable for this channel: {exc}")

    with tabs[2]:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown("##### Fault classification — AI suggestion")
            if ai is None:
                st.info("No promoted PC-tier classifier for dc_motor, or the run is shorter than one input window.")
            else:
                st.html('<div class="spade-root spade-badges">' + badge_html("REQUIRES VERIFICATION", "warning", "▲", "A model output on simulated data, not a certified diagnosis.") + badge_html(f"model {ai['model_id']}", "training") + "</div>")
                order = np.argsort(ai["probs"])
                fig = go.Figure(go.Bar(x=[ai["probs"][i] for i in order], y=[ai["classes"][i] for i in order], orientation="h", marker_color=[series_color("candidate") if ai["classes"][i] == ai["predicted"] else series_color("base") for i in order], text=[f"{ai['probs'][i]:.0%}" for i in order], textposition="auto"))
                style_figure(fig, height=60 + 48 * len(order), x_title="Class probability (softmax, uncalibrated)", legend=False)
                fig.update_xaxes(range=[0, 1])
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key="dc_ai_probs")
                f1 = ai.get("test_macro_f1")
                st.caption(f"Prediction on the last input window of this run. Model test macro-F1 at training time: {f1:.3f} on n={ai.get('n_test')} windows (simulated data, grouped-by-run split)." if f1 is not None else "No test macro-F1 recorded for this model.")
                st.caption(f"Ground truth of this simulation: **{p['fault_label']}** (the injected condition). Agreement here says nothing about real machines.")
        with c2:
            view = derive_motor_view(None, "dc_motor", motor_id="dc_motor/sim", source_mode="simulated", rpm=rpm, rpm_target=target_rpm, vibration_rms_g=sim.vibration_rms(df), timestamp_utc=run["utc"], prediction=ai["predicted"] if ai else None, confidence=ai["confidence"] if ai else None, model_id=ai["model_id"] if ai else None)
            render_motor_status(view)
            st.page_link("pages/monitoring.py", label="Evaluate with the monitoring agents", icon=":material/monitor_heart:")

    with tabs[3]:
        st.markdown("##### Operating envelope (ω, i_a)")
        st.plotly_chart(_envelope_figure(df, entry["envelope"]), width="stretch", config=PLOTLY_CONFIG, key="dc_envelope")

    with footer:
        st.markdown("##### Run metadata")
        c1, c2 = st.columns([3, 1])
        with c1:
            provenance_table({
                "Run ID": run["run_id"], "Data origin": "Synthetic simulation (driveflow.datagen.run_scenario, dc_perm_ex_v1)",
                "Controller": p["controller"], "Injected condition": p["fault_label"], "Seed": p["seed"], "Duration": f"{p['duration_s']} s",
                "Configuration hash": run["config_hash"], "Timestamp (UTC)": run["utc"], "Validation": "Not gated (simulation output)",
                "Vibration": "synthetic_approximate (1 calibrated axis)",
            })
        with c2:
            st.download_button("Export run (CSV)", df.to_csv(index=False), file_name=f"spade_{run['run_id']}.csv", key="dc_export_csv", width="stretch", icon=":material/download:")
            meta = {"run": run, "params": p, "ai_suggestion": ai, "vibration_provenance": sim.VIBRATION_PROVENANCE}
            st.download_button("Export metadata (JSON)", json.dumps(meta, indent=2, default=str), file_name=f"spade_{run['run_id']}_metadata.json", key="dc_export_meta", width="stretch", icon=":material/data_object:")


# ------------------------------------------------------------------------- Advanced flow mode
def _describe_segment(seg: dict) -> str:
    ref = f"ω_ref={seg['setpoint']:.0f} rad/s" if seg["control_mode"] == "speed" else f"τ_ref={seg['setpoint']:.1f} N·m"
    return f"{seg['duration_s']:.2f}s @ {ref}, {seg['fault_label']}"


def _parse_float_list(raw: str) -> list:
    out = []
    for token in raw.split(","):
        try:
            out.append(float(token.strip()))
        except ValueError:
            continue
    return out


def _flow() -> None:
    custom_faults = load_custom_faults()
    defaults = sim.dc_defaults()
    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl:
        st.caption("Each segment is an independent simulation (fresh start); the flow shows a sequence of operating states on one timeline, not one continuous trajectory.")
        with st.container(border=True):
            st.markdown("**Motor (shared by all segments)**")
            c1, c2, c3 = st.columns(3)
            r_a = c1.number_input("r_a (Ω)", min_value=0.001, value=float(defaults["params"]["r_a"]), format="%.4f", key="flow_r_a")
            l_a = c2.number_input("l_a (H)", min_value=1e-6, value=float(defaults["params"]["l_a"]), format="%.6f", key="flow_l_a")
            psi_e = c3.number_input("ψ_e (Wb)", min_value=0.01, value=float(defaults["params"]["psi_e"]), format="%.4f", key="flow_psi_e")
            controller = st.selectbox("Controller", ["PI", "MPC"], key="flow_controller")
            seed = st.number_input("Seed (shared)", value=0, step=1, key="flow_seed")
        with st.container(border=True):
            st.markdown("**Add a segment**")
            mode = st.radio("Control mode", ["Speed (ω_ref)", "Torque (τ_ref)"], horizontal=True, key="flow_add_control_mode")
            if mode.startswith("Speed"):
                setpoint = slider_with_custom(st, "Setpoint ω_ref (rad/s)", 10.0, 350.0, 300.0, key="flow_add_speed")
            else:
                setpoint = slider_with_custom(st, "Setpoint τ_ref (N·m)", 0.0, 38.0, 16.0, key="flow_add_torque")
            duration = slider_with_custom(st, "Duration (s)", 0.05, 2.0, 0.5, key="flow_add_duration")
            fault_label = st.selectbox("Fault condition", BUILTIN_FAULT_TYPES + sorted(custom_faults), key="flow_add_fault")
            elec = slider_with_custom(st, "Electrical severity (N·m)", 0.0, 20.0, 8.0, key="flow_add_elec_sev")
            mech = slider_with_custom(st, "Mechanical severity", 0.0, 0.2, float(CALIBRATED_MECHANICAL_SEVERITY.get(None if fault_label == "healthy" else fault_label, 0.0)), format="%.3f", key="flow_add_mech_sev")
            base_seg = dict(control_mode="speed" if mode.startswith("Speed") else "torque", setpoint=float(setpoint), duration_s=float(duration), fault_label=fault_label, electrical_severity=float(elec), mechanical_severity=float(mech), electrical_noise_pct=0.0, mechanical_noise_pct=0.0)
            if st.button("+ Add segment", key="flow_add_button", width="stretch"):
                st.session_state.setdefault("flow_segments", []).append(dict(base_seg))
                st.rerun()
            with st.expander("Add a sequence (vary one field)"):
                vary = st.radio("Vary", ["Fault type", "Setpoint"], horizontal=True, key="flow_seq_vary")
                if vary == "Fault type":
                    faults = st.multiselect("Fault conditions, in order", BUILTIN_FAULT_TYPES + sorted(custom_faults), key="flow_seq_faults")
                    if st.button("+ Add sequence", key="flow_seq_add_fault", disabled=len(faults) < 2):
                        st.session_state.setdefault("flow_segments", []).extend({**base_seg, "fault_label": f} for f in faults)
                        st.rerun()
                else:
                    sps = _parse_float_list(st.text_input("Setpoints, comma-separated", key="flow_seq_setpoints", placeholder="e.g. 100, 200, 300"))
                    if st.button("+ Add sequence", key="flow_seq_add_setpoint", disabled=len(sps) < 2):
                        st.session_state.setdefault("flow_segments", []).extend({**base_seg, "setpoint": sp} for sp in sps)
                        st.rerun()
        segments = st.session_state.get("flow_segments", [])
        with st.container(border=True):
            st.markdown(f"**Queue ({len(segments)} segment{'s' if len(segments) != 1 else ''})**")
            if not segments:
                st.caption("Empty — add at least one segment.")
            for i, seg in enumerate(segments):
                c_desc, c_up, c_down, c_del = st.columns([5, 1, 1, 1])
                c_desc.caption(f"{i}. {_describe_segment(seg)}")
                if c_up.button("▲", key=f"flow_seg_up_{i}", disabled=i == 0, help="Move earlier"):
                    segments[i - 1], segments[i] = segments[i], segments[i - 1]
                    st.rerun()
                if c_down.button("▼", key=f"flow_seg_down_{i}", disabled=i == len(segments) - 1, help="Move later"):
                    segments[i + 1], segments[i] = segments[i], segments[i + 1]
                    st.rerun()
                if c_del.button("✕", key=f"flow_seg_del_{i}", help="Remove"):
                    del segments[i]
                    st.rerun()
        c_run, c_clear = st.columns(2)
        run_clicked = c_run.button("Run flow", key="flow_run_button", type="primary", disabled=not segments, icon=":material/play_arrow:")
        if c_clear.button("Clear queue", key="flow_clear_button", disabled=not segments):
            st.session_state["flow_segments"] = []
            st.rerun()
    if run_clicked:
        motor = dict(r_a=r_a, l_a=l_a, psi_e=psi_e)
        with view, st.spinner(f"Simulating {len(segments)} segment(s)…"):
            from driveflow.datagen import Scenario

            scenarios = []
            for i, seg in enumerate(segments):
                ft, override, extras = sim.resolve_fault(seg["fault_label"], custom_faults)
                kw = dict(scenario_id=f"flow_segment_{i}", controller_type=controller, fault_type=ft, duration_s=seg["duration_s"], seed=int(seed), electrical_severity=seg["electrical_severity"], mechanical_severity=seg["mechanical_severity"], motor_parameter_overrides=motor, electrical_noise_pct=seg["electrical_noise_pct"], mechanical_noise_pct=seg["mechanical_noise_pct"], fault_order_override=override, extra_faults=extras)
                kw["omega_ref_rad_s" if seg["control_mode"] == "speed" else "torque_ref_nm"] = seg["setpoint"]
                scenarios.append(Scenario(**kw))
            df = pd.DataFrame.from_records(run_flow(scenarios))
        run = record_run("flow", {"segments": segments, "motor": motor, "seed": int(seed), "controller": controller}, {"segments": len(segments)})
        runs = st.session_state.setdefault("flow_runs", [])
        runs.append({"id": run["run_id"], "name": f"Flow {run['run_id']}", "df": df, "segments": [dict(s) for s in segments], "run": run})
        del runs[:-MAX_FLOW_RUNS]
    with view:
        runs = st.session_state.get("flow_runs", [])
        if not runs:
            st.info("Queue one or more segments on the left, then click **Run flow**.", icon=":material/info:")
            return
        for entry in reversed(runs):
            df = entry["df"]
            with st.container(border=True):
                entry["name"] = st.text_input("Name", value=entry["name"], key=f"flow_run_name_{entry['id']}", label_visibility="collapsed")
                st.caption(f"{entry['run']['run_id']} · {df['timestamp_s'].iloc[-1]:.2f} s total · " + " → ".join(f"[{i}] {_describe_segment(s)}" for i, s in enumerate(entry["segments"])))
                omega_ref = np.full(len(df), np.nan)
                torque_ref = np.full(len(df), np.nan)
                for i, seg in enumerate(entry["segments"]):
                    mask = (df["segment_index"] == i).to_numpy()
                    if seg["control_mode"] == "speed":
                        omega_ref[mask] = seg["setpoint"] * 60 / (2 * np.pi)
                    else:
                        torque_ref[mask] = seg["setpoint"]
                bounds = df.loc[df["segment_index"].diff() != 0, "timestamp_s"].iloc[1:].tolist()
                c1, c2 = st.columns(2)
                c1.plotly_chart(time_series_figure(df, _ELEC_ROWS, ref_cols=[None, omega_ref, torque_ref, None], vlines=bounds), width="stretch", config=PLOTLY_CONFIG, key=f"flow_chart_elec_{entry['id']}")
                c2.plotly_chart(time_series_figure(df, _VIB_ROWS, role="vibration", vlines=bounds), width="stretch", config=PLOTLY_CONFIG, key=f"flow_chart_vib_{entry['id']}")
                c_dl, c_rm = st.columns([1, 4])
                c_dl.download_button("Export CSV", df.to_csv(index=False), file_name=f"spade_{entry['id']}.csv", key=f"flow_dl_{entry['id']}")
                if c_rm.button("Remove", key=f"flow_remove_{entry['id']}"):
                    runs.remove(entry)
                    st.rerun()


def render() -> None:
    page_header(
        "Simulation / DC motor",
        "Motor health & signal diagnostics",
        "Configure the experiment, inspect the physical signals, and compare the injected condition with the AI suggestion.",
        [source_badge_html("simulated"), badge_html("VIBRATION: SYNTHETIC (1 AXIS CALIBRATED)", "warning", "▲", sim.VIBRATION_PROVENANCE)],
    )
    mode = st.radio("Mode", ["Single simulation", "Advanced flow"], horizontal=True, key="app_mode", label_visibility="collapsed")
    if mode == "Single simulation":
        _single()
    else:
        _flow()
