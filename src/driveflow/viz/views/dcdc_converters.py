"""Simulation / DC-DC Converters (spec Sec. 8.1; expansion task E6). One route with a tab per
converter: Buck now, DAB only when task E9 adds it. Same Generate pattern as the VSC / DPC page:
controls on the left, nothing drawn until a run exists, the charts always belong to the run named
in the provenance table."""

from __future__ import annotations

import streamlit as st

from driveflow.viz.adapters import dcdc_adapter as da
from driveflow.viz.adapters.session import config_hash, record_run
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.metric_cards import kpi_row
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.components.signal_chart import time_series_figure
from driveflow.viz.theme.plotly import PLOTLY_CONFIG


def _fmt(x, spec: str, none: str = "—") -> str:
    return none if x is None else format(x, spec)


def _buck_controls() -> tuple[dict, bool]:
    d = da.buck_defaults()
    with st.container(border=True):
        st.markdown("**Operating conditions**")
        vin = slider_with_custom(st, "Input voltage V_in (V)", 12.0, 100.0, float(d["vin_v"]), key="buck_vin")
        vref = slider_with_custom(st, "Output reference v_ref (V)", 1.0, 95.0, float(d["vout_ref_v"]), key="buck_vref")
        load = slider_with_custom(st, "Load resistance R (Ω)", 1.0, 50.0, float(d["load_resistance_ohm"]), key="buck_load")
        duration = slider_with_custom(st, "Duration (s)", 0.005, 0.1, float(d["duration_s"]), format="%.3f", key="buck_duration")
    with st.container(border=True):
        st.markdown("**Load step**")
        step_on = st.toggle("Apply a load step", value=bool(d["load_step_on"]), key="buck_step_on")
        step_ohm, step_t = float(d["load_step_ohm"]), float(d["load_step_time_s"])
        if step_on:
            step_ohm = st.slider("Load after the step (Ω)", 1.0, 50.0, step_ohm, key="buck_step_ohm")
            step_t = st.slider("Step time (s)", 0.001, 0.1, step_t, format="%.3f", key="buck_step_t")
    with st.expander("Plant parameters", icon=":material/tune:"):
        l_uh = st.number_input("Inductance L (µH)", 1.0, 10_000.0, d["l_h"] * 1e6, 10.0, key="buck_l_uh")
        c_uf = st.number_input("Capacitance C (µF)", 1.0, 10_000.0, d["c_f"] * 1e6, 10.0, key="buck_c_uf")
        r_l = st.number_input("Inductor resistance r_L (Ω)", 0.0, 5.0, float(d["r_l_ohm"]), 0.01, key="buck_rl")
        i_max = st.number_input("Inductor current rating i_max (A)", 0.5, 200.0, float(d["i_max_a"]), 0.5, key="buck_imax",
                                help="Caps the controller's current reference.")
        st.caption(f"Control period τ = {d['tau_s'] * 1e6:g} µs (one update per switching period at {1 / d['tau_s'] / 1e3:g} kHz).")
    params = {"vin_v": float(vin), "vout_ref_v": float(vref), "load_resistance_ohm": float(load), "duration_s": float(duration),
              "load_step_on": bool(step_on), "load_step_ohm": float(step_ohm), "load_step_time_s": float(step_t),
              "l_h": float(l_uh) * 1e-6, "c_f": float(c_uf) * 1e-6, "r_l_ohm": float(r_l), "i_max_a": float(i_max)}
    current = st.session_state.get("buck_result")
    unchanged = current is not None and current["run"]["config_hash"] == config_hash(params)
    clicked = st.button("Generate Buck scenario", type="primary", key="buck_run_button", width="stretch", disabled=unchanged, icon=":material/play_arrow:")
    return params, clicked


def _buck_tab() -> None:
    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl:
        params, clicked = _buck_controls()
    if clicked:
        try:
            scenario = da.buck_scenario(params)
        except ValueError as exc:
            with view:
                st.error(f"Invalid configuration: {exc}", icon=":material/error:")
            return
        with view, st.spinner("Simulating…"):
            df = da.run_buck(params)
        metrics = da.buck_metrics(df, params)
        summary = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
        st.session_state["buck_result"] = {"df": df, "params": params, "metrics": metrics, "scenario_id": scenario.scenario_id,
                                           "run": record_run("buck", params, summary)}
        st.rerun()
    with view:
        result = st.session_state.get("buck_result")
        if result is None:
            st.info("Configure the converter on the left and click **Generate Buck scenario**.", icon=":material/info:")
            return
        df, p, m, run = result["df"], result["params"], result["metrics"], result["run"]
        if run["config_hash"] != config_hash(params):
            st.warning(f"Controls changed since run {run['run_id']} — charts still show that run.", icon=":material/history:")
        if not m["finite"]:
            st.error("The simulation produced non-finite values.", icon=":material/error:")
        settled = m["settling_time_ms"] is not None
        kpi_row([
            {"label": "Start-up settling (2 %)", "value": _fmt(m["settling_time_ms"], ".2f", "did not settle"), "unit": "ms" if settled else None,
             "sub": f"overshoot {m['overshoot_pct']:.1f} %", "tone": "primary" if settled else "warning"},
            {"label": "Steady-state error", "value": f"{m['steady_state_error_mv']:.2f}", "unit": "mV", "sub": "mean |v_ref − v_out|, last 10 % before the step", "tone": "primary"},
            {"label": "Load-step dip", "value": _fmt(m["load_dip_v"], ".2f"), "unit": "V" if m["load_dip_v"] is not None else None,
             "sub": "no load step" if m["load_dip_v"] is None else f"back within 1 % in {_fmt(m['load_recovery_ms'], '.2f', 'not recovered')} ms",
             "tone": "training"},
            {"label": "Operating point (end)", "value": f"{m['final_v_out']:.2f}", "unit": "V", "sub": f"i_L {m['final_i_l']:.2f} A · duty {m['final_duty']:.3f}", "tone": "healthy"},
        ])
        if m["max_i_l"] > p["i_max_a"] * 1.0001:
            st.warning(f"The inductor current peaked at {m['max_i_l']:.2f} A, above the {p['i_max_a']:g} A rating: the controller caps its reference, not the plant.",
                       icon=":material/warning:")
        vlines = [p["load_step_time_s"]] if p["load_step_on"] else None
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Output voltage")
            st.plotly_chart(time_series_figure(df, [("buck_v_out", "v_out (V)")], ref_cols=["buck_v_ref"], vlines=vlines), width="stretch", config=PLOTLY_CONFIG, key="buck_ts_v")
        with c2:
            st.markdown("##### Inductor current")
            st.plotly_chart(time_series_figure(df, [("buck_i_l", "i_L (A)")], ref_cols=["buck_i_ref"], role="vibration", vlines=vlines), width="stretch", config=PLOTLY_CONFIG, key="buck_ts_i")
        st.markdown("##### Duty cycle")
        st.plotly_chart(time_series_figure(df, [("buck_duty", "duty (–)")], role="mechanical", vlines=vlines), width="stretch", config=PLOTLY_CONFIG, key="buck_ts_d")
        st.caption("Averaged model: the duty is held over each period and the switching ripple is not represented. "
                   f"L-C resonance {m['resonance_hz']:.0f} Hz. Dashed: references (v_ref; i_ref from the voltage loop). Dotted line: load step.")
        c1, c2 = st.columns([3, 1])
        with c1:
            provenance_table({"Run ID": run["run_id"], "Data origin": "Simulation: averaged synchronous buck (sim/buck_system.py) + BuckVoltageController",
                              "Plant config": "buck_v1", "Configuration hash": run["config_hash"], "Timestamp (UTC)": run["utc"]})
            st.dataframe(da.buck_parameters_table(p), hide_index=True, width="stretch")
        with c2:
            st.download_button("Export run (CSV)", df.to_csv(index=False), file_name=f"spade_{run['run_id']}.csv", key="buck_export_csv", width="stretch", icon=":material/download:")


def _dab_tab() -> None:
    """Dual active bridge (task E9 preparation): the modulation selector and the phase-shift
    diagram from the chosen parameters. No DAB simulation exists in SPADE yet, so every result panel
    is in its empty state with the reason."""
    import plotly.graph_objects as go

    from driveflow.viz.adapters import dab_adapter as dab
    from driveflow.viz.theme.plotly import series_color, style_figure

    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl, st.container(border=True):
        st.markdown("**Modulation**")
        strategy = st.selectbox("Strategy", list(dab.STRATEGIES), index=3, key="dab_strategy", help="TPS shown because the PLECS circuit supports triple phase shift.")
        st.caption(dab.STRATEGIES[strategy])
        d0 = st.slider("Outer phase shift D0 (× Ts/2)", -1.0, 1.0, 0.3, 0.01, key="dab_d0")
        d1 = st.slider("Primary inner shift D1", 0.0, 1.0, 0.8, 0.01, key="dab_d1", disabled=strategy == "SPS")
        d2 = st.slider("Secondary inner shift D2", 0.0, 1.0, 0.9, 0.01, key="dab_d2", disabled=strategy in ("SPS", "DPS", "EPS"))
    d0, d1, d2 = dab.constrain(strategy, d0, d1, d2)
    status = dab.data_status()
    with view:
        st.markdown("##### Phase-shift diagram (from the selected parameters — not a simulation)")
        w = dab.bridge_waveforms(d0, d1, d2)
        fig = go.Figure(go.Scatter(x=w["t"], y=w["v_p"], mode="lines", name="v_p / V_in (primary bridge)", line=dict(width=2, shape="hv", color=series_color("candidate"))))
        fig.add_trace(go.Scatter(x=w["t"], y=w["v_s"] * 0.9, mode="lines", name="v_s / V_o (secondary bridge, drawn at 0.9)", line=dict(width=2, shape="hv", color=series_color("mechanical"))))
        style_figure(fig, height=260, x_title="t / Ts", y_title="Normalized bridge voltage")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key="dab_phase_diagram")
        st.caption(f"D0 = {d0:.2f}, D1 = {d1:.2f}, D2 = {d2:.2f} ({strategy}). Each bridge is a three-level square wave; "
                   "its non-zero part lasts D·Ts/2 per half-period and the secondary is delayed by D0·Ts/2.")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.markdown("**Simulated waveforms and performance**")
            st.info("No simulation has been executed for this configuration.", icon=":material/info:")
            st.caption("The DAB plant is a PLECS circuit (Fund. DAB_sample1.plecs, licence unclear: non-commercial use, not shipped). "
                       + ("It is present in data/upstream/fund/dab/; " if status["circuit_available"] else "It is not present on this machine; ")
                       + "running it needs PLECS (not executed: no PLECS) and top-level outports added to the circuit. Use PLECS Plant Studio → DAB template.")
        with c2, st.container(border=True):
            st.markdown("**Modulation suggestion**")
            st.html('<div class="spade-root spade-badges">' + badge_html("AI SUGGESTION — REQUIRES VERIFICATION", "warning", "▲") + "</div>")
            if status["trained_runs"]:
                st.caption(f"Trained DAB performance run(s) on disk: {', '.join(status['trained_runs'])}. Suggestions are not wired to this page until real PLECS data exists.")
            else:
                st.info("No DAB performance model has been trained: it needs a PLECS sweep table (none exists).", icon=":material/info:")
            st.button("Compare suggested vs baseline", key="dab_compare", disabled=True, help="Needs a suggestion and PLECS runs of both modulations under the same conditions.")


def render() -> None:
    page_header("Simulation / DC-DC Converters", "DC-DC Converters",
                "Switched-mode DC-DC converters under closed-loop control. Buck: averaged model of a synchronous buck with an average-current-mode "
                "PI cascade regulating the output voltage. DAB: modulation selector and phase-shift diagram; its simulation needs PLECS.",
                [source_badge_html("simulated"), badge_html("AVERAGED MODEL", "unknown", "◆", "No switching ripple; valid well below f_sw")])
    tab_buck, tab_dab = st.tabs(["Buck", "DAB"])
    with tab_buck:
        _buck_tab()
    with tab_dab:
        _dab_tab()
