"""Simulation / VSC -- tab "Parameter Identification" (spec Sec. 8.3; expansion task E10b).

Estimates the filter's L, R_f and C from a window of v_c, i_f and v_o with ai/pinn/identification.py.
Data: a simulated run made in this tab (optionally with a degraded capacitor, so the estimate can be
checked against a known value) or an uploaded CSV. PLECS data arrives with the PLECS tasks and is
shown as not available. The permanent text says what the number is: a parameter estimate, not a
validated fault diagnosis."""

from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.ai.pinn import identification as idn
from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM
from driveflow.viz.adapters import identification_adapter as ia
from driveflow.viz.adapters.session import config_hash, record_run
from driveflow.viz.components.page_header import badge_html, provenance_table, source_badge_html
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure

SOURCES = ("Simulated run", "Uploaded file (CSV)", "PLECS")
MAX_UPLOAD_MB = 50


def _source_controls() -> tuple[dict, pd.DataFrame | None, dict | None]:
    """(params, df, provenance) for the chosen source; df None when there is nothing to analyze yet."""
    source = st.radio("Data source", SOURCES, key="pid_source", horizontal=True,
                      captions=["Closed-loop VSC/DPC simulation", "Columns: " + ", ".join(idn.COLUMNS), "Not available in this version"])
    if source == "PLECS":
        st.info("PLECS data is not available in this version: it arrives with the PLECS plant tasks.", icon=":material/block:")
        return {"source": source}, None, None
    c1, c2 = st.columns(2)
    load_r = c1.number_input("Load resistance R (Ω), given", MIN_STABLE_LOAD_RESISTANCE_OHM, 1000.0, float(_VSC_R_OHM), 0.1, format="%.4f", key="pid_load_r",
                             help="i_load = v_c / R is not measured; R is an operating condition the estimator takes as known.")
    if source == "Simulated run":
        c_scale = c2.slider("Capacitor C as a fraction of nominal (simulation truth)", 0.3, 1.0, 1.0, 0.05, key="pid_c_scale",
                            help="Synthetic capacitor degradation (sim/vsc_faults.py), so the estimate can be compared with a known value.")
        noise = c1.select_slider("Measurement noise on i_f and v_c (% of RMS)", [0.0, 1.0, 2.0, 5.0], 1.0, key="pid_noise")
        params = {"source": source, "load_resistance_ohm": float(load_r), "c_scale": float(c_scale), "noise_pct": float(noise), "duration_s": 0.1}
        return params, None, {"Data origin": "Simulation: identified VSC plant + DPC (datagen.run_scenario)", "C scale (truth)": c_scale, "Noise added": f"{noise:g} %"}
    up = st.file_uploader("Upload a VSC run (CSV)", type=["csv"], key="pid_upload", help=f"At most {MAX_UPLOAD_MB} MB. Rows are samples at Ts = 100 µs; timestamp_s optional.")
    if up is None:
        return {"source": source, "load_resistance_ohm": float(load_r)}, None, None
    raw = up.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is {len(raw) / 1e6:.1f} MB — the limit is {MAX_UPLOAD_MB} MB.")
        return {"source": source}, None, None
    try:
        df = pd.read_csv(up)
    except Exception as exc:  # noqa: BLE001 -- any parse error is reported, the file is data only
        st.error(f"Could not parse '{up.name}': {exc}")
        return {"source": source}, None, None
    sha = hashlib.sha256(raw).hexdigest()
    return ({"source": source, "load_resistance_ohm": float(load_r), "noise_pct": 0.0, "sha256": sha}, df,
            {"Data origin": f"Uploaded file: {up.name}", "SHA-256": sha})


def _window_and_estimator(max_s: float) -> dict:
    c1, c2 = st.columns(2)
    start = c1.number_input("Window start (s)", 0.0002, max(0.0002, max_s - 0.002), min(0.06, max(0.0002, max_s - 0.02)), 0.005, format="%.4f", key="pid_start",
                            help="Leave the start-up transient out; the row before the window gives the initial state.")
    length = c2.select_slider("Window length (ms)", [5, 10, 20, 40], 20, key="pid_length")
    c1, c2 = st.columns(2)
    est = c1.radio("Estimator", idn.ESTIMATORS, format_func=ia.ESTIMATOR_LABELS.get, key="pid_estimator")
    n_boot = c2.number_input("Bootstrap refits", 2, 200 if est == "least_squares" else 20, 5 if est == "pinn" else 50, 1, key=f"pid_nboot_{est}",
                             help="Each refit uses a moving-block resample of the window; the interval is their 5–95 % percentile. PINN refits take ~2–3 s each, least-squares refits ~0.3 s.")
    return {"start_s": float(start), "length_s": float(length) / 1e3, "estimator": est, "n_bootstrap": int(n_boot)}


def _estimate_figure(result: dict, truth: dict | None) -> go.Figure:
    names = list(idn.PARAMS)
    ratio = [result["params"][p]["estimate"] / result["params"][p]["nominal"] for p in names]
    lo = [result["params"][p]["low"] / result["params"][p]["nominal"] for p in names]
    hi = [result["params"][p]["high"] / result["params"][p]["nominal"] for p in names]
    fig = go.Figure(go.Scatter(x=names, y=ratio, mode="markers", name="estimate / nominal", marker=dict(size=11, color=series_color("candidate")),
                               error_y=dict(type="data", symmetric=False, array=[h - r for h, r in zip(hi, ratio)], arrayminus=[r - l for r, l in zip(ratio, lo)], thickness=2, width=8)))
    if truth is not None:
        fig.add_trace(go.Scatter(x=names, y=[truth[p] / result["params"][p]["nominal"] for p in names], mode="markers", name="simulated true value",
                                 marker=dict(size=11, symbol="diamond-open", color=series_color("reference"), line=dict(width=2))))
    fig.add_hline(y=1.0, line_dash="dot", line_color=series_color("base"))
    style_figure(fig, height=300, x_title="Parameter", y_title="Value / nominal")
    return fig


def _display_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in table.iterrows():
        row = {"Parameter": f"{r['Parameter']} ({r['Unit']})", "Nominal": f"{r['Nominal']:.4g}", "Estimate": f"{r['Estimate']:.4g}",
               "5–95 % interval": f"{r['Interval low (5 %)']:.4g} – {r['Interval high (95 %)']:.4g}", "Δ vs nominal": f"{r['Δ vs nominal']:+.1%}"}
        if "Simulated true value" in table.columns:
            row["True (sim.)"] = f"{r['Simulated true value']:.4g}"
            row["Error vs true"] = f"{r['Error vs true']:+.1%}"
        rows.append(row)
    return pd.DataFrame(rows)


def render_tab() -> None:
    st.html('<div class="spade-root spade-badges">' + badge_html("PARAMETER ESTIMATE", "warning", "▲", idn.DISCLAIMER) + "</div>")
    st.warning(f"**{idn.DISCLAIMER}** Nominal values are the identified model's (sim/vsc_params.py), not ground truth for a real converter. This tab has its own data controls; the scenario controls on the left do not apply here.", icon=":material/science:")
    with st.container(border=True):
        col_data, col_fit = st.columns(2, gap="large")
        with col_data:
            st.markdown("**Data**")
            params, df, provenance = _source_controls()
        if params["source"] == "PLECS" or (params["source"] != "Simulated run" and df is None):
            return
        with col_fit:
            st.markdown("**Window and estimator**")
            max_s = params.get("duration_s") or (float(df["timestamp_s"].iloc[-1]) if "timestamp_s" in df.columns else len(df) * 1e-4)
            params.update(_window_and_estimator(max_s))
            current = st.session_state.get("pid_result")
            unchanged = current is not None and current["run"]["config_hash"] == config_hash(params)
            clicked = st.button("Identify parameters", type="primary", key="pid_run", width="stretch", disabled=unchanged, icon=":material/query_stats:")
    view = st.container()
    if clicked:
        with view, st.spinner("Simulating and fitting…" if params["source"] == "Simulated run" else "Fitting…"):
            try:
                if params["source"] == "Simulated run":
                    df = ia.simulate(params["c_scale"], params["load_resistance_ohm"], params["duration_s"])
                result = ia.run_identification(df, params["load_resistance_ohm"], params["start_s"], params["length_s"], params["estimator"], params["n_bootstrap"],
                                               params.get("noise_pct", 0.0))
            except ValueError as exc:
                st.error(f"Cannot identify on this input: {exc}", icon=":material/error:")
                return
        truth = ia.true_parameters(params["c_scale"]) if params["source"] == "Simulated run" else None
        summary = {p: round(v["estimate"], 9) for p, v in result["params"].items()}
        st.session_state["pid_result"] = {"result": result, "params": params, "truth": truth, "provenance": provenance, "run": record_run("vsc_identification", params, summary)}
        st.rerun()
    with view:
        stored = st.session_state.get("pid_result")
        if stored is None:
            st.info("Choose the data and the window above, then click **Identify parameters**.", icon=":material/info:")
            return
        result, p, truth, run = stored["result"], stored["params"], stored["truth"], stored["run"]
        if run["config_hash"] != config_hash(params):
            st.warning(f"Controls changed since run {run['run_id']} — the results still show that run.", icon=":material/history:")
        st.html('<div class="spade-root spade-badges">' + (source_badge_html("simulated") if p["source"] == "Simulated run" else source_badge_html("external", "Uploaded file -- its origin is not verified by SPADE")) + "</div>")
        c_fig, c_tab = st.columns([2, 5], gap="medium")
        with c_fig:
            st.plotly_chart(_estimate_figure(result, truth), width="stretch", config=PLOTLY_CONFIG, key="pid_fig")
        table = ia.results_table(result, truth)
        with c_tab:
            st.dataframe(_display_table(table), hide_index=True, width="stretch")
            with st.expander("Full precision", icon=":material/table:"):
                st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Estimator: {ia.ESTIMATOR_LABELS[result['estimator']]}. Uncertainty: **{result['uncertainty_method']}** over {result['n_bootstrap']} refits"
                   f"{' (each with a new network seed)' if result['estimator'] == 'pinn' else ''}; with few refits the interval is coarse. "
                   f"Window: {result['window_samples']} samples; R = {result['load_resistance_ohm']:g} Ω (given). Time: {result['elapsed_s']:.1f} s.")
        val = ia.validation_summary(result["estimator"], p.get("noise_pct", 0.0))
        with st.expander("Validation of this estimator (simulation)", icon=":material/fact_check:"):
            if val is None:
                st.caption("No validation report found (docs/audit/ai/pinn_identification.json).")
            else:
                st.dataframe(val, hide_index=True, width="stretch", column_config={"Interval coverage": st.column_config.NumberColumn(format="percent")})
                st.caption("Recovery error on simulated runs with known parameters (docs/audit/ai/pinn_identification.md), at the closest evaluated noise level. Same model as the estimator assumes; not evidence for a real converter.")
        st.caption("Longitudinal trend of C: not available — it needs comparable windows from several runs of the same converter.")
        provenance_table({"Run ID": run["run_id"], **(stored["provenance"] or {}), "Configuration hash": run["config_hash"], "Timestamp (UTC)": run["utc"]}, "Provenance")
