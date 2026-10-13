"""PMSM / FOC page: the stand-alone FOC + MTPA control-law demo (legacy Fase A tab 03).

Scope stated on the page itself (AUDIT AI-09 / Q28): this salient PMSM has NO fault model, no
dataset in datagen, no classifier and no registry entry -- it is a control comparison, not part of
the diagnosis pipeline. The motor illustration is therefore not shown here."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from driveflow.control.classical import generate_mtpa_vs_naive_cloud
from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS as PMSM_DEFAULTS
from driveflow.viz.adapters.session import config_hash, record_run
from driveflow.viz.components.inputs import slider_with_custom
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.components.presets import apply_pending_profile_load, profile_manager
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure

I_LIM = 400.0


def _dq_figure(data: dict) -> go.Figure:
    fig = go.Figure()
    cx, cy = zip(*data["limit_circle"])
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=f"current limit ({I_LIM:.0f} A)", line=dict(color=series_color("reference"), dash="dot", width=1.3)))
    mx, my = zip(*data["mtpa_curve"])
    fig.add_trace(go.Scatter(x=mx, y=my, mode="lines", name="MTPA locus (analytic)", line=dict(color=series_color("mechanical"), width=2.2)))
    for key, label, role in [("mtpa", "FOC + MTPA", "electrical"), ("naive", "FOC naive (i_d = 0)", "vibration")]:
        xs, ys = zip(*data[key])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", name=label, marker=dict(size=4, color=series_color(role), opacity=0.55)))
    style_figure(fig, height=560, x_title="i_d (A)", y_title="i_q (A)")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def render() -> None:
    page_header(
        "Simulation / PMSM",
        "PMSM field-oriented control: MTPA vs. naive",
        "A salient permanent-magnet synchronous motor under native dq-frame FOC, comparing maximum-torque-per-ampere with i_d = 0 at equal current magnitude.",
        [source_badge_html("simulated"), badge_html("CONTROL DEMO — NO DIAGNOSIS", "warning", "▲", "No fault model, dataset, classifier or registry entry exists for this PMSM.")],
    )
    apply_pending_profile_load()
    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl:
        with st.container(border=True):
            st.markdown("**Point-cloud density**")
            n_i_s = slider_with_custom(st, "Current-magnitude steps", 3, 25, 18, key="pmsm_n_steps", help="More steps = more points along the MTPA curve.")
            subsample = slider_with_custom(st, "Trajectory sample density", 1, 15, 3, key="pmsm_subsample", help="Lower = more points kept per transient.")
        with st.container(border=True):
            st.markdown("**Motor characteristics**")
            p = st.number_input("Pole pairs p", min_value=1, value=int(PMSM_DEFAULTS["p"]), step=1, key="pmsm_p")
            l_d = st.number_input("d-axis inductance l_d (H)", min_value=1e-6, value=float(PMSM_DEFAULTS["l_d"]), format="%.6f", key="pmsm_l_d")
            l_q = st.number_input("q-axis inductance l_q (H)", min_value=1e-6, value=float(PMSM_DEFAULTS["l_q"]), format="%.6f", key="pmsm_l_q", help="MTPA differs from i_d = 0 only when l_d ≠ l_q (salient motor).")
            r_s = st.number_input("Stator resistance r_s (Ω)", min_value=0.001, value=float(PMSM_DEFAULTS["r_s"]), format="%.4f", key="pmsm_r_s")
            psi_p = st.number_input("PM flux ψ_p (Wb)", min_value=0.001, value=float(PMSM_DEFAULTS["psi_p"]), format="%.4f", key="pmsm_psi_p")
            with st.expander("Motor profiles"):
                profile_manager(st, group="pmsm", current_values=dict(p=p, l_d=l_d, l_q=l_q, r_s=r_s, psi_p=psi_p))
        params = {"p": int(p), "l_d": l_d, "l_q": l_q, "r_s": r_s, "psi_p": psi_p, "n_i_s": int(n_i_s), "subsample": int(subsample)}
        current = st.session_state.get("pmsm_result")
        unchanged = current is not None and current["run"]["config_hash"] == config_hash(params)
        clicked = st.button("Generate PMSM cloud", type="primary", key="pmsm_run_button", width="stretch", disabled=unchanged, icon=":material/play_arrow:")
    if clicked:
        with view, st.spinner("Simulating current steps…"):
            data = generate_mtpa_vs_naive_cloud(i_s_values=list(np.linspace(40, 390, params["n_i_s"])), mp=dict(p=p, l_d=l_d, l_q=l_q, r_s=r_s, psi_p=psi_p), subsample=params["subsample"])
        st.session_state["pmsm_result"] = {"data": data, "params": params, "run": record_run("pmsm", params, {"steps": params["n_i_s"]})}
        st.rerun()
    with view:
        result = st.session_state.get("pmsm_result")
        if result is None:
            st.info("Set the motor and density on the left and click **Generate PMSM cloud**.", icon=":material/info:")
            return
        if result["run"]["config_hash"] != config_hash(params):
            st.warning(f"Controls changed since run {result['run']['run_id']} — the chart still shows that run.", icon=":material/history:")
        st.markdown("##### i_d–i_q plane")
        st.plotly_chart(_dq_figure(result["data"]), width="stretch", config=PLOTLY_CONFIG, key="pmsm_dq")
        st.caption("At equal current magnitude MTPA yields more torque than i_d = 0 by exploiting reluctance torque (l_d ≠ l_q). Closed-form locus in control/classical/pmsm_foc.py, verified against GEM's PermanentMagnetSynchronousMotor._torque_limit().")
        provenance_table({"Run ID": result["run"]["run_id"], "Data origin": "Native FOC simulation (control/classical/pmsm_foc.py)", "Configuration hash": result["run"]["config_hash"], "Timestamp (UTC)": result["run"]["utc"], "Diagnosis coverage": "none — not part of datagen / classifiers / registry"})
