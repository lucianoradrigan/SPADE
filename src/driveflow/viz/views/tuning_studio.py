"""Design & Control / Tuning Studio (spec Sec. 4; expansion task E7).

Five steps on one page: Select system → Search space → Objectives & method → Execute → Review &
approve. The optimisation runs as a job in the worker (driveflow.jobs, kind controller_tuning). Every
chart comes from a finished run. "Save candidate" stores the parameters next to the run and changes
nothing; "Use as default" goes through ApprovalGate, which this round blocks (decision D1).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.optim import objectives as ob
from driveflow.viz.adapters import jobs_adapter as ja
from driveflow.viz.adapters import tuning_adapter as ta
from driveflow.viz.components.approval_gate import approval_gate
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure


def _step(n: int, title: str):
    c = st.container(border=True)
    c.markdown(f"**{n} · {title}**")
    return c


def _setup() -> dict:
    with _step(1, "Select system"):
        options = ta.available_systems()
        key = st.selectbox("Plant and controller", list(options), format_func=options.get, key="tun_system",
                           help="Listed by capability: the DC-motor speed loop, and the buck PI on every available environment whose reference controller is that PI.")
        sysm = ta.get_system(key)
        st.caption(f"Controller: {sysm.controller_label}. " + ("First case: S-04, the ≈15 % overshoot of the PI on small speed steps. The current loop is not tuned."
                                                             if key == "dc_motor_pi" else (sysm.note or "")))
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("Tuning grid (seen by the optimizer)")
            st.dataframe(ta.grid_table(sysm.tuning_grid), hide_index=True, width="stretch")
        with g2:
            st.markdown("Evaluation grid (fixed before tuning, never optimized on)")
            st.dataframe(ta.grid_table(sysm.evaluation_grid), hide_index=True, width="stretch")
        if sysm.grids_overlap():
            st.warning(ob.REUSED_WARNING, icon=":material/warning:")
    with _step(2, "Search space"):
        bounds = []
        cols = st.columns(len(sysm.params))
        for col, p, (lo, hi) in zip(cols, sysm.params, sysm.bounds):
            label, unit = ta.PARAM_LABELS[p]
            r = col.slider(f"{label} ({unit})", float(lo), float(hi), (float(lo), float(hi)), key=f"tun_b_{p}" if key == "dc_motor_pi" else f"tun_b_{key}_{p}")
            bounds.append(list(r))
            col.caption(f"Baseline {sysm.baseline[p]:g}")
    with _step(3, "Objectives & method"):
        methods = [m for m in ta.METHOD_LABELS if m in sysm.methods]
        method = st.radio("Method", methods, format_func=ta.METHOD_LABELS.get, horizontal=True, key="tun_method" if key == "dc_motor_pi" else f"tun_method_{key}")
        c1, c2 = st.columns(2)
        evals = c1.slider("Budget (objective evaluations)", 8, 200, 48 if key == "dc_motor_pi" else 24, 8, key="tun_evals" if key == "dc_motor_pi" else f"tun_evals_{key}",
                          help="Each evaluation simulates the tuning grid.")
        seed = c2.number_input("Seed", 0, 2**31 - 1, 0, key="tun_seed")
        st.caption(("Scalar objective (PSO, Bayesian): " + sysm.objective_text + ". NSGA-II keeps the three terms as separate objectives. Lower is better.")
                   if key == "dc_motor_pi" else "Scalar objective: " + sysm.objective_text)
    return {"method": method, "bounds": bounds, "evaluations": int(evals), "seed": int(seed), "system": key}


def _execute(cfg: dict) -> None:
    with _step(4, "Execute"):
        if any(hi - lo <= 0 for lo, hi in cfg["bounds"]):
            st.error("Each search range needs low < high.", icon=":material/error:")
            return
        if not ja.live_workers():
            st.caption(":material/warning: No worker is running: the job will wait in the queue (Experiment Runs → Start local worker).")
        if st.button("Start tuning", type="primary", key="tun_start", icon=":material/play_arrow:"):
            job = ta.submit(cfg["method"], cfg["bounds"], cfg["evaluations"], cfg["seed"], cfg["system"])
            st.success(f"Queued as job `{job.job_id}`.", icon=":material/schedule:")
        for j in ta.active_jobs():
            st.caption(f":material/schedule: Job `{j['job_id']}`: {j['state']}" + (f" ({j['done']}/{j['total']})" if j["total"] else ""))
        st.button("Refresh", key="tun_refresh", icon=":material/refresh:")


def _review() -> None:
    with _step(5, "Review & approve"):
        rs = ta.runs()
        if not rs:
            st.info("No simulation has been executed for this configuration.", icon=":material/info:")
            return
        run = next(r for r in rs if r.run_id == st.selectbox("Tuning run", [r.run_id for r in rs], key="tun_run",
                                                               format_func=lambda i: f"{i} · {next(r.metrics.get('method') for r in rs if r.run_id == i)}"))
        rep, hist = ta.report(run), ta.history(run)
        if rep is None or hist is None:
            st.warning(f"Run {run.run_id} ended as {run.status} without a tuning report. {run.failure_reason or ''}", icon=":material/error:")
            return
        st.html('<div class="spade-root spade-badges">' + source_badge_html("simulated") + badge_html("CANDIDATE — NO DEFAULT CHANGED", "warning", "▲") + "</div>")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("##### Best objective vs. evaluations")
            fig = go.Figure(go.Scatter(x=hist["evaluation"], y=hist["J"], mode="markers", name="evaluation", marker=dict(size=6, color=series_color("base"))))
            fig.add_trace(go.Scatter(x=hist["evaluation"], y=hist["best J so far"], mode="lines", name="best so far", line=dict(width=2, color=series_color("candidate"))))
            fig.add_hline(y=rep["baseline"]["tuning"]["J"], line_dash="dot", line_color=series_color("reference"), annotation_text="baseline")
            style_figure(fig, height=300, x_title="Objective evaluations", y_title="J on the tuning grid")
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"tun_hist_{run.run_id}")
        candidates = rep["candidates"]
        sysm_run = ta.get_system(ta.run_system_key(rep)) if ta.run_system_key(rep) in ta.available_systems() else None
        params = sysm_run.params if sysm_run else list(candidates[0]["params"])
        st.caption(f"System: {rep.get('subject', ta.SUBJECT)}.")
        sel_idx = 0
        with c2:
            if run.metrics.get("method") == "nsga2" and "overshoot %" in hist:
                st.markdown("##### Pareto front (click a point to select)")
                F = pd.DataFrame([c["objectives"] for c in candidates], columns=["overshoot %", "settling ms", "load dip rad/s"])
                fig = go.Figure(go.Scatter(x=hist["overshoot %"], y=hist["load dip rad/s"], mode="markers", name="evaluated", marker=dict(size=5, color=series_color("base"), opacity=0.4)))
                fig.add_trace(go.Scatter(x=F["overshoot %"], y=F["load dip rad/s"], mode="markers", name="Pareto front", customdata=list(range(len(F))),
                                         marker=dict(size=11, color=F["settling ms"], colorscale="Viridis", showscale=True, colorbar=dict(title="settling ms"))))
                style_figure(fig, height=300, x_title="Mean overshoot (%)", y_title="Mean load dip (rad/s)")
                ev = st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"tun_pareto_{run.run_id}", on_select="rerun", selection_mode="points")
                pts = [p for p in (ev.selection.points if ev and ev.selection else []) if p.get("curve_number") == 1]
                if pts:
                    sel_idx = int(pts[0].get("point_index", 0))
                sel_idx = st.selectbox("Candidate", range(len(candidates)), index=min(sel_idx, len(candidates) - 1), key=f"tun_cand_{run.run_id}",
                                       format_func=lambda k: f"{candidates[k]['label']}: overshoot {candidates[k]['objectives'][0]:.1f} %, settling {candidates[k]['objectives'][1]:.0f} ms, dip {candidates[k]['objectives'][2]:.2f} rad/s")
            else:
                st.markdown("##### Best candidate")
                st.dataframe(pd.DataFrame([{"Parameter": ta.PARAM_LABELS[p][0], "Baseline": rep["baseline"]["params"][p], "Candidate": candidates[0]["params"][p]} for p in params]),
                             hide_index=True, width="stretch")
        cand = candidates[sel_idx]
        st.markdown("##### Baseline vs. candidate on the evaluation grid")
        if rep.get("evaluation_reused_tuning_conditions"):
            st.warning(ob.REUSED_WARNING, icon=":material/warning:")
        better = cand["evaluation"]["J"] < rep["baseline"]["evaluation"]["J"]
        (st.success if better else st.warning)(f"Evaluation-grid J: baseline {rep['baseline']['evaluation']['J']:.2f} → candidate {cand['evaluation']['J']:.2f} "
                                               f"({'better' if better else 'not better'}; tuning-grid J {rep['baseline']['tuning']['J']:.2f} → {cand['tuning']['J']:.2f}).",
                                               icon=":material/compare_arrows:")
        st.dataframe(ta.comparison_table(rep, cand), hide_index=True, width="stretch")
        if ta.run_system_key(rep) == "dc_motor_pi":
            st.caption("The objective has no control-effort or noise term: a faster speed loop always lowers the load dip. Check the current and voltage margins before using any candidate.")
        else:
            st.caption("The objective has no load-step or noise term; check the peak current before using any candidate.")
        c1, c2 = st.columns(2)
        if c1.button("Save candidate", key=f"tun_save_{run.run_id}_{sel_idx}", icon=":material/bookmark:"):
            art = ta.save_candidate(run, cand)
            st.success(f"Saved next to the run ({art['path']}). No default was changed.", icon=":material/check:")
        with c2:
            st.markdown("**Use as default**")
            gate = ta.gate_report(rep, cand)
            approval_gate(rep.get("subject", ta.SUBJECT), run.run_id, gate, lambda who, decision: ta.record_decision(run, cand, gate, who, decision), key=f"tun_gate_{run.run_id}")
        provenance_table({"Run ID": run.run_id, "Method": run.metrics.get("method"), "Seed": run.seed, "Evaluations": run.metrics.get("n_evaluations"),
                          "Configuration hash": run.config_hash, "Reproduce": run.command, "Status": rep["status"]})


def render() -> None:
    page_header("Design & Control / Tuning Studio", "Tuning Studio",
                "Optimize controller parameters against simulated scenarios, then compare the candidate with the baseline on a separate evaluation grid. "
                "Results are candidates: nothing changes the default controller.",
                [source_badge_html("simulated"), badge_html("CANDIDATES ONLY", "warning", "▲", ta.NOT_AUTHORIZED)])
    cfg = _setup()
    _execute(cfg)
    _review()
