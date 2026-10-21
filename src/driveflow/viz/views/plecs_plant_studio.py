"""Simulation / PLECS Plant Studio (spec Sec. 3, 3.14; expansion task E2: the batch routes).

Tabs: My Plants | Create Plant | Compare with SPADE | Connection & Diagnostics. In this version only
the Dataset route works ("Generate simulation datasets": PLECS Standalone RPC sweeps run as jobs by
the worker). The Gymnasium card is visible but blocked -- interactive stepping needs a verified
Co-Simulation FMU (task E3/E4) and is never derived from RPC connectivity -- and the surrogate card
and the comparison arrive with task E5. No chart is drawn without a completed sweep behind it.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.integrations.plecs import batch
from driveflow.viz.adapters import jobs_adapter as ja
from driveflow.viz.adapters import plecs_adapter as pa
from driveflow.viz.components.page_header import badge_html, page_header, provenance_table, source_badge_html
from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure

NOT_AVAILABLE_GYM = "Interactive stepping is not available with the detected backend."
LOCAL_RPC = "PLECS RPC is available only when SPADE's backend runs on the same computer."
_PALETTE = ["#38BDF8", "#F59E0B", "#A78BFA", "#F472B6", "#22C55E", "#94A3B8"]


def _url() -> str:
    return st.session_state.get("plecs_url", pa.DEFAULT_URL)


# ----------------------------------------------------------------------------- My Plants
def _my_plants() -> None:
    search = st.text_input("Search plants", key="plecs_search", placeholder="name or ID")
    rows = pa.profile_rows(search)
    if rows.empty:
        st.info("No PLECS plant profiles yet. Create your first plant to get started." if not search else "No plant matches this search.", icon=":material/info:")
        return
    st.dataframe(rows, hide_index=True, width="stretch")
    for _, r in rows[rows["Readiness"] == "Stale"].iterrows():
        st.warning(f"**{r['Name']}** ({r['ID']}): This profile changed since its last verification. Revalidation is required.", icon=":material/sync_problem:")
    st.caption("Capabilities come from what the backend has shown: 'Batch sweep: Shown' cites the sweep run that completed. "
               "RPC profiles never offer interactive stepping (spec Sec. 3.15). Open a profile from Create Plant → Generate simulation datasets.")


# ----------------------------------------------------------------------------- Create Plant
def _goal_cards() -> str | None:
    st.markdown("#### What would you like to do with your PLECS circuit?")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**Use PLECS as a Gymnasium plant**")
        st.caption("Recommended for closed-loop RL · Train or evaluate a controller that interacts with PLECS at each control interval.")
        st.button("Continue", key="plecs_goal_gym", disabled=True, width="stretch")
        st.caption(f":material/block: {NOT_AVAILABLE_GYM} It needs a verified Co-Simulation FMU; an RPC connection is not enough.")
    with c2, st.container(border=True):
        st.markdown("**Generate simulation datasets**")
        st.caption("Grid or Latin-hypercube sweeps of the circuit's ModelVars; the control lives inside the circuit. Produces a dataset, not an interactive environment.")
        if st.button("Continue", key="plecs_goal_dataset", type="primary", width="stretch"):
            st.session_state["plecs_goal"] = "dataset"
    with c3, st.container(border=True):
        st.markdown("**Build a fast surrogate plant**")
        st.caption("Fit a surrogate to PLECS sweeps and use it as a fast environment, with its error and validity domain shown.")
        st.button("Continue", key="plecs_goal_surrogate", disabled=True, width="stretch")
        st.caption(":material/block: Not available in this version.")
    return st.session_state.get("plecs_goal")


def _import_circuit() -> None:
    with st.expander("Import a circuit (new profile or new version)", expanded=not pa.profiles(), icon=":material/upload_file:"):
        c1, c2 = st.columns(2)
        pid = c1.text_input("Profile ID", key="plecs_new_id", placeholder="e.g. buck_48v", help="a-z, 0-9, '_' or '-'. Re-using an ID creates a new version.")
        name = c2.text_input("Display name", key="plecs_new_name")
        how = st.radio("Circuit", ["Upload .plecs", "File on this computer"], key="plecs_new_how", horizontal=True,
                       help="A circuit that references other files by relative path should be selected in place.")
        uploaded, local = None, None
        if how == "Upload .plecs":
            f = st.file_uploader("Circuit file", type=["plecs"], key="plecs_new_file")
            if f is not None:
                uploaded = (f.getvalue(), f.name)
        else:
            local = st.text_input("Path of the .plecs file on the computer that runs SPADE's backend", key="plecs_new_path")
        st.markdown("**Interface manifest** — declared by hand; SPADE does not read the circuit's blocks.")
        st.caption("`model_vars`: the ModelVars a sweep may set (unit, default, range). `outputs`: the top-level outports in port order (unit).")
        st.html('<div class="spade-root spade-badges">' + badge_html("EXAMPLE CONFIGURATION — NOT EXECUTED", "unknown", "◇", "Template from the spec's buck example; edit it to match your circuit") + "</div>")
        tpl = st.selectbox("Template", list(pa.TEMPLATES), key="plecs_new_template", help="A starting point for the manifest; edit it to match your circuit.")
        if "DAB" in tpl:
            st.caption("DAB template from Fund.'s DAB_sample1.plecs (licence unclear: non-commercial use; the file is not shipped with SPADE). "
                       "That circuit has no top-level outports: add them in PLECS before a sweep can return signals.")
        if st.session_state.get("plecs_new_template_last") != tpl:
            st.session_state["plecs_new_manifest"] = json.dumps(pa.TEMPLATES[tpl], indent=2)
            st.session_state["plecs_new_template_last"] = tpl
        text = st.text_area("Manifest (JSON)", height=300, key="plecs_new_manifest")
        if st.button("Import circuit", key="plecs_import", icon=":material/add:"):
            try:
                p = pa.create_profile(pid.strip(), name.strip(), text, _url(), uploaded, local)
            except ValueError as exc:
                st.error(f"Not imported: {exc}", icon=":material/error:")
            else:
                st.session_state["plecs_selected"] = p.profile_id
                st.success(f"Imported **{p.display_name}** as `{p.profile_id}` v{p.version} (SHA-256 {p.file_hash[:12]}…).", icon=":material/check_circle:")


def _sweep_form(p) -> None:
    man = pa.manifest_of(p)
    stale = pa.readiness(p) == "Stale"
    if stale:
        st.warning("The circuit or mapping changed. Re-run verification. Import the circuit again to create a new version.", icon=":material/sync_problem:")
    if not man.model_vars:
        st.info("This manifest declares no ModelVars, so there is nothing to sweep.", icon=":material/info:")
        return
    names = [v.name for v in man.model_vars]
    swept = st.multiselect("Parameters to vary", names, default=names[:1], key=f"plecs_swept_{p.profile_id}")
    method = st.radio("Method", ["grid", "lhs"], format_func={"grid": "Grid", "lhs": "Latin hypercube"}.get, horizontal=True, key="plecs_method")
    ranges, fixed = {}, {}
    for v in man.model_vars:
        if v.name in swept:
            c1, c2, c3 = st.columns(3)
            lo = c1.number_input(f"{v.name} low ({v.unit})", v.minimum, v.maximum, v.minimum, key=f"plecs_lo_{p.profile_id}_{v.name}")
            hi = c2.number_input(f"{v.name} high ({v.unit})", v.minimum, v.maximum, v.maximum, key=f"plecs_hi_{p.profile_id}_{v.name}")
            levels = c3.number_input("Levels", 1, 50, 3, key=f"plecs_lv_{p.profile_id}_{v.name}", disabled=method != "grid")
            ranges[v.name] = (lo, hi, levels) if method == "grid" else (lo, hi)
        else:
            fixed[v.name] = st.number_input(f"{v.name} (fixed, {v.unit})", v.minimum, v.maximum, v.default, key=f"plecs_fx_{p.profile_id}_{v.name}")
    c1, c2, c3, c4 = st.columns(4)
    n_samples = c1.number_input("Samples (LHS)", 1, batch.MAX_CONDITIONS, 9, key="plecs_nsamp", disabled=method != "lhs")
    seed = c2.number_input("Seed", 0, 2**31 - 1, 0, key="plecs_seed")
    span = c3.number_input("Simulated time per run (s)", 1e-6, 1e4, 0.02, format="%.4g", key="plecs_span")
    limit = c4.number_input("Time limit per run (s)", 1.0, 3600.0, 60.0, key="plecs_limit", help="Passed to PLECS as SolverOpts.Timeout; a run over it is recorded as failed.")
    pts = st.number_input("Output points per run (0 = every solver step)", 0, 100_000, 201, key="plecs_pts")
    if not swept:
        st.caption("Choose at least one parameter to vary.")
        return
    try:
        conds = pa.design(p, ranges, method, int(n_samples), int(seed), fixed)
    except ValueError as exc:
        st.error(f"Invalid sweep: {exc}", icon=":material/error:")
        return
    st.caption(f"Budget: **{len(conds)} condition(s)** × up to {limit:g} s each. ETA unavailable until comparable runs exist. Engine: one PLECS run at a time.")
    with st.expander(f"Conditions ({len(conds)})", icon=":material/table:"):
        st.dataframe(pd.DataFrame(conds), hide_index=True, width="stretch")
    workers = ja.live_workers()
    if not workers:
        st.caption(":material/warning: No worker is running: the sweep will wait in the queue until one starts (Experiment Runs → Start local worker).")
    if st.button("Start sweep", type="primary", key="plecs_start", icon=":material/play_arrow:", disabled=stale):
        job = pa.submit_sweep(p, conds, method, int(seed), float(span), float(limit), int(pts) or None)
        st.success(f"Queued as job `{job.job_id}`. Follow it in Experiment Runs; results appear below when it finishes.", icon=":material/schedule:")


def _provenance(run, p) -> None:
    provenance_table({"Run ID": run.run_id, "Origin": run.origin, "Profile": f"{p.profile_id} v{p.version}", "Circuit": p.parameters.get("circuit_name"),
                      "Circuit SHA-256": p.file_hash, "Seed": run.seed, "Configuration hash": run.config_hash, "Completed (UTC)": run.completed_utc,
                      "Reproduce": run.command})


def _sweep_results(p):
    for j in pa.active_sweeps(p.profile_id):
        prog = f" ({j['done']}/{j['total']})" if j["total"] else ""
        st.caption(f":material/schedule: Sweep job `{j['job_id']}`: {j['state']}{prog}.")
    st.button("Refresh results", key=f"plecs_refresh_{p.profile_id}", icon=":material/refresh:")
    runs = pa.sweep_runs(p.profile_id)
    if not runs:
        st.info("No simulation has been executed for this configuration.", icon=":material/info:")
        return None
    run = runs[0] if len(runs) == 1 else next(r for r in runs if r.run_id == st.selectbox("Sweep run", [r.run_id for r in runs], key=f"plecs_run_{p.profile_id}"))
    ok = [c for c in run.conditions if c["status"] == "ok"]
    failed = [c for c in run.conditions if c["status"] == "failed"]
    st.html('<div class="spade-root spade-badges">' + source_badge_html("SIMULATED · PLECS", "Simulated in PLECS; not hardware") + badge_html(f"{run.status.upper()}", "warning" if failed else "healthy") + "</div>")
    st.caption(f"Conditions: **{len(ok)} completed / {len(failed)} failed** of {len(run.conditions)}. Failed conditions are listed, never interpolated. {batch.NOT_GYM}")
    man = pa.manifest_of(p)
    if ok:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            sig = st.selectbox("Signal", man.output_names, key=f"plecs_sig_{p.profile_id}", format_func=lambda n: f"{n} ({next(o.unit for o in man.outputs if o.name == n)})")
            fig = go.Figure()
            for k, c in enumerate(ok[:12]):
                df = pa.condition_frame(run, c["index"])
                if df is not None:
                    label = ", ".join(f"{a}={b:g}" for a, b in c["condition"].items())
                    fig.add_trace(go.Scatter(x=df["timestamp_s"], y=df[sig], mode="lines", name=label, line=dict(width=1.6, color=_PALETTE[k % len(_PALETTE)])))
            style_figure(fig, height=320, x_title="Time (s)", y_title=sig)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"plecs_curves_{run.run_id}")
            if len(ok) > 12:
                st.caption(f"First 12 of {len(ok)} completed conditions shown.")
        with c2:
            walls = [c.get("seconds") for c in ok if c.get("seconds") is not None]
            fig = go.Figure(go.Histogram(x=walls, marker_color=series_color("base")))
            style_figure(fig, height=320, x_title="Wall time per run (s)", y_title="Runs", legend=False)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"plecs_walls_{run.run_id}")
    st.dataframe(pa.conditions_table(run), hide_index=True, width="stretch")
    if failed:
        with st.expander(f"Failures and logs ({len(failed)})", icon=":material/error:"):
            for c in failed:
                st.caption(f"#{c['index']} {c['condition']}: {c.get('error')}")
    c1, c2 = st.columns(2)
    c1.download_button("Export dataset (zip)", pa.dataset_zip(run), file_name=f"plecs_{p.profile_id}_{run.run_id}.zip", key=f"plecs_zip_{run.run_id}", icon=":material/download:")
    man_text = pa.artifact_text(run, "manifest.json")
    if man_text:
        c2.download_button("Export manifest", man_text, file_name="manifest.json", key=f"plecs_man_{run.run_id}", icon=":material/description:")
    return run


def _create_plant() -> None:
    goal = _goal_cards()
    if goal != "dataset":
        return
    st.divider()
    st.markdown("#### Generate simulation datasets")
    st.caption("Select circuit → Select variables → Choose sweep method → Review budget → Start sweep → Inspect results → Export dataset")
    _import_circuit()
    plist = pa.profiles()
    batch_profiles = [p for p in plist if p.kind == batch.KIND]
    if not batch_profiles:
        st.info("No circuit has been imported yet.", icon=":material/info:")
        return
    ids = [p.profile_id for p in batch_profiles]
    default = ids.index(st.session_state["plecs_selected"]) if st.session_state.get("plecs_selected") in ids else 0
    pid = st.selectbox("Circuit profile", ids, index=default, key="plecs_profile", format_func=lambda i: f"{next(p.display_name for p in batch_profiles if p.profile_id == i)} ({i})")
    p = pa.get_profile(pid)
    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("**Circuit and sweep**")
        _sweep_form(p)
    with right, st.container(border=True):
        st.markdown("**Results**")
        run = _sweep_results(p)
    if run is not None:
        _provenance(run, p)


# ----------------------------------------------------------------------------- Compare with SPADE
def _compare() -> None:
    from driveflow.viz.adapters import surrogate_adapter as sa

    st.markdown("#### Native SPADE buck vs. surrogate")
    sur, run = sa.load()
    if sur is None:
        st.info("No surrogate has been fitted yet (experiments/fit_buck_surrogate.py).", icon=":material/info:")
    else:
        st.html('<div class="spade-root spade-badges">' + source_badge_html("SIMULATED", "Native averaged buck") + source_badge_html("SIMULATED · SURROGATE", f"Fitted surrogate {run}") + "</div>")
        st.caption(f"Surrogate run {run}, fitted to SPADE's own averaged buck — it checks the surrogate mechanics, not fidelity to a PLECS circuit. "
                   f"Validated range: V_in {sur['domain']['v_in_v']} V, R {sur['domain']['load_resistance_ohm']} Ω. Both plants are driven by the same PI with the same reference and load step.")
        c1, c2, c3, c4, c5 = st.columns(5)
        family = c1.selectbox("Surrogate family", ["zoh", "mlp"], index=["zoh", "mlp"].index(sur["default_family"]), key="cmp_family",
                              format_func=lambda f: {"zoh": "ZOH grey box", "mlp": "MLP"}[f] + (" (default)" if f == sur["default_family"] else ""))
        vin = c2.number_input("V_in (V)", 5.0, 120.0, 48.0, 1.0, key="cmp_vin")
        r = c3.number_input("Load R (Ω)", 0.5, 200.0, 12.0, 0.5, key="cmp_r")
        vref = c4.number_input("v_ref (V)", 1.0, 119.0, 24.0, 1.0, key="cmp_vref")
        step = c5.number_input("Load step to (Ω), 0 = none", 0.0, 200.0, 6.0, 0.5, key="cmp_step")
        if vref >= vin:
            st.error("v_ref must be below V_in for a buck.", icon=":material/error:")
        elif st.button("Run comparison", type="primary", key="cmp_run", icon=":material/compare:"):
            st.session_state["cmp_result"] = sa.compare(sur, family, float(vin), float(r), float(vref), float(step) or None)
        res = st.session_state.get("cmp_result")
        if res is None:
            st.info("No comparison has been generated yet. Select compatible plants and run the comparison.", icon=":material/info:")
        else:
            if res["out_of_domain"]:
                st.warning(f"**Surrogate outside validated operating range**: {'; '.join(res['out_of_domain'])}. The comparison is shown, but it does not claim equivalence.", icon=":material/warning:")
            if res["status"] == "failed":
                st.error("Failed: the surrogate trajectory became non-finite.", icon=":material/error:")
            else:
                st.caption(f"{res['family']}: v_out RMSE {res['rmse_v_out_v']:.4f} V, max |Δv_out| {res['max_abs_v_out_v']:.4f} V, i_L RMSE {res['rmse_i_l_a']:.4f} A over 20 ms.")
            df = sa.overlay_frame(res)
            g1, g2 = st.columns(2)
            for col, ch, unit in ((g1, "v_out", "V"), (g2, "i_l", "A")):
                fig = go.Figure(go.Scatter(x=df["t_ms"], y=df[f"native_{ch}"], mode="lines", name="native (SIMULATED)", line=dict(width=2, color=series_color("base"))))
                fig.add_trace(go.Scatter(x=df["t_ms"], y=df[f"surrogate_{ch}"], mode="lines", name="surrogate", line=dict(width=1.6, dash="dash", color=series_color("candidate"))))
                style_figure(fig, height=280, x_title="Time (ms)", y_title=f"{ch} ({unit})")
                col.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"cmp_{ch}")
        with st.expander("Error per condition (validation, test and out-of-range conditions of the fit)", icon=":material/table:"):
            if st.button("Run the condition grid", key="cmp_grid"):
                st.session_state["cmp_grid_table"] = (family, sa.grid_table(sur, family))
            g = st.session_state.get("cmp_grid_table")
            if g is not None and g[0] == family:
                st.dataframe(g[1], hide_index=True, width="stretch")
                st.caption("Reference 0.5·V_in, load halved at 10 ms. Failed and out-of-range rows are listed, not hidden.")
    st.divider()
    st.markdown("#### PLECS circuit vs. SPADE")
    st.info("No comparison has been generated yet. Select compatible plants and run the comparison.", icon=":material/info:")
    st.caption("Needs PLECS simulation data of a circuit (PLECS Plant Studio → Generate simulation datasets on a machine with PLECS). None exists yet.")


# ----------------------------------------------------------------------------- Diagnostics
def _diagnostics() -> None:
    url = st.text_input("PLECS RPC endpoint", pa.DEFAULT_URL, key="plecs_url", help=LOCAL_RPC)
    st.caption(f"{LOCAL_RPC} RPC is disabled by default in PLECS: enable it in the preferences. 'PLECS available' means the tool answers — it is not telemetry.")
    if st.button("Test connection", key="plecs_test", icon=":material/lan:"):
        st.session_state["plecs_report"] = pa.test_connection(url)
    rep = st.session_state.get("plecs_report")
    sections = st.columns(5)
    with sections[0], st.container(border=True):
        st.markdown("**RPC connectivity**")
        if rep is None:
            st.caption("Not tested")
        elif rep["reachable"]:
            st.success(f"PLECS available (RPC answered in {rep['latency_ms']:.0f} ms)", icon=":material/check_circle:")
        else:
            st.error(rep["message"], icon=":material/error:")
            st.caption(rep.get("hint", ""))
    with sections[1], st.container(border=True):
        st.markdown("**FMU import support**")
        st.caption("Not supported in this version: FMI import needs the FMU spike (task E3).")
    with sections[2], st.container(border=True):
        st.markdown("**Step capability**")
        st.caption(f"Not supported. {NOT_AVAILABLE_GYM}")
    with sections[3], st.container(border=True):
        st.markdown("**Circuit readiness**")
        plist = pa.profiles()
        stale = sum(pa.readiness(p) == "Stale" for p in plist)
        st.caption(f"{len(plist)} profile(s); {stale} stale." if plist else "No circuit has been imported yet.")
    with sections[4], st.container(border=True):
        st.markdown("**Environment readiness**")
        st.caption("No PLECS Gymnasium environment. Not derived from RPC connectivity.")
    if rep is not None:
        with st.expander("Report details", icon=":material/description:"):
            st.caption(f"PLECS version: {rep['plecs_version']}. Checked: {', '.join(rep['checked'])}. Not checked: {', '.join(rep['not_checked']) or '—'}.")
            if rep.get("methods"):
                st.caption("Methods listed by the server: " + ", ".join(rep["methods"]))
            st.code(pa.diagnostics_json(rep), language="json")
            st.download_button("Save report", pa.diagnostics_json(rep), file_name="plecs_diagnostics.json", key="plecs_save_report", icon=":material/download:")


def render() -> None:
    page_header("Simulation / PLECS Plant Studio", "PLECS Plant Studio",
                "Create an environment from a PLECS circuit. In this version: simulation datasets from PLECS Standalone RPC sweeps. "
                "Interactive (Gymnasium) plants need a verified Co-Simulation FMU and are not available yet.",
                [source_badge_html("SIMULATED · PLECS", "Results are PLECS simulations, not hardware"), badge_html("DATASET ROUTE ONLY", "warning", "▲", NOT_AVAILABLE_GYM)])
    t_plants, t_create, t_compare, t_diag = st.tabs(["My Plants", "Create Plant", "Compare with SPADE", "Connection & Diagnostics"])
    with t_plants:
        _my_plants()
    with t_create:
        _create_plant()
    with t_compare:
        _compare()
    with t_diag:
        _diagnostics()
