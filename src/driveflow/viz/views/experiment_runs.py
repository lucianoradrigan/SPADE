"""Workspace / Experiment Runs (spec Sec. 6 wireframe, Sec. 7; expansion task E0).

Everything on this page is read from the durable store (driveflow.runs / driveflow.jobs via
adapters/jobs_adapter.py): the job list, states, progress, logs and runs survive page changes,
browser reloads and Streamlit restarts. No example rows are fabricated -- an empty database shows an
empty state. Jobs execute in a separate worker process; this page only queues, observes and
requests cancellation.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from driveflow.viz.adapters import jobs_adapter as ja
from driveflow.viz.components.failure_panel import failure_panel
from driveflow.viz.components.job_card import job_card
from driveflow.viz.components.page_header import badge_html, page_header, status_badge_html
from driveflow.viz.components.provenance_footer import provenance_footer
from driveflow.viz.components.run_comparison import run_comparison
from driveflow.viz.components.source_badge import source_badge_html, source_badge_view

_PLANTS = {"DC motor (PI speed sweep)": ("dc_perm_ex_v1", "Speed references ω_ref [rad/s]", "100, 150, 200"),
           "VSC / DPC (voltage magnitude sweep)": ("vsc_dpc_v1", "Voltage references |v_ref| [V]", "30, 50")}


def _parse_values(text: str) -> list[float]:
    return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]


def _worker_panel() -> None:
    workers = ja.live_workers()
    if workers:
        st.caption(f":material/check_circle: {len(workers)} worker(s) alive (heartbeat within 15 s): " + ", ".join(f"{w['id']} · pid {w['pid']} · {w['state']}" for w in workers))
        return
    st.warning("No worker is running. Queued jobs wait until one starts; nothing executes inside this page.", icon=":material/warning:")
    c1, c2 = st.columns([1, 2])
    if c1.button("Start local worker", key="runs_start_worker", icon=":material/play_arrow:"):
        pid = ja.start_worker()
        st.session_state["runs_worker_started"] = pid
    c2.caption("Or start it yourself in a terminal (same workspace):")
    c2.code("python -m driveflow.jobs.worker serve", language="bash")
    if st.session_state.get("runs_worker_started"):
        st.caption(f"Worker process started (pid {st.session_state['runs_worker_started']}); it appears above after its first heartbeat. Output: workspace/worker.log")


def _new_experiment() -> None:
    with st.expander("New experiment — scenario sweep", icon=":material/add:"):
        label = st.selectbox("Plant", list(_PLANTS), key="runs_new_plant")
        plant, values_label, default = _PLANTS[label]
        values_text = st.text_input(values_label, value=default, key=f"runs_new_values_{plant}", help="Comma-separated; one simulation per value (the frozen condition grid).")
        c1, c2 = st.columns(2)
        duration = c1.number_input("Duration per condition [s]", 0.01, 5.0, 0.2, 0.05, key="runs_new_duration")
        seed = int(c2.number_input("Seed", 0, 2**31 - 1, 0, 1, key="runs_new_seed"))
        try:
            values = _parse_values(values_text)
            spec = ja.build_sweep_spec(plant, duration, values, seed) if values else None
        except Exception as exc:  # parsing or backend validation message, shown inline
            st.error(f"Invalid sweep: {exc}", icon=":material/error:")
            return
        if spec is None:
            st.info("Enter at least one value.", icon=":material/info:")
            return
        st.markdown("**Experiment summary**")
        st.caption(f"Plant profile: {spec.plant_profile_id} · {len(spec.conditions)} condition(s) · seed {spec.seed} · {duration:g} s each · config hash {spec.config_hash()[:16]}…")
        if st.button("Queue experiment", key="runs_submit", type="primary", icon=":material/playlist_add:"):
            try:
                job = ja.submit(spec)
                st.session_state["runs_selected_job"] = job.job_id
                st.success(f"Job queued ({job.job_id}). You can leave this page while the worker runs.", icon=":material/schedule:")
            except Exception as exc:
                st.error(f"The experiment was not queued: {exc}", icon=":material/error:")


def _jobs_table(jobs) -> None:
    rows = [{"Job ID": j.job_id, "Type": j.kind, "State": j.state.value.replace("_", " "),
             "Progress": f"{j.progress_done}/{j.progress_total}" if j.progress_total else "—", "Queued (UTC)": j.queued_at,
             "Run": j.run_id or "—"} for j in jobs]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _run_details(run) -> None:
    badge = source_badge_html(source_badge_view(run.origin, run_id=run.run_id, run_failed=run.status == "failed"))
    st.html(f'<div class="spade-root spade-badges">{badge}{status_badge_html(run.status)}</div>')
    tabs = st.tabs(["Config", "Metrics", "Artifacts", "Logs", "Provenance"])
    with tabs[0]:
        st.json({"kind": run.kind, "experiment_id": run.experiment_id, "seed": run.seed, "params": run.params,
                 "conditions": [c.get("condition") for c in run.conditions]}, expanded=2)
    with tabs[1]:
        if run.metrics:
            st.json(run.metrics, expanded=False)
        ok = [{"#": c["index"], **(c.get("metrics") or {})} for c in run.conditions if c["status"] == "ok"]
        if ok:
            st.dataframe(pd.DataFrame(ok), hide_index=True, width="stretch")
        failure_panel(list(run.conditions))
    with tabs[2]:
        if not run.artifacts:
            st.caption("No artifacts were written by this run.")
        for i, a in enumerate(run.artifacts):
            data = ja.artifact_bytes(run, a)
            c1, c2 = st.columns([3, 2])
            if data is None:
                c1.caption(f"{a['path']} — missing on disk")
            else:
                c1.download_button(a["path"].rsplit("/", 1)[-1], data, file_name=f"{run.run_id}_{a['path'].rsplit('/', 1)[-1]}", key=f"runs_art_{run.run_id}_{i}")
            c2.caption(f"{a['bytes'] / 1024:.1f} KB · sha256 {a['sha256'][:16]}…")
    with tabs[3]:
        log = next((a for a in run.artifacts if a["path"].endswith("log.txt")), None)
        data = ja.artifact_bytes(run, log) if log else None
        st.code(data.decode(errors="replace") if data else "No log was saved with this run.", language="text")
    with tabs[4]:
        provenance_footer(run)


def _selected_job(job_id: str) -> None:
    details = ja.job_details(job_id)
    job = details["job"]

    def cancel():
        ja.cancel_job(job_id)

    job_card(job, details["eta"], on_cancel=cancel, key=f"runs_card_{job_id}")
    with st.expander("Logs", expanded=not job.terminal, icon=":material/receipt_long:"):
        lines = [f"{e['utc']}  [{e['level']}]  {e['message']}" for e in details["events"]]
        st.code("\n".join(lines) or "No events yet.", language="text")
    if details["run"] is not None:
        st.markdown("##### Run details")
        _run_details(details["run"])
    elif job.terminal:
        st.caption("This job ended without a run record (for example, cancelled while queued).")


def _comparison() -> None:
    runs = [r for r in ja.list_runs() if r.status != "legacy_session"]
    if len(runs) < 2:
        run_comparison(None)
        return
    labels = {r.run_id: f"{r.run_id} · {r.kind} · {r.status}" for r in runs}
    picked = st.multiselect("Runs to compare (first = baseline)", list(labels), format_func=labels.get, key="runs_compare_ids")
    if len(picked) < 2:
        run_comparison(None)
        return
    first = ja.comparison(picked, None)
    if not first["available_metrics"]:
        run_comparison(first)
        return
    c1, c2 = st.columns([2, 1])
    metric = c1.selectbox("Metric", first["available_metrics"], key="runs_compare_metric")
    lower = c2.radio("Better is", ["lower", "higher"], horizontal=True, key="runs_compare_dir") == "lower"
    run_comparison(ja.comparison(picked, metric, lower))


def render() -> None:
    summary = ja.queue_summary()
    page_header("Workspace / Experiment Runs", "Experiment Runs",
                "Durable queue and history of experiments. Jobs keep running in a separate worker while you navigate SPADE; every run keeps its configuration hash, seed, versions, logs and a reproducible command.",
                [badge_html(f"{summary['running']} RUNNING · {summary['queued']} QUEUED", "primary" if summary["running"] else "unknown", "▶" if summary["running"] else "…"),
                 badge_html(f"{summary['workers']} WORKER(S)", "healthy" if summary["workers"] else "warning", "●" if summary["workers"] else "▲")])
    if summary.get("error"):
        st.error(f"The job store could not be read: {summary['error']}", icon=":material/error:")
        return
    _worker_panel()
    _new_experiment()

    st.markdown("##### Jobs")
    c1, c2 = st.columns([3, 2])
    flt = c1.radio("Filter", list(ja.FILTERS), horizontal=True, key="runs_filter", label_visibility="collapsed")
    search = c2.text_input("Search", key="runs_search", placeholder="job id, type, run id…", label_visibility="collapsed")
    jobs = ja.list_jobs(flt, search)
    if not jobs:
        job_card(None)
    else:
        _jobs_table(jobs)
        ids = [j.job_id for j in jobs]
        default = st.session_state.get("runs_selected_job")
        sel = st.selectbox("Selected job", ids, index=ids.index(default) if default in ids else 0, key="runs_job_select")
        st.session_state["runs_selected_job"] = sel
        active = not ja.job_details(sel)["job"].terminal

        @st.fragment(run_every=2.0 if active else None)
        def _live():
            _selected_job(sel)

        _live()

    st.markdown("##### Compare runs")
    _comparison()
    with st.expander("Interactive page records (this workspace)", icon=":material/history:"):
        legacy = [r for r in ja.list_runs() if r.status == "legacy_session"]
        if not legacy:
            st.caption("No interactive page records yet. Runs made on the DC Motor, PMSM, VSC / DPC, Monitoring, AI Diagnostics and Transfer Learning pages are recorded here (parameters and summary only).")
        else:
            st.dataframe(pd.DataFrame([{"Record": r.run_id, "Page": r.kind.removeprefix("session:"), "UTC": r.completed_utc,
                                        "Params": json.dumps(r.params, default=str)[:200]} for r in legacy]), hide_index=True, width="stretch")
