"""JobCard (spec Sec. 6 wireframe, Sec. 9): the persisted state of one job. Progress is shown only
when measurable and the ETA only when the service has enough history -- otherwise the card says so
instead of estimating."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from driveflow.viz.components.state_panel import ComponentView, render_state

_ACTIVE_TEXT = {"queued": "Job queued", "preparing": "Preparing", "running": "Running", "cancelling": "Cancellation requested. The current solver call may need to finish."}


def job_view(job, loading: bool = False, error: str | None = None) -> ComponentView:
    if loading:
        return ComponentView("loading", "Loading job…")
    if error:
        return ComponentView("error", "Job failed — view log", error)
    if job is None:
        return ComponentView("empty", "No jobs yet", "Nothing has been generated yet. Configure an experiment and select Generate.")
    state = job.state.value
    if state in _ACTIVE_TEXT:
        return ComponentView("loading", _ACTIVE_TEXT[state], job.phase)
    if state == "failed":
        return ComponentView("error", "Job failed — view log", job.error)
    if state == "completed":
        return ComponentView("success", "Job completed")
    if state == "cancelled":
        return ComponentView("negative", "Job cancelled", job.error)
    return ComponentView("negative", "Completed with unsuccessful conditions", job.error)


def eta_text(eta_seconds: float | None) -> str:
    if eta_seconds is None:
        return "ETA unavailable until enough evaluations are completed."
    return f"ETA ≈ {eta_seconds:.0f} s (median of previous jobs of this kind)"


def job_card(job, eta_seconds: float | None = None, on_cancel: Callable[[], None] | None = None, key: str = "job_card") -> None:
    view = job_view(job)
    render_state(view, label=None if job is None else job.state.value.replace("_", " ").upper())
    if job is None:
        return
    frac = job.progress_fraction
    if job.active or job.state.value == "queued":
        if frac is None:
            st.caption("Progress not measurable for this job yet.")
        else:
            st.progress(frac, text=f"{job.progress_done} / {job.progress_total} · {job.phase or ''}")
        st.caption(eta_text(eta_seconds))
        if on_cancel is not None and not job.cancel_requested:
            st.button("Cancel request", key=f"{key}_cancel", on_click=on_cancel, icon=":material/cancel:")
    elif job.progress_total:
        st.caption(f"Completed simulations: {job.progress_done} / {job.progress_total}")
    if not job.resumable and job.terminal and job.state.value != "completed":
        st.caption("Resume unavailable — no recoverable checkpoint was saved.")
