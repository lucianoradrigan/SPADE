"""ProvenanceFooter (spec Sec. 9, 16.2): origin, run id, config hash, versions and the reproducible
command of a persistent Run (driveflow.runs.schemas.Run)."""

from __future__ import annotations

import streamlit as st

from driveflow.viz.components.page_header import provenance_table
from driveflow.viz.components.state_panel import ComponentView, render_state


def provenance_view(run, loading: bool = False, error: str | None = None) -> ComponentView:
    if loading:
        return ComponentView("loading", "Preparing manifest…")
    if error:
        return ComponentView("error", "Reproducibility manifest unavailable", error)
    if run is None:
        return ComponentView("empty", "No run generated")
    if run.status in ("failed", "cancelled"):
        tb = next((a["path"] for a in run.artifacts if a["path"].endswith("traceback.txt")), None)
        return ComponentView("negative", f"Run {run.status}" + (f" — traceback: {tb}" if tb else ""), run.failure_reason)
    if not run.command:
        return ComponentView("negative", "Session record — no reproducible command", "Recorded by an interactive page; re-create it from the page with the same parameters.")
    return ComponentView("success", "Copy reproducible command")


def provenance_rows(run) -> dict:
    env = run.environment or {}
    return {
        "Origin": run.origin,
        "Run": run.run_id,
        "Status": run.status,
        "Config hash (SHA-256)": run.config_hash,
        "Seed": run.seed,
        "Commit": (env.get("commit") or "unknown") + (" (uncommitted changes)" if env.get("dirty_worktree") else ""),
        "Python / driveflow": f"{env.get('python', '—')} / {(env.get('packages') or {}).get('driveflow', '—')}",
        "Completed (UTC)": run.completed_utc,
    }


def provenance_footer(run, loading: bool = False, error: str | None = None) -> None:
    view = provenance_view(run, loading, error)
    if view.state in ("empty", "loading", "error"):
        render_state(view)
        return
    provenance_table(provenance_rows(run), title="Provenance")
    if run.command:
        st.caption("Copy reproducible command — run it from a SPADE checkout; paths are relative to the workspace ($SPADE_WORKSPACE or ./workspace).")
        st.code(run.command, language="bash")
    if view.state == "negative":
        render_state(view)
