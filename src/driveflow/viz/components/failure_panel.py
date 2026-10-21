"""FailurePanel (spec Sec. 9, 19): per-condition failures of a run, shown with the same detail as
successes -- never averaged away."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driveflow.viz.components.state_panel import ComponentView, render_state


def failure_view(conditions, loading: bool = False, error: str | None = None) -> ComponentView:
    if loading:
        return ComponentView("loading", "Checking failed runs…")
    if error:
        return ComponentView("error", "Failure records unavailable", error)
    if not conditions:
        return ComponentView("empty", "No failures recorded", "This run has no per-condition records.")
    failed = [c for c in conditions if c["status"] == "failed"]
    skipped = [c for c in conditions if c["status"] == "skipped"]
    if not failed and not skipped:
        return ComponentView("success", "All conditions completed", f"{len(conditions)} of {len(conditions)} conditions ok.")
    return ComponentView("negative", f"{len(failed)} failed · {len(skipped)} skipped of {len(conditions)} conditions",
                         "Divergence, solver errors and invalid conditions are listed below; nothing was imputed.")


def failure_rows(conditions) -> list[dict]:
    return [{"#": c.get("index"), "Status": c["status"].upper(), "Condition": str(c.get("condition")), "Error": c.get("error") or ""}
            for c in conditions if c["status"] != "ok"]


def failure_panel(conditions, loading: bool = False, error: str | None = None) -> None:
    view = failure_view(conditions, loading, error)
    render_state(view)
    if view.state == "negative":
        st.dataframe(pd.DataFrame(failure_rows(conditions)), hide_index=True, width="stretch")
