"""SourceBadge (spec Sec. 9): where the numbers on screen came from. It states provenance only --
never activity or telemetry."""

from __future__ import annotations

import streamlit as st

from driveflow.runs.schemas import ORIGINS
from driveflow.viz.components.page_header import badge_html
from driveflow.viz.components.state_panel import ComponentView, render_state, state_html
from driveflow.viz.theme.tokens import SOURCE_META


def source_badge_view(origin: str | None, run_id: str | None = None, dataset_hash: str | None = None, license: str | None = None,
                      tool_ver: str | None = None, run_failed: bool = False, loading: bool = False) -> ComponentView:
    if loading:
        return ComponentView("loading", "Checking source…")
    if not origin:
        return ComponentView("empty", "No data source selected")
    if origin not in ORIGINS:
        return ComponentView("error", "Source cannot be verified", f"Unknown origin label {origin!r}.")
    parts = [p for p in (f"run {run_id}" if run_id else None, f"dataset {dataset_hash[:12]}…" if dataset_hash else None,
                         f"license: {license}" if license else None, tool_ver) if p]
    if run_failed:
        return ComponentView("negative", "Origin known; run failed", " · ".join([origin, *parts]))
    return ComponentView("success", origin, " · ".join(parts) or None)


def source_badge_html(view: ComponentView) -> str:
    """Inline badge when the source is known; the full state row otherwise."""
    if view.state == "success":
        label, tone = SOURCE_META.get(view.message, (view.message, "unknown"))
        return badge_html(label, tone, "◆", view.detail)
    return state_html(view)


def source_badge(origin: str | None, **kwargs) -> None:
    view = source_badge_view(origin, **kwargs)
    if view.state == "success":
        st.html(f'<div class="spade-root spade-badges">{source_badge_html(view)}</div>')
    else:
        render_state(view)
