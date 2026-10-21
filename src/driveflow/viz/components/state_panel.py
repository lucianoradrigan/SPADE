"""The five display states shared by the E0 components (spec Sec. 9): empty, loading, error,
success and negative result. Each is an icon glyph + a text label + a message -- never colour
alone -- and the message strings are the spec's microcopy (Sec. 10)."""

from __future__ import annotations

import dataclasses
from html import escape

import streamlit as st

from driveflow.viz.theme.tokens import DARK, STATUS_META

STATES = ("empty", "loading", "error", "success", "negative")


@dataclasses.dataclass(frozen=True)
class ComponentView:
    """What a component shows, computed without Streamlit so it can be tested directly."""

    state: str
    message: str
    detail: str | None = None

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {self.state!r}")


def state_html(view: ComponentView, label: str | None = None) -> str:
    text, glyph, tone = STATUS_META[view.state]
    detail = f'<div class="spade-check-detail">{escape(view.detail)}</div>' if view.detail else ""
    return (f'<div class="spade-root spade-checks" role="status" aria-live="polite"><div class="spade-check" style="--tone:{DARK[tone]}">'
            f'<span class="spade-check-glyph" aria-hidden="true">{escape(glyph)}</span>'
            f'<div><div class="spade-check-name">{escape(view.message)}</div>{detail}</div>'
            f'<span class="spade-badge" style="--tone:{DARK[tone]}">{escape(label or text)}</span></div></div>')


def render_state(view: ComponentView, label: str | None = None) -> None:
    st.html(state_html(view, label))
