"""KPI cards and check lists (gate items, integrity checks)."""

from html import escape

import streamlit as st

from driveflow.viz.theme.tokens import DARK, STATUS_META


def kpi_row(items: list) -> None:
    """items: [{"label", "value", "unit"?, "sub"?, "tone"?}] -- value is pre-formatted text, so the
    caller decides precision and "—" for unavailable data (never a fabricated number)."""
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        tone = DARK.get(item.get("tone", "primary"), item.get("tone", DARK["primary"]))
        unit = f'<span class="spade-kpi-unit">{escape(item["unit"])}</span>' if item.get("unit") else ""
        sub = f'<div class="spade-kpi-sub">{escape(item["sub"])}</div>' if item.get("sub") else ""
        with col:
            st.html(
                f'<div class="spade-root spade-kpi" style="--tone:{tone}"><div class="spade-kpi-label">{escape(item["label"])}</div>'
                f'<div class="spade-kpi-value">{escape(str(item["value"]))}{unit}</div>{sub}</div>'
            )


def check_list(items: list) -> None:
    """items: [(name, status, detail)] with status in STATUS_META keys (pass/fail/pending/...)."""
    rows = []
    for name, status, detail in items:
        label, glyph, tone = STATUS_META.get(str(status).lower(), (str(status).upper(), "?", "unknown"))
        rows.append(
            f'<div class="spade-check" style="--tone:{DARK[tone]}"><span class="spade-check-glyph" aria-hidden="true">{escape(glyph)}</span>'
            f'<div><div class="spade-check-name">{escape(name)}</div><div class="spade-check-detail">{escape(detail or "")}</div></div>'
            f'<span class="spade-badge" style="--tone:{DARK[tone]}">{escape(label)}</span></div>'
        )
    st.html(f'<div class="spade-root spade-checks">{"".join(rows)}</div>')
