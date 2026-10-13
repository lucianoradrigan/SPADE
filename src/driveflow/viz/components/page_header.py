"""Page header, status/source badges, provenance table and empty state.

Every dynamic string is HTML-escaped before it reaches st.html/markdown -- uploaded file names,
acquisition IDs and model names are user-controlled (AUDIT AI-12)."""

from html import escape

import streamlit as st

from driveflow.viz.theme.tokens import DARK, SOURCE_META, STATUS_META


def badge_html(label: str, tone: str = "unknown", glyph: str | None = None, title: str | None = None) -> str:
    color = DARK.get(tone, tone)
    g = f'<span class="spade-glyph" aria-hidden="true">{escape(glyph)}</span>' if glyph else ""
    t = f' title="{escape(title)}"' if title else ""
    return f'<span class="spade-badge" style="--tone:{color}"{t}>{g}{escape(label)}</span>'


def status_badge_html(status: str, title: str | None = None) -> str:
    label, glyph, tone = STATUS_META.get(status.lower(), (status.upper(), "?", "unknown"))
    return badge_html(label, tone, glyph, title)


def source_badge_html(source: str, title: str | None = None) -> str:
    label, tone = SOURCE_META.get(source, (source.upper(), "unknown"))
    return badge_html(label, tone, "◆", title)


def page_header(eyebrow: str, title: str, description: str, badges: list | None = None) -> None:
    """badges: list of already-built badge HTML strings (status_badge_html/source_badge_html)."""
    right = f'<div class="spade-badges">{"".join(badges or [])}</div>' if badges else ""
    st.html(
        f'<div class="spade-root spade-header"><div><p class="spade-eyebrow">{escape(eyebrow)}</p>'
        f'<h1 class="spade-title">{escape(title)}</h1><p class="spade-desc">{escape(description)}</p></div>{right}</div>'
    )


def badges_row(badges: list) -> None:
    st.html(f'<div class="spade-root spade-badges">{"".join(badges)}</div>')


def provenance_table(rows: dict, title: str | None = None) -> None:
    """rows: {label: value}; values are shown verbatim (escaped), None as an em dash."""
    body = "".join(f"<tr><td>{escape(str(k))}</td><td><code>{escape('—' if v is None else str(v))}</code></td></tr>" for k, v in rows.items())
    head = f'<p class="spade-eyebrow">{escape(title)}</p>' if title else ""
    st.html(f'<div class="spade-root">{head}<table class="spade-prov">{body}</table></div>')


def empty_state(text: str) -> None:
    """A neutral placeholder for "nothing to show yet" -- plain text, no decorative animation."""
    st.info(text, icon=":material/info:")
