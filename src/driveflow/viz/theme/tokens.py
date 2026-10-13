"""SPADE design tokens -- the single source of truth for every colour the app uses (guide Sec. 3.3).

.streamlit/config.toml's [theme.dark]/[theme.light] sections carry the same values for Streamlit's
native widgets (they cannot import Python); tests/test_theme_tokens.py checks the two stay in sync.
Custom components (theme/css.py) and Plotly traces read from here.

Status colours are always paired with a text label and an icon glyph (STATUS_META) -- state is
never conveyed by colour alone (accessibility, guide Sec. 3.1 item 7).
"""

DARK = {
    "bg_app": "#0B1525",
    "bg_nav": "#081321",
    "surface": "#12263B",
    "surface_raised": "#19344E",
    "border": "#2B465D",
    "text": "#E2E8F0",
    "text_muted": "#93A9C1",
    "primary": "#38BDF8",
    "healthy": "#22C55E",
    "warning": "#F59E0B",
    "critical": "#EF4444",
    "training": "#A78BFA",
    "unknown": "#94A3B8",
}

#: Same semantic roles for the native light theme; accents darkened for contrast on white.
LIGHT = {
    "bg_app": "#F5F8FB",
    "bg_nav": "#EAF0F6",
    "surface": "#FFFFFF",
    "surface_raised": "#E6EEF6",
    "border": "#C9D6E3",
    "text": "#0F1E2E",
    "text_muted": "#4A6076",
    "primary": "#0369A1",
    "healthy": "#15803D",
    "warning": "#B45309",
    "critical": "#B91C1C",
    "training": "#6D28D9",
    "unknown": "#64748B",
}

#: Functional series colours for charts: one meaning per colour, used consistently across pages
#: (electrical = primary, mechanical = healthy-green family is avoided to not read as "healthy").
SERIES = {
    "electrical": "#38BDF8",
    "mechanical": "#F59E0B",
    "vibration": "#A78BFA",
    "reference": "#94A3B8",
    "secondary": "#F472B6",
    "training": "#38BDF8",
    "validation": "#A78BFA",
    "test": "#F59E0B",
    "base": "#94A3B8",
    "candidate": "#38BDF8",
}

#: status -> (label, glyph, token). Glyphs are plain Unicode so they render without icon fonts.
STATUS_META = {
    "healthy": ("HEALTHY", "●", "healthy"),
    "warning": ("WARNING", "▲", "warning"),
    "critical": ("CRITICAL", "■", "critical"),
    "offline": ("OFFLINE", "○", "unknown"),
    "unknown": ("UNKNOWN", "?", "unknown"),
    "pass": ("PASS", "✓", "healthy"),
    "fail": ("FAIL", "✕", "critical"),
    "pending": ("PENDING", "…", "warning"),
    "incomplete": ("INCOMPLETE", "…", "warning"),
    "not_applicable": ("N/A", "–", "unknown"),
    "info": ("INFO", "i", "primary"),
    "approved": ("APPROVED", "✓", "healthy"),
    "blocked": ("BLOCKED", "✕", "critical"),
}

#: Data-source labels (guide Sec. 4.2): what produced the numbers on screen.
SOURCE_META = {
    "simulated": ("SIMULATED", "primary"),
    "demo": ("DEMO", "warning"),
    "connected": ("CONNECTED", "healthy"),
    "stale": ("STALE", "warning"),
    "offline": ("OFFLINE", "unknown"),
    "unavailable": ("NO SOURCE", "unknown"),
    "synthetic": ("SYNTHETIC DATA", "primary"),
    "external": ("EXTERNAL FILE", "warning"),
}


def palette(theme_type: str | None = None) -> dict:
    """Token set for the active theme. `theme_type` is st.context.theme.type when available; it
    is None outside a browser session (tests), where the default dark set is returned."""
    return LIGHT if theme_type == "light" else DARK
