"""X2 (round PLECS/OM): WCAG 2.1 AA contrast of SPADE's colour tokens, both themes.

- Text (normal size) >= 4.5:1: body text and muted text on every background; badge labels (theme text
  on the 14 % tone tint of a badge); the eyebrow, which is the primary tone used as text.
- Essential UI elements >= 3:1: every status and provenance tone (borders, glyphs, KPI rails) on every
  background of its own theme -- and the components must pick the ACTIVE theme's tones
  (tokens.active_palette), not the dark set. Evidence: docs/audit/expansion/x2_contrast.md.
"""

import pytest

from driveflow.runs.schemas import ORIGINS
from driveflow.viz.theme import tokens
from driveflow.viz.theme.tokens import DARK, LIGHT, SOURCE_META, STATUS_META

BACKGROUNDS = ("bg_app", "bg_nav", "surface", "surface_raised")
THEMES = {"dark": DARK, "light": LIGHT}


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def mix(fg, bg, alpha):
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    g = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(alpha * x + (1 - alpha) * y):02X}" for x, y in zip(f, g))


TONES = sorted({t for _l, _g, t in STATUS_META.values()} | {t for _l, t in SOURCE_META.values()})


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("text", ["text", "text_muted"])
def test_text_on_every_background(theme, text):
    p = THEMES[theme]
    assert min(ratio(p[text], p[b]) for b in BACKGROUNDS) >= 4.5


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("tone", TONES)
def test_status_and_provenance_tones(theme, tone):
    p = THEMES[theme]
    assert min(ratio(p[tone], p[b]) for b in BACKGROUNDS) >= 3.0, f"{tone} as a UI element in {theme}"
    for b in BACKGROUNDS:
        assert ratio(p["text"], mix(p[tone], p[b], 0.14)) >= 4.5, f"badge label on the {tone} tint over {b} in {theme}"


@pytest.mark.parametrize("theme", THEMES)
def test_primary_used_as_text_eyebrow(theme):
    p = THEMES[theme]
    assert min(ratio(p["primary"], p[b]) for b in BACKGROUNDS) >= 4.5


def test_every_provenance_label_has_a_tone_and_is_a_valid_origin():
    for origin in ORIGINS:
        assert origin in SOURCE_META, origin
    assert "SIMULATED · OPENMODELICA" in ORIGINS and not any("LIVE" in o for o in ORIGINS)


def test_components_use_the_active_theme(monkeypatch):
    monkeypatch.setattr(tokens, "active_palette", lambda: LIGHT)
    from driveflow.viz.components.page_header import badge_html

    html = badge_html("SIMULATED", "primary")
    assert LIGHT["primary"] in html and DARK["primary"] not in html
