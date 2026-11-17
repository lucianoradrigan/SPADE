"""R1 (2026-11-15, hotfix v0.2.1): the 18 pages never claim what SPADE has not done, and keep the provenance labels.

For every page of the app (driveflow.viz.app.PAGES), rendered as a user opens it and, on the pages that show model
predictions, after their primary action:
  - no "hardware verified", "production ready" or the word "live" (case-insensitive, word boundaries), except the
    occurrences in ALLOWED, each justified below;
  - a page that shows a model prediction carries the full label "AI SUGGESTION — REQUIRES VERIFICATION";
  - a page that shows results carries "SIMULATED".
"""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.viz.app import PAGES as APP_PAGES

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")
PAGES = [Path(f).stem for items in APP_PAGES.values() for f, _, _ in items]
FULL_AI_LABEL = "AI SUGGESTION — REQUIRES VERIFICATION"
BANNED = {"hardware verified": re.compile(r"\bhardware[\s-]verified\b", re.I),
          "production ready": re.compile(r"\bproduction[\s-]ready\b", re.I),
          "live": re.compile(r"\blive\b", re.I)}
#: (page, banned word, regex of the allowed occurrence, why it is not a claim)
ALLOWED = [
    ("edge_deployment", "live", r"(const int live =|= live \?|live \? s)", "a variable name in the generated C inference code (non-constant channel), not a claim"),
    ("documentation", "live", r"CONNECTED / LIVE", "the documentation explains the status label reserved for real telemetry, which SPADE never shows"),
]
#: Pages that show a model prediction, and how to make them show it (key of the button to click, after optional setup).
PREDICTION_PAGES = {"dc_motor": "dc_run_button", "ai_diagnostics": "ia_generate", "monitoring": "lm_run_button"}
#: Pages without results of their own (nothing to label SIMULATED), with the reason.
NO_RESULTS = {"algorithm_guide": "a static guide to algorithm families; it shows no run, metric or prediction"}


def _page(name: str, action: str | None = None) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page(f"pages/{name}.py").run()
    if action:
        if name == "dc_motor":
            at.slider(key="dc_duration_slider").set_value(1.0).run()
        at.button(key=action).click().run()
    return at


def _text(at) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "success", "code", "text", "title", "header", "subheader") for e in at.get(kind) if isinstance(getattr(e, "value", None), str)]
    parts += [str(getattr(e, "label", "")) for kind in ("button", "selectbox", "slider", "radio", "checkbox", "number_input", "tab", "expander", "metric") for e in at.get(kind)]
    parts += [str(getattr(e, "value", "")) for e in at.get("metric")]
    # visible text of st.html only: markup (e.g. aria-live="polite") is not something the page says
    parts += [re.sub(r"<[^>]+>", " ", str(getattr(h, "body", ""))) for h in at.get("html")]
    return "\n".join(parts)


def _banned(page: str, text: str) -> list:
    hits = []
    for word, rx in BANNED.items():
        for m in rx.finditer(text):
            ctx = text[max(0, m.start() - 40): m.end() + 40]
            if not any(p == page and w == word and re.search(a, ctx) for p, w, a, _ in ALLOWED):
                hits.append((word, ctx.replace("\n", " ")))
    return hits


def test_the_check_covers_the_18_pages():
    assert len(PAGES) == 18 and set(PREDICTION_PAGES) <= set(PAGES) and set(NO_RESULTS) <= set(PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_page_tells_the_truth(page):
    at = _page(page, PREDICTION_PAGES.get(page))
    assert not at.exception, at.exception
    text = _text(at)
    assert not _banned(page, text), _banned(page, text)
    if page in PREDICTION_PAGES:
        assert FULL_AI_LABEL in text, f"{page} shows a prediction without '{FULL_AI_LABEL}'"
    if page not in NO_RESULTS:
        assert "SIMULATED" in text, f"{page} shows results without the SIMULATED label"


def test_the_banned_word_check_catches_what_it_should():
    assert _banned("overview", "Status: LIVE") and _banned("overview", "hardware-verified model") and _banned("x", "Production Ready")
    assert not _banned("overview", "delivered, olive, lively")  # word boundaries
    assert not _banned("documentation", "label CONNECTED / LIVE is reserved")


@pytest.mark.parametrize("page", ["edge_deployment", "model_registry"])
def test_the_esp32_low_speed_limitation_is_visible(page):
    """R-D5: no ESP32 round is open; its low-speed limitation (AI-N28, AI-N36) is shown where its model appears."""
    from driveflow.viz.components.model_notices import ESP32_LOW_SPEED

    at = _page(page)
    assert not at.exception
    assert any(w.value == ESP32_LOW_SPEED for w in at.warning), f"{page}: ESP32 low-speed notice missing"
    assert "AI-N28" in ESP32_LOW_SPEED and "AI-N36" in ESP32_LOW_SPEED and "AI_TECHNICAL_AUDIT.md" in ESP32_LOW_SPEED


def test_the_board_measured_tile_says_there_are_no_board_measurements():
    at = _page("model_registry")
    text = _text(at)
    assert "Board-measured" in text and "no board measurements recorded yet" in text
