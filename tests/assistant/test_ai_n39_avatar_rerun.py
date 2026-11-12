"""AI-N39 (2026-11-12): on the Assistant page in a real browser, the avatar reported its state back to Python on every
repaint (a new component each time). Setting a component's state requests a rerun, so every state change during a
turn restarted the script and the reply never arrived (one model call, then a rerun storm). Since the fix the component
reports only when asked (report_env=True), and only the avatar demo page asks. The browser check is in
docs/audit/assistant/ai_n39_avatar_rerun.md (Playwright is not a project dependency)."""

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "src" / "driveflow" / "viz" / "components" / "radrigan_avatar" / "avatar.js"


def test_the_avatar_does_not_report_unless_asked():
    from driveflow.viz.components.radrigan_avatar import avatar_props

    assert avatar_props("thinking")["report_env"] is False
    assert avatar_props("thinking", report_env=True)["report_env"] is True


def test_the_javascript_reports_only_when_asked():
    js = JS.read_text()
    calls = [line for line in js.splitlines() if "setStateValue(" in line and "const" not in line]
    assert calls, "the component no longer reports at all (the demo page relies on it)"
    assert "if (d.report_env && setStateValue" in js


def test_the_assistant_page_never_asks_for_the_report():
    import driveflow.viz.views.assistant as page

    assert "report_env" not in inspect.getsource(page)


def test_the_demo_page_still_asks_for_it():
    assert "report_env=True" in (ROOT / "scripts" / "assistant_avatar_demo.py").read_text()
