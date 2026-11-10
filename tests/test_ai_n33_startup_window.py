"""AI-N33 (found 2026-11-09 by the fresh-clone check of v0.2.0): with AI Diagnostics' default sample (healthy, PI,
300 rad/s, 0.6 s) the classified window (0.40-0.60 s) lies in the start-up transient and the promoted PC classifier says
'cage' at 100 % calibrated confidence, shown as a known condition. The page now warns whenever the evaluated window
starts inside the start-up transient (t < 0.5 s); defaults are unchanged."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from driveflow.viz.adapters import ai_adapter as da_ai

APP = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def test_the_window_bounds():
    df = pd.DataFrame({"timestamp_s": [(k + 1) * 1e-4 for k in range(10000)]})
    assert da_ai.startup_window(df, 2048) is None                     # 1.0 s run: window 0.795-1.0 s
    short = df.iloc[:6000]
    start, end = da_ai.startup_window(short, 2048)
    assert start < 0.5 and abs(end - 0.6) < 1e-9


def _ai_page():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    return at


def test_default_sample_warns_about_the_startup_window():
    at = _ai_page()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception
    assert any("Start-up window (AI-N33)" in w.value for w in at.warning)


def test_a_steady_state_sample_does_not_warn():
    at = _ai_page()
    at.slider(key="ia_duration_slider").set_value(1.0).run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception
    assert not any("Start-up window (AI-N33)" in w.value for w in at.warning)
