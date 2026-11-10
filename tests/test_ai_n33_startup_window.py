"""AI-N33 (found 2026-11-09 by the fresh-clone check of v0.2.0; corrected under decision N-D1).

Before: AI Diagnostics classified the LAST rows of a 0.6 s sample (t = 0.40-0.60 s, inside the start-up transient)
and the promoted PC classifier said 'cage' at 100 % on a healthy motor at 300 rad/s. Now the page evaluates the last
window aligned like the training windows (from t = 0; acquisitions.window_starts) and the default duration is 0.82 s,
the shortest whose last aligned window lies after t = 0.5 s. The start-up warning stays for shorter durations."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from driveflow.viz.adapters import ai_adapter as da_ai
from driveflow.viz.views import ai_diagnostics as page

APP = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _df(seconds: float) -> pd.DataFrame:
    return pd.DataFrame({"timestamp_s": [(k + 1) * 1e-4 for k in range(round(seconds * 1e4))]})


def test_the_window_follows_the_training_grid():
    assert da_ai.training_window_start(_df(0.6), 2048) == 2048          # 0.2049-0.4096 s
    assert da_ai.training_window_start(_df(0.82), 2048) == 3 * 2048     # 0.6145-0.8192 s
    assert da_ai.training_window_start(_df(1.0), 2048) == 3 * 2048
    assert da_ai.training_window_start(_df(0.1), 2048) is None


def test_the_default_duration_is_the_shortest_steady_one():
    d = page.DC_DEFAULT_DURATION_S
    start = da_ai.training_window_start(_df(d), 2048)
    assert start * 1e-4 >= da_ai.STARTUP_S
    shorter = da_ai.training_window_start(_df(d - 0.01), 2048)
    assert shorter * 1e-4 < da_ai.STARTUP_S


def test_startup_window_detection():
    df = _df(1.0)
    assert da_ai.startup_window(df, 2048) is None
    start, end = da_ai.startup_window(_df(0.6), 2048)
    assert start < 0.5 and abs(end - 0.6) < 1e-9


def _ai_page():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    return at


def _generate(at):
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception


def test_default_healthy_sample_is_normal_without_warning():
    at = _ai_page()
    _generate(at)
    assert not any("Start-up window (AI-N33)" in w.value for w in at.warning)
    assert any(m.value == "normal" for m in at.metric)


def test_a_short_sample_still_warns():
    at = _ai_page()
    at.slider(key="ia_duration_slider").set_value(0.6).run()
    _generate(at)
    assert any("Start-up window (AI-N33)" in w.value for w in at.warning)
