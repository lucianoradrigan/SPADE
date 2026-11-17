"""AI-N40 (2026-11-15): the dc_motor classifiers were trained and evaluated on 80-220 rad/s (simulated); above that they can be
confidently wrong (a healthy motor at 350 rad/s read as 'cage' 1.00 by the shipped RPi5 model). The GUI defaults to 300
rad/s, so AI Diagnostics and the DC Motor AI card show a visible notice outside the grid (and below it, AI-N29).
No default and no model changed."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.viz.components.model_notices import ABOVE_GRID, DC_GRID_RAD_S, speed_grid_message

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")


@pytest.mark.parametrize("w,expect", [(300.0, "AI-N40"), (350.0, "AI-N40"), (220.0, None), (80.0, None), (150.0, None), (50.0, "AI-N29"), (None, None)])
def test_the_message_depends_only_on_the_speed(w, expect):
    msg = speed_grid_message(w)
    assert (msg is None) if expect is None else (expect in msg)


def _page(name):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page(f"pages/{name}.py").run()
    return at


def test_ai_diagnostics_default_sample_shows_the_notice():
    at = _page("ai_diagnostics")
    at.slider(key="ia_omega_ref_slider").set_value(300.0).run()  # 2026-11-16 (D2): the default moved from 300 to 150 rad/s; this test keeps 300
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception
    assert any("Outside the classifier's training grid" in w.value and "AI-N40" in w.value for w in at.warning), "the 300 rad/s default sample is above the grid"


def test_ai_diagnostics_inside_the_grid_has_no_notice():
    at = _page("ai_diagnostics")
    at.slider(key="ia_omega_ref_slider").set_value(150.0).run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception
    assert not any("Outside the classifier's training grid" in w.value for w in at.warning)


def test_dc_motor_ai_card_shows_the_notice_at_the_default_speed():
    at = _page("dc_motor")
    at.slider(key="dc_duration_slider").set_value(1.0).run()
    at.button(key="dc_run_button").click().run()
    assert not at.exception
    assert any("Outside the classifier's training grid" in w.value and "AI-N40" in w.value for w in at.warning)


def test_the_grid_is_the_shipped_classifiers_grid():
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import generate_diagnosis_grid_dataset as g
    assert DC_GRID_RAD_S == (min(g.SPEEDS_RAD_S), max(g.SPEEDS_RAD_S))
    assert "{hi:.0f}" in ABOVE_GRID
