"""DC-DC Converters → DAB tab (task E9 preparation): parameter diagram only, empty result states."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from driveflow.viz.adapters import dab_adapter as dab
from tests.streamlit_compat import chart_keys

APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "driveflow" / "viz" / "app.py")


@pytest.mark.parametrize("strategy, expect", [("SPS", (0.3, 1.0, 1.0)), ("DPS", (0.3, 0.6, 0.6)), ("EPS", (0.3, 0.6, 1.0)), ("TPS", (0.3, 0.6, 0.7))])
def test_strategy_constraints(strategy, expect):
    assert dab.constrain(strategy, 0.3, 0.6, 0.7) == expect


@pytest.mark.parametrize("d", [1.0, 0.6, 0.25])
def test_bridge_waveform_duty_matches_the_inner_shift(d):
    w = dab.bridge_waveforms(0.4, d, 1.0, n=4000)
    assert np.mean(w["v_p"] != 0) == pytest.approx(d, abs=2e-3)
    assert np.mean(w["v_p"]) == pytest.approx(0.0, abs=2e-3)  # symmetric, no DC component


def test_dab_tab_shows_the_diagram_and_empty_states():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.switch_page("pages/dcdc_converters.py").run()
    assert not at.exception and [t.label for t in at.tabs][:2] == ["Buck", "DAB"]
    at.selectbox(key="dab_strategy").set_value("SPS").run()
    assert at.slider(key="dab_d1").disabled and at.slider(key="dab_d2").disabled
    assert "dab_phase_diagram" in chart_keys(at)
    text = " ".join(e.value for kind in ("info", "caption", "markdown") for e in at.get(kind))
    assert "No simulation has been executed for this configuration." in text and "not a simulation" in text
    assert at.button(key="dab_compare").disabled
