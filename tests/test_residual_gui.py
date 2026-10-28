"""Forecast-residual detector in the GUI (expansion task E14): the three states of the AI Diagnostics
panel (no model, below threshold, above threshold) and its place in the Monitoring consensus,
where it never raises an alert by itself."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import driveflow.viz.adapters.ai_adapter as da_ai
from tests.streamlit_compat import chart_keys

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


class _MeanForecaster:
    def __call__(self, x, training=False):
        return np.zeros((len(x), 16, x.shape[2]), np.float32)

    def predict(self, x, batch_size=None, verbose=0):
        return self(x)


def _install(monkeypatch, tmp_path, threshold, gate_s=None):
    run_dir = tmp_path / "dc_motor_residual_pc" / "run01"
    run_dir.mkdir(parents=True)
    metrics = {"channels": ["current_r", "rpm", "voltage_v"], "input_window": 256, "horizon": 16,
               "residual_detector": {"target_channels": ["current_r", "rpm"], "channel_scale": [1.0, 1.0], "quantile": 0.99, "threshold": threshold,
                                     "fitted_on": {"windows": "validation", "labels": ["normal"], "n": 100}}}
    if gate_s is not None:
        metrics["residual_detector"]["gate"] = {"gate_s": gate_s}
    original = da_ai.promoted_model

    def fake(domain, tier, block):
        if block == "regressor":
            return _MeanForecaster(), SimpleNamespace(tier="pc", input_window=256, horizon=16), metrics, run_dir
        return original(domain, tier, block)

    monkeypatch.setattr(da_ai, "promoted_model", fake)


def _diagnostics():
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    at.button(key="ia_generate").click().run()
    assert not at.exception
    return at


def _text(at):
    return " ".join(e.value for kind in ("info", "warning", "caption", "markdown") for e in at.get(kind))


def test_no_model_state():
    at = _diagnostics()
    assert "Forecast residual" in _text(at) and "Residual detector not available" in _text(at)


def test_below_threshold_state(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, threshold=1e9)
    at = _diagnostics()
    text = _text(at)
    assert "Below threshold." in text and "AI SUGGESTION — REQUIRES VERIFICATION" in " ".join(str(h.body) for h in at.get("html"))
    assert "behaviour differs from learned healthy dynamics — not a fault class" in text  # the caption explains what a high residual means
    assert "ia_residual_dc_motor" in chart_keys(at)


def test_above_threshold_state(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, threshold=1e-9)
    at = _diagnostics()
    warn = " ".join(w.value for w in at.warning)
    assert "Above threshold:" in warn and "behaviour differs from learned healthy dynamics — not a fault class" in warn


def test_vsc_keeps_its_forecaster_panel():
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    at.selectbox(key="ia_domain").set_value("vsc_dpc").run()
    at.button(key="ia_generate").click().run()
    assert not at.exception and "Forecast residual" not in _text(at)


@pytest.mark.parametrize("threshold, expected", [(None, "NOT_CONFIGURED"), (1e-9, "AVAILABLE")])
def test_monitoring_lists_the_residual_detector_without_alerting(monkeypatch, tmp_path, threshold, expected):
    from driveflow.monitoring.agents.agent_server import ServerAgent
    from driveflow.viz.adapters import monitoring_adapter as ma

    if threshold is not None:
        _install(monkeypatch, tmp_path, threshold)
        monkeypatch.setattr(ma, "residual_assessment", da_ai.residual_assessment)
    res = ma.run_monitoring_pass("dc_motor", {"omega_ref_rad_s": 150.0, "duration_s": 0.3, "seed": 0}, ServerAgent())
    agent = res["agents"]["Forecast residual"]
    assert agent["availability"] == expected and agent["emitted"] is False
    if expected == "AVAILABLE":
        assert res["residual"]["share_above"] > 0 and "does not raise alerts by itself" in agent["detail"]


def test_steady_state_gate_skips_the_first_gate_s_of_the_data(monkeypatch, tmp_path):
    """A detector with a steady-state gate scores only windows starting gate_s after the data start."""
    import pandas as pd

    t = np.arange(1, 6001) * 1e-4
    df = pd.DataFrame({"timestamp_s": t, "current_r": 5.0 + 0 * t, "rpm": 1400.0 + 0 * t, "voltage_v": 30.0 + 0 * t})
    _install(monkeypatch, tmp_path, threshold=1.0)
    ungated = da_ai.residual_assessment("dc_motor", df)
    _install(monkeypatch, tmp_path / "g", threshold=1.0, gate_s=0.3)
    gated = da_ai.residual_assessment("dc_motor", df)
    assert ungated["gate_s"] is None and gated["gate_s"] == 0.3
    assert 0 < gated["n_windows"] < ungated["n_windows"]
    _install(monkeypatch, tmp_path / "h", threshold=1.0, gate_s=10.0)
    short = da_ai.residual_assessment("dc_motor", df)
    assert not short["available"] and "steady-state gate" in short["reason"]
