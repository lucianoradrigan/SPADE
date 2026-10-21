"""VSC / DPC page, tab "Parameter Identification" (expansion task E10b, spec Sec. 8.3)."""

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from driveflow.ai.pinn import identification as idn
from driveflow.viz.adapters import identification_adapter as ia

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/vsc_dpc.py").run()
    assert not at.exception
    return at


def _text(at):
    return " ".join(e.value for kind in ("warning", "info", "caption", "error", "markdown") for e in at.get(kind))


def test_tab_exists_with_permanent_disclaimer_before_any_run():
    at = _open()
    assert any(t.label == "Parameter Identification" for t in at.tabs)
    assert "Parameter estimate — not a validated fault diagnosis." in _text(at)
    assert at.button(key="pid_run") is not None


def test_least_squares_on_simulated_degraded_capacitor():
    at = _open()
    at.slider(key="pid_c_scale").set_value(0.6).run()
    at.select_slider(key="pid_noise").set_value(0.0).run()
    at.radio(key="pid_estimator").set_value("least_squares").run()
    at.number_input(key="pid_nboot_least_squares").set_value(20).run()
    at.button(key="pid_run").click().run()
    assert not at.exception
    table = next(d.value for d in at.dataframe if "Simulated true value" in getattr(d.value, "columns", []))
    c_row = table[table["Parameter"] == "Filter capacitance C"].iloc[0]
    assert abs(c_row["Estimate"] - 9.0) < 1e-3 and abs(c_row["Simulated true value"] - 9.0) < 1e-9  # µF
    assert "moving-block bootstrap" in _text(at)
    assert at.button(key="pid_run").disabled  # unchanged controls


def test_plecs_source_is_not_available():
    at = _open()
    at.radio(key="pid_source").set_value("PLECS").run()
    assert not at.exception
    assert "PLECS data is not available in this version" in _text(at)


def test_upload_with_missing_columns_is_reported():
    at = _open()
    at.radio(key="pid_source").set_value("Uploaded file (CSV)").run()
    assert not at.exception
    # file_uploader cannot be driven by AppTest; the adapter path it feeds is tested directly
    df = pd.DataFrame({"vc_real": [1.0] * 300})
    try:
        ia.run_identification(df, 8.0, 0.01, 0.02, "least_squares", 5)
    except ValueError as exc:
        assert "missing VSC column" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validation_summary_reads_the_closest_noise_level(tmp_path):
    rep = {"summary": [{"estimator": "pinn", "param": p, "noise_pct": n, "mean_abs_error_pct": 1.0, "max_abs_error_pct": 2.0, "coverage": 0.5, "mean_rel_width_pct": 1.0, "n": 3}
                       for p in idn.PARAMS for n in (0.0, 5.0)]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rep))
    out = ia.validation_summary("pinn", 4.0, path)
    assert set(out["Noise (%)"]) == {5.0} and len(out) == 3
    assert ia.validation_summary("least_squares", 0.0, path) is None
    assert ia.validation_summary("pinn", 0.0, tmp_path / "missing.json") is None
