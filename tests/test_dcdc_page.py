"""DC-DC Converters page (expansion task E6; spec Sec. 8.1): Buck tab with the Generate pattern."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.viz.adapters import dcdc_adapter as da
from driveflow.viz.adapters import jobs_adapter as ja
from tests.streamlit_compat import chart_key

APP = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.switch_page("pages/dcdc_converters.py").run()
    assert not at.exception, at.exception
    return at


def _buck_charts(at):
    return [c for c in at.get("plotly_chart") if str(chart_key(c)).startswith("buck_")]


def test_empty_state_before_generate_and_tabs():
    at = _page()
    assert [t.label for t in at.tabs] == ["Buck", "DAB"]  # DAB tab added in round 2026-10-21 (D8)
    assert any("Generate Buck scenario" in i.value for i in at.info)
    assert not _buck_charts(at)  # no Buck chart before a run (the DAB tab only draws its parameter diagram)


def test_generate_runs_the_real_scenario():
    at = _page()
    at.button(key="buck_run_button").click().run()
    assert not at.exception, at.exception
    assert len(_buck_charts(at)) == 3
    html = " ".join(str(h.body) for h in at.get("html"))
    assert "Start-up settling" in html and "buck-001" in html
    assert at.button(key="buck_run_button").disabled  # unchanged controls -> no duplicate run


def test_adapter_metrics_match_controller_measurements():
    p = da.buck_defaults()
    m = da.buck_metrics(da.run_buck(p), p)
    assert m["settling_time_ms"] == pytest.approx(1.32, abs=0.05)
    assert m["overshoot_pct"] < 5 and m["steady_state_error_mv"] < 1.0
    assert 1.5 < m["load_dip_v"] < 2.5 and m["load_recovery_ms"] == pytest.approx(1.7, abs=0.1)
    assert m["resonance_hz"] == pytest.approx(723.4, abs=0.1)


def test_invalid_reference_is_rejected_by_the_backend():
    p = {**da.buck_defaults(), "vout_ref_v": 60.0}
    with pytest.raises(ValueError, match="below V_in"):
        da.buck_scenario(p)


def test_plant_overrides_reach_the_scenario():
    p = {**da.buck_defaults(), "l_h": 470e-6}
    s = da.buck_scenario(p)
    assert s.buck_parameter_overrides == {"l_h": 470e-6}
    assert da.buck_scenario(da.buck_defaults()).buck_parameter_overrides is None


def test_buck_sweep_spec_validates():
    spec = ja.build_sweep_spec("buck_v1", 0.005, [12.0, 24.0], 0)
    assert [c["buck_vout_ref_v"] for c in spec.conditions] == [12.0, 24.0]
    from driveflow.jobs.handlers import handler_for

    handler_for("scenario_sweep").validate(spec)


def test_buck_sweep_runs_through_the_worker(tmp_path):
    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker

    svc = JobService(tmp_path)
    job = svc.submit(ja.build_sweep_spec("buck_v1", 0.005, [12.0, 24.0], 0))
    assert Worker(tmp_path).serve(once=True) == 1
    run = svc.run_for(svc.get(job.job_id))
    assert run.status == "completed"
    errs = [c["metrics"]["rms_voltage_error_v_second_half"] for c in run.conditions]
    assert all(e is not None and e < 0.05 for e in errs)
