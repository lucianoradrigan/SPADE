"""UI acceptance tests T-UI-01..T-UI-05 (README_FOR_CLAUDE.md Sec. 6) plus design-token sync.

T-UI-06 (responsive clipping at 1366x768 / 1920x1080) needs a real browser; it was checked with
real screenshots (docs/screenshots/) and is not automated here."""

import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from driveflow.ai.registry import RegistryError
from driveflow.analysis import amplitude_spectrum, dominant_peak
from driveflow.monitoring.motor_status import MotorViewModel
from driveflow.viz.adapters.registry_adapter import artifacts, list_runs
from driveflow.viz.components.motor_status import motor_status_html, rotation_period_s
from driveflow.viz.theme.tokens import DARK, LIGHT

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")
PAGES = ["overview", "experiment_runs", "dc_motor", "pmsm", "vsc_dpc", "dcdc_converters", "plecs_plant_studio", "tuning_studio", "rl_lab", "ai_diagnostics", "edge_deployment", "transfer_learning", "model_registry", "monitoring", "alert_history", "documentation", "algorithm_guide"]
#: Spanish UI words that used to appear in the legacy dashboard ("Fase", tab titles) -- the
#: frontend must be English (guide Sec. 0).
_SPANISH = re.compile(r"\b(Fase|Corrientes|Envolvente|Plano|Generar|vibración|Evaluar|Monitoreo)\b")


def _page(name: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.switch_page(f"pages/{name}.py").run()
    return at


def _visible_text(at) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind) if isinstance(getattr(e, "value", None), str)]
    parts += [str(getattr(e, "label", "")) for kind in ("button", "selectbox", "slider", "radio", "checkbox", "number_input", "tab", "expander") for e in at.get(kind)]
    parts += [str(getattr(h, "body", "")) for h in at.get("html")]
    return " ".join(parts)


# ----------------------------------------------------------------------------------- T-UI-01
@pytest.mark.parametrize("page", PAGES)
def test_every_page_opens_without_exceptions_and_in_english(page):
    at = _page(page)
    assert not at.exception, at.exception
    text = _visible_text(at)
    assert not _SPANISH.findall(text), _SPANISH.findall(text)


# ----------------------------------------------------------------------------------- T-UI-02
def test_results_are_frozen_per_run_and_survive_navigation():
    at = _page("dc_motor")
    at.slider(key="dc_duration_slider").set_value(0.1).run()
    at.button(key="dc_run_button").click().run()
    run1 = at.session_state["dc_result"]["run"]
    assert run1["params"]["duration_s"] == 0.1
    at.switch_page("pages/vsc_dpc.py").run()
    at.button(key="dpc_run_button").click().run()
    at.switch_page("pages/dc_motor.py").run()
    assert at.session_state["dc_result"]["run"]["run_id"] == run1["run_id"]
    at.slider(key="dc_duration_slider").set_value(0.2).run()  # change a control, do not re-run
    assert at.session_state["dc_result"]["run"]["params"]["duration_s"] == 0.1  # not relabelled
    assert any("Controls changed since this result" in w.value for w in at.warning)
    assert at.session_state["dpc_result"]["run"]["run_id"] != run1["run_id"]


# ----------------------------------------------------------------------------------- T-UI-03
def test_registry_lists_only_existing_artifacts_with_correct_hashes():
    import hashlib

    runs = [r for r in list_runs() if "error" not in r]
    assert runs
    for r in runs:
        files = artifacts(r["run_dir"])
        names = {a["file"] for a in files}
        assert "model.weights.h5" in names and "metrics.json" in names
        assert not names & {"model.onnx", "model.pt"}
        for a in files:
            assert a["path"].exists() and a["size_bytes"] == a["path"].stat().st_size
            assert a["sha256"] == hashlib.sha256(a["path"].read_bytes()).hexdigest()
        assert r["exported"] == (r["run_dir"] / "model.tflite").exists()
        assert r["hardware_verified"] is False  # no deployment evidence exists in the repo


def test_recorded_metric_is_never_listed_without_its_audit_finding():
    # AUDIT Q15: the promoted dc_motor classifier's 0.895 never measured inner_race. Since then every
    # promoted classifier ships its known limits (e.g. 2026-10-18: no ball class, simulated data only).
    promoted = [r for r in list_runs() if r.get("status") == "Promoted" and r["domain"] == "dc_motor" and r["block"] == "classifier"]
    assert promoted
    for r in promoted:
        assert r["audit_findings"], r["run"]
        assert "audit_findings.json" in {a["file"] for a in artifacts(r["run_dir"])}


def test_flex_ops_in_an_export_are_reported():
    # AUDIT Q22: the rpi5 GRU regressor was exported with SELECT_TF_OPS and does not run on a
    # builtin-only interpreter; every ESP32 export must stay builtin-only (TFLite Micro).
    by_run = {r["run"]: r for r in list_runs() if r.get("exported")}
    assert by_run["rpi5_edge/2026-09-13_run01"]["deployment"]["artifact"]["flex_ops"]
    for r in by_run.values():
        if r["tier"] == "esp32":
            assert r["deployment"]["artifact"]["flex_ops"] == [], r["run"]


def test_dc_export_matches_the_displayed_run():
    at = _page("dc_motor")
    at.slider(key="dc_duration_slider").set_value(0.1).run()
    at.button(key="dc_run_button").click().run()
    df = at.session_state["dc_result"]["df"]
    assert len(df) == 1000  # 0.1 s at 10 kHz
    assert not at.exception


# ----------------------------------------------------------------------------------- T-UI-04
def _view(**kw):
    base = dict(motor_id="m", domain="dc_motor", source_mode="simulated", status="healthy", rpm=1500.0)
    return MotorViewModel(**{**base, **kw})


@pytest.mark.parametrize("status", ["healthy", "warning", "critical", "offline", "unknown"])
def test_motor_widget_reflects_the_view_model_status_in_text_and_attribute(status):
    html = motor_status_html(_view(status=status))
    assert f'data-status="{status}"' in html
    assert status.upper() in html  # text label, not colour only
    assert 'role="img"' in html and "aria-label" in html


def test_motor_rotation_stops_at_zero_rpm_and_offline():
    assert "--play:paused" in motor_status_html(_view(rpm=0.0))
    assert "--play:paused" in motor_status_html(_view(rpm=1500.0, status="offline"))
    assert "--play:paused" in motor_status_html(_view(rpm=None))
    assert "--play:running" in motor_status_html(_view(rpm=1500.0))
    assert rotation_period_s(3000) < rotation_period_s(1500)  # scales with speed
    assert "prefers-reduced-motion: reduce" in motor_status_html(_view())


def test_ai_suggestion_is_not_an_alarm_in_the_widget():
    html = motor_status_html(_view(status="unknown", prediction="outer_race", confidence=0.8, model_id="pc_server/run01"))
    assert "not an alarm" in html and 'data-status="unknown"' in html


def test_invalid_status_is_rejected():
    with pytest.raises(ValueError):
        _view(status="green")


# ----------------------------------------------------------------------------------- T-UI-05
def test_backend_failure_is_reported_honestly(monkeypatch):
    import driveflow.viz.views.ai_diagnostics as ai

    def _raise(*a, **k):
        raise RegistryError("registry unavailable (simulated failure)")

    monkeypatch.setattr(ai, "promoted_model", _raise)
    monkeypatch.setattr(ai.da_ai, "promoted_model", _raise)
    at = _page("ai_diagnostics")
    at.button(key="ia_generate").click().run()
    assert not at.exception
    info = " ".join(i.value for i in at.info)
    assert "No classifier is registered" in info and "No regressor is registered" in info
    assert "PRODUCTION" not in _visible_text(at).upper()


def test_nothing_is_labelled_live_or_production_ready():
    for page in ("overview", "monitoring", "model_registry"):
        text = _visible_text(_page(page)).upper()
        for banned in ("PRODUCTION READY", "LIVE / CONNECTED", "ENGINE READY", "HARDWARE VERIFIED ✓"):
            assert banned not in text, (page, banned)


# ----------------------------------------------------------------------------------- tokens / numerics
def test_streamlit_theme_config_matches_design_tokens():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    for name, tokens in (("dark", DARK), ("light", LIGHT)):
        t = cfg["theme"][name]
        assert t["primaryColor"] == tokens["primary"]
        assert t["backgroundColor"] == tokens["bg_app"]
        assert t["textColor"] == tokens["text"]
        assert t["borderColor"] == tokens["border"]
        assert t["sidebar"]["backgroundColor"] == tokens["bg_nav"]


def test_amplitude_spectrum_recovers_known_sine():
    fs = 10_000.0
    t = np.arange(10_000) / fs
    f, a = amplitude_spectrum(3.0 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 870 * t), fs)
    peak_f, peak_a = dominant_peak(f, a)
    assert peak_f == pytest.approx(120.0) and peak_a == pytest.approx(3.0, rel=1e-3)
    assert a[np.argmin(abs(f - 870))] == pytest.approx(0.5, rel=1e-3)
    with pytest.raises(ValueError):
        amplitude_spectrum([1.0, np.nan] * 10, fs)


def test_every_overview_video_guide_is_embedded_and_in_the_readme():
    from driveflow.viz.views.overview import VIDEO_GUIDES

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    assert len(VIDEO_GUIDES) == 4
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for video, *_ in VIDEO_GUIDES:
        assert f"https://www.youtube.com/watch?v={video}" in readme


def test_readme_credits_every_author_and_the_citation_shown_in_the_app():
    from driveflow.viz.components.citation import AUTHORS, CITATION_BIBTEX, CITATION_TEXT, PROFILES

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert CITATION_TEXT in readme and CITATION_BIBTEX in readme
    assert all(name in readme and mail in readme for name, mail, _ in AUTHORS)
    assert all(url in readme for links in PROFILES.values() for _, url in links)


@pytest.mark.parametrize("page", ["overview", "documentation"])
def test_the_citation_is_shown_in_the_app(page):
    from driveflow.viz.components.citation import CITATION_TEXT

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.switch_page(f"pages/{page}.py").run()
    assert not at.exception
    assert CITATION_TEXT in [c.value for c in at.code]

