"""T3 (round 2026-10-22): every run that can answer ball or cage carries
"ball/cage: synthetic-only, uncalibrated amplitude", in its audit_findings.json and next to any
ball/cage prediction in the GUI."""

import json
from pathlib import Path

import keras
import pytest
from streamlit.testing.v1 import AppTest

from driveflow.ai.audit_findings import SYNTHETIC_LABEL, ensure_synthetic_class_finding, load_audit_findings
from driveflow.ai.registry import REPO_ROOT

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")
RUN_ROOTS = [REPO_ROOT / "configs" / "classifiers"]


def _ball_cage_runs():
    for root in RUN_ROOTS:
        for metrics_path in sorted(root.glob("**/metrics.json")):
            classes = set(json.loads(metrics_path.read_text()).get("classes") or [])
            if classes & {"ball", "cage"}:
                yield metrics_path.parent


def test_there_are_ball_cage_runs_to_check():
    assert len(list(_ball_cage_runs())) >= 6  # PC, RPi5 and ESP32, 512 and 2048 windows


@pytest.mark.parametrize("run_dir", list(_ball_cage_runs()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_ball_cage_run_carries_the_finding(run_dir):
    assert any(SYNTHETIC_LABEL in str(f.get("summary")) for f in load_audit_findings(run_dir)), f"{run_dir} lacks '{SYNTHETIC_LABEL}'"


def test_helper_adds_the_finding_once_and_only_for_ball_or_cage(tmp_path):
    assert not ensure_synthetic_class_finding(tmp_path, ["normal", "outer_race"])
    assert not (tmp_path / "audit_findings.json").exists()
    assert ensure_synthetic_class_finding(tmp_path, ["normal", "cage"])
    assert not ensure_synthetic_class_finding(tmp_path, ["normal", "cage"])
    assert sum(SYNTHETIC_LABEL in f["summary"] for f in load_audit_findings(tmp_path)) == 1


def test_an_imported_model_with_ball_gets_the_finding(tmp_path):
    from driveflow.ai.transfer import external as ext

    inp = keras.Input((64, 2))
    out = keras.layers.Dense(2, activation="softmax")(keras.layers.GlobalAveragePooling1D()(inp))
    p = tmp_path / "m.keras"
    keras.Model(inp, out).save(p)
    run_dir = ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r", "acc_x"], ["normal", "ball"], root=tmp_path / "ext")
    assert any(SYNTHETIC_LABEL in f["summary"] for f in load_audit_findings(run_dir))


def test_a_gui_trained_model_with_cage_gets_the_finding(tmp_path):
    from driveflow.ai import custom_training as ct

    r = ct.default_request("dc_motor")
    r["classes"] = [ct.dc_class(ct.HEALTHY), ct.dc_class("cage", "cage")]
    r["operating"].update(setpoints=[150.0], severity_scales=[1.0], runs_per_cell=3, duration_s=0.15, skip_start_s=0.05, drift={}, noise_max_pct=0.0)
    r["model"].update(preset="esp32_dscnn", input_window=256, channels=["current_r", "acc_x"], epochs=1, cap_per_class=60, keep_rpm_level=False, fit_ood=False)
    df, _ = ct.simulate(ct.build_conditions(r))
    run_dir = ct.train(r, df, root=tmp_path / "m", dataset_root=tmp_path / "d")
    assert any(SYNTHETIC_LABEL in f["summary"] for f in load_audit_findings(run_dir))


def _text(at):
    return " ".join(str(e.value) for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind))


def test_ai_diagnostics_labels_a_ball_prediction():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/ai_diagnostics.py").run()
    at.selectbox(key="ia_fault_type").set_value("ball").run()
    at.button(key="ia_generate").click().run()
    assert not at.exception
    assert SYNTHETIC_LABEL in _text(at)


def test_model_registry_shows_the_limitation_for_every_ball_cage_run():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/model_registry.py").run()
    assert not at.exception
    table = next(d.value for d in at.dataframe if "class limitation" in getattr(d.value, "columns", []))
    flagged = set(table.loc[table["class limitation"] == SYNTHETIC_LABEL, "run"])
    expected = {f"{p.parent.name}/{p.name}" for p in _ball_cage_runs() if p.parent.parent.name == "classifiers"}
    assert expected and expected <= flagged
    assert "pc_server/2026-10-18_run01" not in flagged
