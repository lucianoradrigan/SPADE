"""X4 (round PLECS/OM): the promoted ESP32 classifier has 3 classes while PC and RPi5 have 5; the
Model Registry says so next to that entry, and the run's audit_findings.json carries it (AI-N13)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from driveflow.ai.audit_findings import load_audit_findings
from driveflow.ai.registry import REPO_ROOT, resolve
from driveflow.viz.adapters.registry_adapter import class_coverage_warnings, list_runs

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def test_the_three_class_esp32_classifier_is_the_one_flagged():
    warnings = class_coverage_warnings(list_runs())
    esp32 = Path(resolve("dc_motor", "esp32", "classifier"))
    assert len(warnings) == 1
    assert f"({esp32.parent.name}/{esp32.name})" in warnings[0]
    assert "ball or cage" in warnings[0] and "AI-N13" in warnings[0]


def test_the_esp32_run_records_the_class_coverage_finding():
    run = REPO_ROOT / resolve("dc_motor", "esp32", "classifier")
    assert any(f["id"] == "three-class-vs-five" and "AI-N13" in f["summary"] for f in load_audit_findings(run))


def test_model_registry_shows_the_class_coverage_warning():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/model_registry.py").run()
    assert not at.exception
    assert any("Class coverage" in str(w.value) and "esp32" in str(w.value) for w in at.warning)
    table = next(d.value for d in at.dataframe if "class limitation" in getattr(d.value, "columns", []))
    assert set(table.loc[table["class limitation"] == "fewer classes than other tiers (AI-N13)", "run"]) == {"esp32_dscnn/2026-10-18_run01"}
