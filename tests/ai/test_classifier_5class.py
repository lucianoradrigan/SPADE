"""A2 (TODO_CLAUDE_CODE.md): the 5-class dc_motor classifiers (healthy, outer_race, inner_race,
ball, cage). Candidates, NOT promoted; ball/cage are synthetic-only classes with an uncalibrated
amplitude. docs/AI_TECHNICAL_AUDIT.md Sec. 8.4, docs/audit/ai/classifier_evaluation_5class.md."""

import json
from pathlib import Path

import pandas as pd
import pytest

from driveflow.ai.registry import resolve
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.export_parquet import records_to_dataframe

from .test_classifier_evaluation import _recall

pytestmark = pytest.mark.ai

ROOT = Path(__file__).resolve().parents[2]
RUNS = [ROOT / "configs" / "classifiers" / f"{c}_5class" / "2026-10-20_run01" for c in ("pc_server", "rpi5_resnet1d_se", "esp32_dscnn")]
CLASSES = ["ball", "cage", "inner_race", "normal", "outer_race"]


@pytest.mark.parametrize("run_dir", RUNS, ids=lambda p: p.parent.name)
def test_run_carries_the_synthetic_only_label(run_dir):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["classes"] == CLASSES
    assert metrics["synthetic_only_classes"] == ["ball", "cage"]
    findings = json.loads((run_dir / "audit_findings.json").read_text())["findings"]
    synthetic = next(f for f in findings if f["id"] == "A2-synthetic-ball-cage")
    assert "synthetic-only, uncalibrated amplitude" in synthetic["summary"]


def test_five_class_models_are_not_promoted():
    for tier in ("pc", "rpi5", "esp32"):
        assert "5class" not in str(resolve("dc_motor", tier, "classifier"))


@pytest.fixture(scope="module")
def fresh_five_classes():
    """One fresh run per class (seed 901, never used in training), steady state t in [0.5, 1.2) s."""
    frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"fresh5_{ft}", fault_type=ft, duration_s=1.2, seed=901)))
              for ft in (None, "outer_race", "inner_race", "ball", "cage")]
    df = pd.concat(frames, ignore_index=True)
    return df[df["timestamp_s"] >= 0.5]


@pytest.mark.parametrize("cls", CLASSES)
def test_pc_5class_recognizes_every_class_at_the_nominal_point(fresh_five_classes, cls):
    # Measured 2026-10-20 on the held-out evaluation: 1.000 for every class at 150 rad/s steady state.
    assert _recall(RUNS[0], fresh_five_classes, cls) >= 0.9, cls
