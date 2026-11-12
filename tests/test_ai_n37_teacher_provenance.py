"""AI-N37 (2026-11-12): a distilled run records WHICH teacher run it learned from in run_metadata.json -- the run the
registry resolved, or the --teacher-run given -- not only the registry key that metrics.json["distilled_from"] keeps."""

import json
from pathlib import Path

from driveflow.ai.registry import promote
from experiments.train_model import train_from_config
from tests.test_train_model import CLASSIFIER_CONFIG_YAML, RPI5_CLASSIFIER_CONFIG_YAML, dc_motor_dataset_path  # noqa: F401 (fixture)


def _teacher(tmp_path, dataset, registry_path=None):
    cfg = tmp_path / "configs" / "classifiers" / "tiny_pc.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(CLASSIFIER_CONFIG_YAML)
    run = train_from_config(cfg, dataset, epochs=1, batch_size=4, seed=0)
    if registry_path is not None:
        promote(run, registry_path=registry_path)
    return run


def _student(tmp_path):
    cfg = tmp_path / "configs" / "classifiers" / "rpi5.yaml"
    cfg.write_text(RPI5_CLASSIFIER_CONFIG_YAML)
    return cfg


def test_registry_teacher_is_recorded_as_the_resolved_run(tmp_path, dc_motor_dataset_path):
    registry_path = tmp_path / "registry.yaml"
    teacher = _teacher(tmp_path, dc_motor_dataset_path, registry_path)
    run = train_from_config(_student(tmp_path), dc_motor_dataset_path, epochs=1, batch_size=4, seed=0, distill=True, registry_path=registry_path)
    meta = json.loads((run / "run_metadata.json").read_text())
    assert Path(meta["teacher_run"]).resolve() == Path(teacher).resolve()
    assert meta["teacher_source"] == "registry (dc_motor/pc/classifier)"
    assert meta["distill"] is True and meta["distill_alpha"] == 0.5
    assert json.loads((run / "metrics.json").read_text())["distilled_from"] == "dc_motor/pc/classifier"  # metrics.json schema unchanged


def test_explicit_teacher_run_is_recorded(tmp_path, dc_motor_dataset_path):
    teacher = _teacher(tmp_path, dc_motor_dataset_path)
    run = train_from_config(_student(tmp_path), dc_motor_dataset_path, epochs=1, batch_size=4, seed=0, distill=True, distill_alpha=0.3, teacher_run=teacher)
    meta = json.loads((run / "run_metadata.json").read_text())
    assert Path(meta["teacher_run"]) == Path(teacher) and meta["teacher_source"] == "--teacher-run" and meta["distill_alpha"] == 0.3


def test_undistilled_run_has_no_teacher(tmp_path, dc_motor_dataset_path):
    run = _teacher(tmp_path, dc_motor_dataset_path)
    meta = json.loads((run / "run_metadata.json").read_text())
    assert meta["distill"] is False and "teacher_run" not in meta
