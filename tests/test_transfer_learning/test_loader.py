"""Tests for ModelLoader (Phase 2): loads a real, promoted tiny classifier run and confirms the
returned model actually predicts, not just that construction didn't raise. Reuses
tests/test_train_model.py's tiny synthetic fixtures (a real trained run, not a claim about model
quality -- see that module's docstring) the same way tests/test_registry.py already does."""

import numpy as np
import pytest

from driveflow.ai.registry import RegistryError, promote
from driveflow.ai.transfer.loader import ModelLoader
from tests.test_train_model import (
    CLASSIFIER_CONFIG_YAML,
    dc_motor_dataset_path,  # noqa: F401 -- pytest fixture, referenced by name below
)
from experiments.train_model import train_from_config


@pytest.fixture
def classifier_run_dir(tmp_path, dc_motor_dataset_path):
    config_path = tmp_path / "configs" / "classifiers" / "tiny_pc.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CLASSIFIER_CONFIG_YAML)
    return train_from_config(config_path, dc_motor_dataset_path, epochs=1, batch_size=4, seed=0)


@pytest.fixture
def promoted_registry_path(tmp_path, classifier_run_dir):
    registry_path = tmp_path / "registry.yaml"
    promote(classifier_run_dir, registry_path=registry_path)
    return registry_path


class TestLoadLatest:
    def test_loads_a_working_model(self, promoted_registry_path):
        loader = ModelLoader(registry_path=promoted_registry_path, domain="dc_motor", block="classifier", tier="pc")
        result = loader.load_latest()
        assert set(result) == {"model", "config", "metrics", "run_dir"}
        n_channels = len(result["metrics"]["channels"])
        window = np.zeros((1, result["config"].input_window, n_channels), dtype="float32")
        probs = result["model"](window, training=False).numpy()
        assert probs.shape == (1, result["config"].num_classes)
        assert np.isclose(probs.sum(), 1.0, atol=1e-4)

    def test_raises_when_nothing_promoted(self, tmp_path):
        loader = ModelLoader(registry_path=tmp_path / "empty_registry.yaml", domain="dc_motor", block="classifier", tier="pc")
        with pytest.raises(RegistryError):
            loader.load_latest()


class TestLoadByGeneration:
    def test_loads_the_same_run_directly_by_path(self, classifier_run_dir, promoted_registry_path):
        loader = ModelLoader(registry_path=promoted_registry_path, domain="dc_motor", block="classifier", tier="pc")
        result = loader.load_by_generation(classifier_run_dir)
        assert result["run_dir"] == classifier_run_dir

    def test_raises_a_clear_error_for_a_missing_run(self, tmp_path, promoted_registry_path):
        loader = ModelLoader(registry_path=promoted_registry_path, domain="dc_motor", block="classifier", tier="pc")
        with pytest.raises(RegistryError, match="no such run directory"):
            loader.load_by_generation(tmp_path / "does_not_exist")


class TestGetMetadata:
    def test_returns_config_and_metrics_without_building_a_model(self, promoted_registry_path):
        loader = ModelLoader(registry_path=promoted_registry_path, domain="dc_motor", block="classifier", tier="pc")
        meta = loader.get_metadata()
        assert meta["config"]["domain"] == "dc_motor"
        assert "test_accuracy" in meta["metrics"]
