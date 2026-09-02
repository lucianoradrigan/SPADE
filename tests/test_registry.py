"""Tests driveflow/ai/registry.py (docs/design_ai_layer_transversal.md Sec. 7, Sec. 8 step 6):
promote() marks a train_model.py run directory as production for its (domain, tier, block),
resolve() looks that up, and load_promoted_model() returns a ready-to-use, weight-loaded
keras.Model. Reuses tests/test_train_model.py's tiny synthetic fixtures to produce real (if
tiny) run directories to promote -- not a claim about model quality, see that module's docstring.
"""

import numpy as np
import pytest
import tensorflow as tf

from driveflow.ai.registry import RegistryError, load_promoted_model, promote, resolve
from experiments.train_model import train_from_config
from tests.test_train_model import (
    CLASSIFIER_CONFIG_YAML,
    FORECASTER_CONFIG_YAML,
    dc_motor_dataset_path,  # noqa: F401 -- pytest fixture, referenced by name in test signatures
    vsc_dpc_dataset_path,  # noqa: F401
)


@pytest.fixture
def classifier_run_dir(tmp_path, dc_motor_dataset_path):
    config_path = tmp_path / "configs" / "classifiers" / "tiny_pc.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CLASSIFIER_CONFIG_YAML)
    return train_from_config(config_path, dc_motor_dataset_path, epochs=1, batch_size=4, seed=0)


@pytest.fixture
def forecaster_run_dir(tmp_path, vsc_dpc_dataset_path):
    config_path = tmp_path / "configs" / "regressors" / "tiny_pc.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(FORECASTER_CONFIG_YAML)
    return train_from_config(config_path, vsc_dpc_dataset_path, epochs=1, batch_size=4, seed=0)


class TestResolveWithoutPromotion:
    def test_raises_a_clear_error(self, tmp_path):
        registry_path = tmp_path / "registry.yaml"
        with pytest.raises(RegistryError, match="no promoted run"):
            resolve("dc_motor", "pc", "classifier", registry_path=registry_path)


class TestPromote:
    def test_promoted_run_resolves_back(self, tmp_path, classifier_run_dir):
        registry_path = tmp_path / "registry.yaml"
        entry = promote(classifier_run_dir, registry_path=registry_path)
        assert entry.domain == "dc_motor"
        assert entry.tier == "pc"
        assert entry.block == "classifier"

        resolved = resolve("dc_motor", "pc", "classifier", registry_path=registry_path)
        assert resolved == classifier_run_dir.resolve()

    def test_forecaster_block_auto_detected(self, tmp_path, forecaster_run_dir):
        registry_path = tmp_path / "registry.yaml"
        entry = promote(forecaster_run_dir, registry_path=registry_path)
        assert entry.domain == "vsc_dpc"
        assert entry.block == "regressor"

    def test_promoting_again_overwrites_not_duplicates(self, tmp_path, classifier_run_dir, dc_motor_dataset_path):
        registry_path = tmp_path / "registry.yaml"
        promote(classifier_run_dir, registry_path=registry_path)

        config_path = classifier_run_dir / "config.yaml"
        second_run_dir = train_from_config(config_path, dc_motor_dataset_path, epochs=1, batch_size=4, seed=1)
        promote(second_run_dir, registry_path=registry_path)

        resolved = resolve("dc_motor", "pc", "classifier", registry_path=registry_path)
        assert resolved == second_run_dir.resolve()

    def test_missing_artifact_raises(self, tmp_path, classifier_run_dir):
        (classifier_run_dir / "metrics.json").unlink()
        registry_path = tmp_path / "registry.yaml"
        with pytest.raises(RegistryError, match="metrics.json"):
            promote(classifier_run_dir, registry_path=registry_path)


class TestLoadPromotedModel:
    def test_classifier_loads_and_runs(self, tmp_path, classifier_run_dir):
        registry_path = tmp_path / "registry.yaml"
        promote(classifier_run_dir, registry_path=registry_path)

        model, config, metrics = load_promoted_model("dc_motor", "pc", "classifier", registry_path=registry_path)
        out = model(tf.zeros((1, config.input_window, len(metrics["channels"]))))
        assert out.shape == (1, config.num_classes)
        assert np.allclose(np.sum(out.numpy(), axis=1), 1.0, atol=1e-5)

    def test_forecaster_loads_and_runs(self, tmp_path, forecaster_run_dir):
        registry_path = tmp_path / "registry.yaml"
        promote(forecaster_run_dir, registry_path=registry_path)

        model, config, metrics = load_promoted_model("vsc_dpc", "pc", "regressor", registry_path=registry_path)
        out = model(tf.zeros((1, config.input_window, len(metrics["channels"]))))
        assert out.shape == (1, config.horizon, len(metrics["channels"]))

    def test_unpromoted_block_raises(self, tmp_path, classifier_run_dir):
        registry_path = tmp_path / "registry.yaml"
        promote(classifier_run_dir, registry_path=registry_path)
        with pytest.raises(RegistryError, match="no promoted run"):
            load_promoted_model("dc_motor", "pc", "regressor", registry_path=registry_path)
