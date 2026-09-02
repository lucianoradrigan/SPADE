"""Integration test for experiments/train_model.py (docs/design_ai_layer_transversal.md Sec. 6.4,
Sec. 8 step 5): end-to-end proof that config -> load data -> build model -> train -> evaluate ->
save the three artifacts (config.yaml, model.weights.h5, metrics.json) actually works, for both
config kinds (classifier/dc_motor and forecaster/vsc_dpc) it's meant to dispatch between. Uses
tiny synthetic scenarios (small input_window, 1 epoch) purely to exercise the pipeline's
plumbing -- NOT a claim about Fase C/D.1 model quality, see train_model.py's module docstring.
"""

import json

import pandas as pd
import pytest

from driveflow.datagen import Scenario, export_parquet, run_scenario
from experiments.train_model import train_from_config

CLASSIFIER_CONFIG_YAML = """\
domain: dc_motor
tier: pc
input_window: 64
num_classes: 2
blocks:
  - type: conv1d
    filters: 4
    kernel_size: 3
    use_se: false
dense_units: [8]
dropout: 0.0
"""

FORECASTER_CONFIG_YAML = """\
domain: vsc_dpc
tier: pc
input_window: 16
horizon: 8
recurrent_type: lstm
layers: [8]
use_attention: false
"""


@pytest.fixture(scope="module")
def dc_motor_dataset_path(tmp_path_factory):
    scenarios = [
        Scenario(scenario_id=f"train_dc_{fault}_{seed}", fault_type=fault, duration_s=0.1, seed=seed)
        for fault in (None, "outer_race")
        for seed in (0, 1, 2)
    ]
    records = [r for s in scenarios for r in run_scenario(s)]
    out_path = tmp_path_factory.mktemp("data") / "dc_motor.parquet"
    export_parquet(records, out_path)
    return out_path


@pytest.fixture(scope="module")
def vsc_dpc_dataset_path(tmp_path_factory):
    scenarios = [
        Scenario(scenario_id=f"train_vsc_{seed}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.01, seed=seed)
        for seed in (0, 1, 2, 3)
    ]
    records = [r for s in scenarios for r in run_scenario(s)]
    out_path = tmp_path_factory.mktemp("data") / "vsc_dpc.parquet"
    export_parquet(records, out_path)
    return out_path


class TestTrainClassifierFromConfig:
    def test_saves_the_three_artifacts(self, tmp_path, dc_motor_dataset_path):
        config_path = tmp_path / "configs" / "classifiers" / "tiny_pc.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(CLASSIFIER_CONFIG_YAML)

        run_dir = train_from_config(config_path, dc_motor_dataset_path, epochs=1, batch_size=4, seed=0)

        assert (run_dir / "config.yaml").read_text() == CLASSIFIER_CONFIG_YAML
        assert (run_dir / "model.weights.h5").exists()
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["kind"] == "classifier"
        assert metrics["domain"] == "dc_motor"
        assert metrics["n_train"] > 0
        assert 0.0 <= metrics["test_accuracy"] <= 1.0

    def test_second_run_gets_a_distinct_run_dir(self, tmp_path, dc_motor_dataset_path):
        config_path = tmp_path / "configs" / "classifiers" / "tiny_pc.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(CLASSIFIER_CONFIG_YAML)

        run_dir_1 = train_from_config(config_path, dc_motor_dataset_path, epochs=1, batch_size=4, seed=0)
        run_dir_2 = train_from_config(config_path, dc_motor_dataset_path, epochs=1, batch_size=4, seed=0)
        assert run_dir_1 != run_dir_2
        assert run_dir_1.exists() and run_dir_2.exists()


class TestTrainForecasterFromConfig:
    def test_saves_the_three_artifacts(self, tmp_path, vsc_dpc_dataset_path):
        config_path = tmp_path / "configs" / "regressors" / "tiny_pc.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(FORECASTER_CONFIG_YAML)

        run_dir = train_from_config(config_path, vsc_dpc_dataset_path, epochs=1, batch_size=4, seed=0)

        assert (run_dir / "config.yaml").read_text() == FORECASTER_CONFIG_YAML
        assert (run_dir / "model.weights.h5").exists()
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["kind"] == "forecaster"
        assert metrics["domain"] == "vsc_dpc"
        assert metrics["n_train"] > 0
        assert metrics["test_rmse"] is None or metrics["test_rmse"] >= 0.0


class TestConfigKindDetection:
    def test_unrecognized_config_raises_a_clear_error(self, tmp_path, dc_motor_dataset_path):
        config_path = tmp_path / "ambiguous.yaml"
        config_path.write_text("domain: dc_motor\ntier: pc\n")
        with pytest.raises(ValueError, match="cannot tell"):
            train_from_config(config_path, dc_motor_dataset_path)
