"""Tests driveflow/ai/c2000_export.py: the generated C reproduces Keras on the PC (gcc), the plan
rejects what the kernels cannot reproduce, and the memory verdict is enforced. Board measurements
live in docs/audit/c2000/."""

import json
import shutil
import subprocess
import zipfile
from io import BytesIO

import keras
import numpy as np
import pytest
from keras import layers

from driveflow.ai.c2000_export import ARENA_LIMIT_BYTES, C2000MemoryError, UnsupportedLayerError, export_c2000, host_parity, kit_zip, plan_model
from driveflow.ai.calibration import decide
from driveflow.ai.deployment_status import c2000_status, deployment_status
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig
from driveflow.models.regressors.builder import build_forecaster
from driveflow.models.regressors.schemas import ForecasterConfig

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
STEPS = 64


def _raw_windows(n_channels, n=3):
    rng = np.random.default_rng(0)
    raw = rng.normal(5.0, 2.0, (n, STEPS, n_channels)).astype(np.float32)
    raw[:, :, -1] = 7.5  # a dead channel: training zeroes it
    return raw


def _classifier(block_type, se=False):
    config = ClassifierConfig(domain="dc_motor", tier="esp32", input_window=STEPS, num_classes=3,
                              blocks=(ConvBlockConfig(filters=4, kernel_size=3, block_type=block_type, use_se=se), ConvBlockConfig(filters=4, kernel_size=4, block_type=block_type, use_se=se)),
                              dense_units=(), dropout=0.0)
    return build_classifier(config, n_channels=3)


def _randomize(model):
    """Trained-looking statistics: a fresh BatchNormalization is an identity and would hide errors."""
    rng = np.random.default_rng(1)
    model.set_weights([rng.normal(0.3, 0.2, w.shape).astype(np.float32) if w.ndim else w for w in model.get_weights()])
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            layer.set_weights([gamma, beta, mean, np.abs(var) + 0.5])
    return model


METRICS = {"channels": ["a", "b", "c"], "classes": ["x", "y", "z"], "level_scaling": {"b": [5.0, 2.0]}, "calibration": {"temperature": 0.9, "abstain_below": 0.5}}


@needs_gcc
class TestHostParity:
    @pytest.mark.parametrize("block_type,se", [("dsconv1d", False), ("conv1d", False), ("conv1d", True)])
    def test_classifier_matches_keras(self, tmp_path, block_type, se):
        model = _randomize(_classifier(block_type, se))
        export_c2000(model, METRICS, _raw_windows(3), tmp_path / "kit")
        report = host_parity(tmp_path / "kit")
        assert report["passed"], report

    def test_causal_dilated_forecaster_matches_keras(self, tmp_path):
        config = ForecasterConfig(domain="vsc_dpc", tier="esp32", input_window=STEPS, horizon=8, recurrent_type="none", layers=(4, 4, 4), use_attention=False)
        model = _randomize(build_forecaster(config, n_channels=3))
        metrics = {"channels": ["a", "b", "c"]}
        export_c2000(model, metrics, _raw_windows(3), tmp_path / "kit")
        report = host_parity(tmp_path / "kit")
        assert report["passed"], report

    def test_maxpool_and_add_match_keras(self, tmp_path):
        inputs = keras.Input((STEPS, 3))
        a = layers.Conv1D(3, 3, padding="same")(inputs)
        x = layers.Add()([a, inputs])
        x = layers.MaxPooling1D(3)(x)
        x = layers.GlobalAveragePooling1D()(x)
        model = _randomize(keras.Model(inputs, layers.Dense(3, activation="softmax")(x)))
        export_c2000(model, METRICS, _raw_windows(3), tmp_path / "kit")
        report = host_parity(tmp_path / "kit")
        assert report["passed"], report

    def test_decide_matches_the_python_rule(self, tmp_path):
        model = _randomize(_classifier("dsconv1d"))
        kit = tmp_path / "kit"
        export_c2000(model, METRICS, _raw_windows(3), kit)
        (kit / "decide_main.c").write_text('#include <stdio.h>\n#include "spade_nn.h"\nint main(void){float p[3]={0.41f,0.35f,0.24f};float q[3]={0.8f,0.15f,0.05f};int a=spade_decide(p,3,0.9f,0.5f);int b=spade_decide(q,3,0.9f,0.5f);printf("%d %d %.6f\\n",a,b,q[0]);return 0;}\n')
        subprocess.run(["gcc", "-I.", "decide_main.c", "spade_nn.c", "-lm", "-o", "decide"], cwd=kit, check=True)
        a, b, top_p = subprocess.run([str(kit / "decide")], check=True, capture_output=True, text=True).stdout.split()
        assert (a, b) == ("-1", "0")
        cal = {"calibration": METRICS["calibration"]}
        assert decide(np.array([0.41, 0.35, 0.24]), METRICS["classes"], cal)[0] != "x"
        assert float(top_p) == pytest.approx(decide(np.array([0.8, 0.15, 0.05]), METRICS["classes"], cal)[1][0], abs=1e-5)


class TestPlan:
    @pytest.mark.parametrize("layer", [layers.GRU(4, return_sequences=True), layers.LSTM(4, return_sequences=True), layers.Conv1D(4, 3, strides=2, padding="same"), layers.Conv1D(4, 3, padding="valid"), layers.Lambda(lambda x: x * 2.0)])
    def test_rejects_what_the_kernels_do_not_reproduce(self, layer):
        inputs = keras.Input((STEPS, 3))
        model = keras.Model(inputs, layer(inputs))
        with pytest.raises(UnsupportedLayerError):
            plan_model(model)

    def test_buffers_that_never_coexist_share_the_arena(self):
        plan = plan_model(_classifier("dsconv1d"))
        total = sum(view.size for view in {id(v.root): v for v in plan.views.values()}.values())
        assert plan.arena_floats < total

    def test_a_model_too_big_for_the_board_writes_no_kit(self, tmp_path):
        inputs = keras.Input((8192, 6))
        model = keras.Model(inputs, layers.GlobalAveragePooling1D()(layers.Conv1D(32, 3, padding="same")(inputs)))
        assert plan_model(model).arena_bytes > ARENA_LIMIT_BYTES
        with pytest.raises(C2000MemoryError):
            export_c2000(model, {"channels": list("abcdef")}, np.zeros((1, 8192, 6), dtype=np.float32), tmp_path / "kit")
        assert not (tmp_path / "kit").exists()


class TestArtifacts:
    def test_status_and_zip(self, tmp_path):
        model = _randomize(_classifier("dsconv1d"))
        run_dir = tmp_path / "run"
        manifest = export_c2000(model, METRICS, _raw_windows(3), run_dir / "c2000", source_run="x/y")
        assert manifest["ood_ported"] is False and manifest["memory"]["fits"]
        (run_dir / "c2000" / "build").mkdir()
        (run_dir / "c2000" / "build" / "spade_model.out").write_text("binary")
        assert c2000_status(run_dir)["run"] is None
        status = deployment_status(run_dir)
        assert status["stages"]["exported"] == "not_exported" and status["c2000"]["manifest"]["source_run"] == "x/y"
        names = zipfile.ZipFile(BytesIO(kit_zip(run_dir / "c2000"))).namelist()
        assert "spade_c2000/spade_model.c" in names and not any("build" in n for n in names)

    def test_reexport_drops_the_previous_board_evidence(self, tmp_path):
        model = _randomize(_classifier("dsconv1d"))
        kit = tmp_path / "run" / "c2000"
        export_c2000(model, METRICS, _raw_windows(3), kit)
        (kit / "hardware_run.json").write_text(json.dumps({"stale": True}))
        export_c2000(model, METRICS, _raw_windows(3), kit)
        assert not (kit / "hardware_run.json").exists()
