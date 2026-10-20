"""scripts/edge_benchmark (TODO_CLAUDE_CODE.md P3) runs end to end on the host. This is NOT a
hardware test: the RPi5 script falls back to TensorFlow's interpreter here and says so."""

import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import tensorflow as tf

from driveflow.ai.tflite_export import export_float16
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig

KIT = Path(__file__).resolve().parents[1] / "scripts" / "edge_benchmark"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rpi5_benchmark_runs_on_host_and_reports_the_fallback(tmp_path):
    config = ClassifierConfig(domain="dc_motor", tier="rpi5", input_window=32, num_classes=3,
                              blocks=(ConvBlockConfig(filters=4, kernel_size=3),), dense_units=(), dropout=0.0)
    model = build_classifier(config, n_channels=2)
    model(tf.zeros((1, 32, 2)))
    tfl = tmp_path / "model.tflite"
    export_float16(model, tfl)
    x = np.random.default_rng(0).normal(size=(12, 32, 2)).astype(np.float32)
    keras_out = model.predict(x, verbose=0)
    vec = tmp_path / "v.npz"
    np.savez(vec, inputs=x, keras_out=keras_out, host_tflite_out=keras_out, meta=json.dumps({"run": "tiny", "block": "classifier"}))

    res = _load("rpi5_benchmark").run(str(tfl), str(vec), warmup=2, repeats=1)
    assert res["n_windows"] == 12 and res["latency_ms_p50"] > 0 and res["latency_ms_p99"] >= res["latency_ms_p50"]
    assert res["top1_agreement_vs_keras"] >= 0.9
    if platform.machine() not in ("aarch64", "armv7l"):
        assert res["ran_on_arm_board_with_native_interpreter"] is False


def test_readme_states_it_was_not_run_on_hardware():
    text = (KIT / "README.md").read_text()
    assert "Not executed on hardware" in text
