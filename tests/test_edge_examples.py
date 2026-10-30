"""The inference examples of the Edge Deployment page: the shipped normalization block equals the
training rule, the Python example equals Keras, the C++ examples parse, and the C2000 example
links against a real kit when TI's compiler is installed."""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers
from streamlit.testing.v1 import AppTest

from driveflow.ai.c2000_export import C2000MemoryError, export_c2000
from driveflow.ai.edge_examples import ModelFacts, c2000_example, esp32_example, normalization_block, rpi_cpp_example, rpi_python_example
from driveflow.models.common.windowing import normalize_window_like_training

CHANNELS = ["a", "rpm", "dead"]
CLASSIFIER = ModelFacts(channels=CHANNELS, window=32, classes=["x", "y"], level_scaling={"rpm": [1432.4, 477.5]}, calibration={"temperature": 0.9, "abstain_below": 0.7}, tflite_ops=["CONV_2D", "MEAN", "SOFTMAX"])
FORECASTER = ModelFacts(channels=CHANNELS, window=32, horizon=4, tflite_ops=["CONV_2D", "RESHAPE"])
CGT = Path.home() / "ti/ti-cgt-c2000_25.11.1.LTS/bin/cl2000"
TFLITE_STUBS = {
    "esp_rom_sys.h": "void esp_rom_delay_us(unsigned);",
    "freertos/FreeRTOS.h": "",
    "freertos/task.h": "void vTaskDelay(int);",
    "tensorflow/lite/micro/micro_interpreter.h": """#include <stddef.h>
#include <stdint.h>
enum { kTfLiteOk };
struct Params { float scale; int zero_point; };
union Data { int8_t *int8; };
struct TfLiteTensor { Params params; Data data; };
namespace tflite { struct Model {}; const Model *GetModel(const void *);
struct MicroInterpreter { template <class R> MicroInterpreter(const Model *, R &, uint8_t *, int); int AllocateTensors(); int Invoke(); size_t arena_used_bytes(); TfLiteTensor *input(int); TfLiteTensor *output(int); }; }""",
    "tensorflow/lite/micro/micro_mutable_op_resolver.h": "namespace tflite { template <int N> struct MicroMutableOpResolver { void AddConv2D(); void AddMean(); void AddSoftmax(); void AddReshape(); }; }",
    "tensorflow/lite/schema/schema_generated.h": "",
    "tensorflow/lite/interpreter.h": """#include <memory>
enum { kTfLiteOk };
namespace tflite { struct Interpreter { int AllocateTensors(); int Invoke(); template <class T> T *typed_input_tensor(int); template <class T> const T *typed_output_tensor(int); }; }""",
    "tensorflow/lite/kernels/register.h": "namespace tflite { namespace ops { namespace builtin { struct BuiltinOpResolver {}; } } }",
    "tensorflow/lite/model.h": """#include <memory>
namespace tflite { struct FlatBufferModel { static std::unique_ptr<FlatBufferModel> BuildFromFile(const char *); };
struct InterpreterBuilder { template <class R> InterpreterBuilder(FlatBufferModel &, R &); void operator()(std::unique_ptr<Interpreter> *); }; }""",
}


def _raw():
    rng = np.random.default_rng(3)
    raw = np.stack([rng.normal(2.0, 0.5, 32), rng.normal(1500.0, 40.0, 32), np.full(32, 3.0)], axis=1).astype(np.float32)
    return raw


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_normalization_block_equals_the_training_rule(tmp_path):
    raw = _raw()
    (tmp_path / "t.c").write_text("#include <math.h>\n#include <stdio.h>\n" + normalization_block(CLASSIFIER) + """
int main(void) {
  static float raw[WINDOW_SAMPLES * NUM_CHANNELS], out[WINDOW_SAMPLES * NUM_CHANNELS]; float mu[NUM_CHANNELS], sigma[NUM_CHANNELS];
  for (int i = 0; i < WINDOW_SAMPLES * NUM_CHANNELS; i++) scanf("%f", &raw[i]);
  normalize_window(raw, out, mu, sigma);
  for (int i = 0; i < WINDOW_SAMPLES * NUM_CHANNELS; i++) printf("%.9e\\n", out[i]);
  return 0;
}
""")
    subprocess.run(["gcc", "-std=c99", "-O1", "t.c", "-lm", "-o", "t"], cwd=tmp_path, check=True)
    text = "\n".join(f"{v:.9e}" for v in raw.ravel())
    out = subprocess.run([str(tmp_path / "t")], input=text, check=True, capture_output=True, text=True).stdout
    expected, _, _ = normalize_window_like_training(raw, CHANNELS, CLASSIFIER.level_scaling)
    assert np.allclose(np.array(out.split(), dtype=np.float32).reshape(raw.shape), expected, atol=1e-5)


@pytest.mark.parametrize("facts", [CLASSIFIER, FORECASTER], ids=["classifier", "forecaster"])
def test_python_example_matches_keras(tmp_path, monkeypatch, facts):
    inputs = keras.Input((32, 3))
    x = layers.Conv1D(3, 3, padding="same", activation="relu")(inputs)
    if facts.is_classifier:
        output = layers.Dense(2, activation="softmax")(layers.GlobalAveragePooling1D()(x))
    else:
        output = layers.Reshape((4, 3))(layers.Dense(12)(layers.GlobalAveragePooling1D()(x)))
    model = keras.Model(inputs, output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    (tmp_path / "model.tflite").write_bytes(converter.convert())

    package, module = types.ModuleType("ai_edge_litert"), types.ModuleType("ai_edge_litert.interpreter")
    module.Interpreter = tf.lite.Interpreter
    monkeypatch.setitem(sys.modules, "ai_edge_litert", package)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", module)
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "example"}
    exec(rpi_python_example(facts), namespace)  # noqa: S102 -- the generated example, the subject of this test

    raw = _raw()
    window, mu, sigma = normalize_window_like_training(raw, CHANNELS, facts.level_scaling)
    keras_out = model.predict(window[np.newaxis], verbose=0)[0]
    result = namespace["infer"](raw)
    if facts.is_classifier:
        calibrated = keras_out ** (1 / facts.temperature)
        assert np.allclose(result[1], calibrated / calibrated.sum(), atol=1e-4)
    else:
        assert np.allclose(result, keras_out * sigma + mu, atol=1e-3)


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
@pytest.mark.parametrize("generator", [esp32_example, rpi_cpp_example], ids=["esp32", "rpi5_cpp"])
@pytest.mark.parametrize("facts", [CLASSIFIER, FORECASTER], ids=["classifier", "forecaster"])
def test_cpp_examples_parse(tmp_path, generator, facts):
    """Against stub headers: it catches broken C++ in the template, not a wrong TFLite API."""
    for name, text in TFLITE_STUBS.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("#pragma once\n" + text)
    (tmp_path / "main.cc").write_text(generator(facts))
    subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "-I.", "main.cc"], cwd=tmp_path, check=True)


def test_esp32_example_registers_the_ops_of_the_model():
    code = esp32_example(CLASSIFIER)
    assert "MicroMutableOpResolver<3>" in code and "AddConv2D" in code and "AddSoftmax" in code


@pytest.mark.skipif(not CGT.exists(), reason="TI CGT not installed")
@pytest.mark.parametrize("facts", [CLASSIFIER, FORECASTER], ids=["classifier", "forecaster"])
def test_c2000_example_links_against_a_kit(tmp_path, facts):
    inputs = keras.Input((32, 3))
    x = layers.GlobalAveragePooling1D()(layers.Conv1D(3, 3, padding="same")(inputs))
    output = layers.Dense(2, activation="softmax")(x) if facts.is_classifier else layers.Reshape((4, 3))(layers.Dense(12)(x))
    metrics = {"channels": CHANNELS, "classes": facts.classes, "level_scaling": facts.level_scaling, "calibration": facts.calibration}
    export_c2000(keras.Model(inputs, output), metrics, _raw()[np.newaxis], tmp_path / "kit")
    (tmp_path / "kit" / "main.c").write_text(c2000_example(facts))
    result = subprocess.run(["make", "-C", str(tmp_path / "kit")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


class TestPage:
    """The export button. export_run is replaced: a real export would write into configs/."""

    @staticmethod
    def _open():
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py"), default_timeout=120)
        at.run()
        at.switch_page("pages/edge_deployment.py").run()
        return at

    def test_every_board_shows_code_and_tutorial(self):
        at = self._open()
        assert not at.exception
        assert {"ESP32", "Raspberry Pi 5", "TI C2000"} <= {t.label.rsplit(": ", 1)[-1] for t in at.tabs}
        languages = {c.language for c in at.code}
        assert {"cpp", "python", "c", "bash"} <= languages

    def test_a_measured_kit_needs_confirmation_before_it_is_replaced(self):
        at = self._open()
        button = next(b for b in at.button if b.key.startswith("edge_c2000_export_"))
        assert button.disabled
        next(c for c in at.checkbox if c.key.startswith("edge_c2000_replace_")).check().run()
        assert not next(b for b in at.button if b.key.startswith("edge_c2000_export_")).disabled

    def test_a_refused_model_shows_the_reason(self, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise C2000MemoryError("does not fit LAUNCHXL-F28379D")

        monkeypatch.setattr("driveflow.viz.views.edge_deployment.export_run", refuse)
        at = self._open()
        next(c for c in at.checkbox if c.key.startswith("edge_c2000_replace_")).check().run()
        next(b for b in at.button if b.key.startswith("edge_c2000_export_")).click().run()
        assert not at.exception
        assert any("does not fit" in e.value for e in at.error)
