"""Tests driveflow/ai/tflite_export.py (docs/design_ai_layer_transversal.md Sec. 4.1/4.2,
Sec. 8 step 8): float16 export for the rpi5 tier, int8 (calibrated against real windows) for
the esp32 tier -- both actually produce a loadable, correctly-typed TFLite model, not just bytes.
"""

import numpy as np
import pytest
import tensorflow as tf

from driveflow.ai.tflite_export import export_float16, export_int8
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig


@pytest.fixture
def tiny_model():
    config = ClassifierConfig(
        domain="dc_motor", tier="esp32", input_window=32, num_classes=3,
        blocks=(ConvBlockConfig(filters=4, kernel_size=3, block_type="dsconv1d"),),
        dense_units=(), dropout=0.0,
    )
    model = build_classifier(config, n_channels=2)
    model(tf.zeros((1, 32, 2)))  # build weights
    return model, config


class TestExportFloat16:
    def test_produces_a_loadable_tflite_model(self, tmp_path, tiny_model):
        model, config = tiny_model
        out_path = tmp_path / "model.tflite"
        export_float16(model, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        interpreter = tf.lite.Interpreter(model_path=str(out_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        assert input_details[0]["shape"].tolist() == [1, config.input_window, 2]

    def test_matches_keras_output_reasonably(self, tmp_path, tiny_model):
        """float16 is a light quantization -- the TFLite model's prediction should stay close to
        the original keras model's, not just be structurally loadable."""
        model, config = tiny_model
        out_path = tmp_path / "model.tflite"
        export_float16(model, out_path)

        x = np.random.randn(1, config.input_window, 2).astype(np.float32)
        keras_out = model(x, training=False).numpy()

        interpreter = tf.lite.Interpreter(model_path=str(out_path))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(out["index"])

        assert np.allclose(keras_out, tflite_out, atol=0.05)


class TestExportInt8:
    def test_produces_an_int8_quantized_model(self, tmp_path, tiny_model):
        model, config = tiny_model
        representative = np.random.randn(20, config.input_window, 2).astype(np.float32)
        out_path = tmp_path / "model.tflite"
        export_int8(model, representative, out_path)

        assert out_path.exists()
        interpreter = tf.lite.Interpreter(model_path=str(out_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        assert input_details[0]["dtype"] == np.int8
        assert output_details[0]["dtype"] == np.int8

    def test_is_smaller_than_float16(self, tmp_path, tiny_model):
        model, config = tiny_model
        representative = np.random.randn(20, config.input_window, 2).astype(np.float32)
        int8_path = tmp_path / "int8.tflite"
        fp16_path = tmp_path / "fp16.tflite"
        export_int8(model, representative, int8_path)
        export_float16(model, fp16_path)
        assert int8_path.stat().st_size <= fp16_path.stat().st_size

    def test_empty_representative_windows_raises(self, tmp_path, tiny_model):
        model, _ = tiny_model
        with pytest.raises(ValueError, match="empty"):
            export_int8(model, np.empty((0, 32, 2), dtype=np.float32), tmp_path / "model.tflite")
