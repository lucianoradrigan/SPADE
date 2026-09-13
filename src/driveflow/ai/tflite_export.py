"""TFLite export for a trained keras.Model (docs/design_ai_layer_transversal.md Sec. 4.1/4.2,
Sec. 8 step 8): float16 for the Raspberry Pi 5 tier ("TFLite float16/int8"), int8 -- calibrated
against real windows, not random noise -- for the ESP32 tier ("TFLite Micro int8").

Deliberately narrow: two functions, no auto-selection of which one to call for a given tier (the
caller, experiments/export_tflite.py, decides that from --tier) -- this module only knows how to
convert, not the design doc's tier->format mapping.

Uses tf.lite.Interpreter (in the int8 sanity path some callers may want, not used internally here)
-- that class is deprecated in favor of the separate `ai_edge_litert` package, not a project
dependency (avoided here to not add one); still functional as of TF 2.21, a known future
migration point, not an oversight.
"""

import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def _export_saved_model(model: keras.Model) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="driveflow_tflite_"))
    saved_path = tmp_dir / "saved_model"
    model.export(str(saved_path))
    return saved_path


def export_float16(model: keras.Model, out_path) -> bytes:
    """Raspberry Pi 5 tier (Sec. 4.1/4.2's 'TFLite float16/int8' row -- float16 half): halves
    weight storage with no representative-data calibration needed, unlike int8."""
    saved_path = _export_saved_model(model)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    return tflite_model


def export_int8(model: keras.Model, representative_windows: np.ndarray, out_path, n_calibration_samples: int = 100) -> bytes:
    """ESP32 tier (Sec. 4.1/4.2's 'TFLite Micro int8' row): full integer quantization, both
    weights and activations. representative_windows: (n, window_samples, n_channels) array of
    REAL input windows (the model's own training/evaluation data, already normalized the same way
    training was) -- calibrating against random noise would give activation ranges that don't
    reflect what the model actually sees, producing a technically-valid but badly-quantized model."""
    if len(representative_windows) == 0:
        raise ValueError("representative_windows is empty -- int8 calibration needs at least one real window")
    saved_path = _export_saved_model(model)
    sample = representative_windows[:n_calibration_samples]

    def _rep_dataset():
        for i in range(len(sample)):
            yield [sample[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    return tflite_model
