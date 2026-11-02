"""One place to open a .tflite file for host-side checks (parity, quantization, deployment status).

``tf.lite.Interpreter`` is deprecated in favour of the ``ai-edge-litert`` package (LiteRT), a
SPADE dependency everywhere except Intel Macs, where no wheel exists for Python 3.12. This module
uses LiteRT when it is installed and TensorFlow's interpreter otherwise.

Neither runs a model with Flex ops (the GRU forecaster ``rpi5_edge``): both build it and read its
tensor details, and both raise RuntimeError at ``allocate_tensors``, because neither Python package
links the Flex delegate. Callers already treat that error as "not runnable on a builtin
interpreter" (experiments/report_tflite_parity.py).

Every SPADE call site goes through ``load_interpreter``, so dropping the fallback once TensorFlow
removes ``tf.lite.Interpreter`` is a change in this file only.
"""

from __future__ import annotations


def load_interpreter(model_path) -> object:
    """An Interpreter for model_path, tensors not yet allocated. ``runtime`` says which one it is."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.runtime = f"tensorflow {tf.__version__}"
        return interp
    interp = Interpreter(model_path=str(model_path))
    interp.runtime = "ai_edge_litert"
    return interp
