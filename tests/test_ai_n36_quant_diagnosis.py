"""AI-N36 diagnosis (2026-11-12): the quantization emulation used in docs/audit/b4/ai_n36_quant_diagnosis.md is exact in
float (BatchNorm folding) and its int8 helpers behave as TFLite's per-tensor scheme."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
RUN = REPO / "configs" / "classifiers" / "esp32_dscnn" / "2026-11-10_run01"
pytestmark = pytest.mark.skipif(not (RUN / "model.weights.h5").exists(), reason="B4 4a run not present")


def test_folded_float_forward_equals_keras():
    import ai_n36_quant_diagnosis as D
    import report_esp32_repair as ER

    model, _ = ER._keras(str(RUN.relative_to(REPO)))
    X = np.random.default_rng(0).normal(size=(8, 512, 6)).astype(np.float32)
    z, _ = D.forward(D._fold(model), X)
    p = np.exp(z - z.max(1, keepdims=True))
    np.testing.assert_allclose(p / p.sum(1, keepdims=True), model.predict(X, verbose=0), atol=1e-4)


def test_per_tensor_weight_rounding_and_fake_quant():
    import ai_n36_quant_diagnosis as D

    w = np.array([-1.0, -0.5, 0.0, 0.3, 1.27])
    q = D._qw(w)
    assert np.max(np.abs(q - w)) <= np.max(np.abs(w)) / 127 / 2 + 1e-9 and np.isclose(q[-1], 1.27)
    x = np.linspace(-2, 2, 101).astype(np.float32)
    y = D._fq(x, (-1.0, 1.0))
    assert y.min() >= -1.0 - 1e-6 and y.max() <= 1.0 + 1e-6 and len(np.unique(y)) <= 256
