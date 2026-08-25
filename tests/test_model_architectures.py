"""Shape/sanity tests for the three Fase C architectures ported from paper_federative
(repo/02_local_baselines_etapa1/train_baseline.py, repo/03_forecast_regressors_etapa2/
train_forecast_envelope_combined.py). Not a numerical-fidelity check against the originals (no
pretrained weights to compare -- these are trained fresh on driveflow's synthetic data, see
docs/macro_fase_C1_diagnosis.md) -- this just confirms the ported architectures build, run, and
produce the right shapes/parameter-sharing structure.
"""

import numpy as np
import tensorflow as tf

from driveflow.models.classifiers import build_gateway, build_sensor
from driveflow.models.regressors import build_envelope_forecaster


class TestSensor:
    def test_output_shape_is_softmax_over_classes(self):
        model = build_sensor(n_classes=3, n_channels=4, window_samples=5000)
        out = model(tf.zeros((2, 5000, 4)))
        assert out.shape == (2, 3)
        assert np.allclose(np.sum(out.numpy(), axis=1), 1.0, atol=1e-5)

    def test_small_param_count(self):
        """paper_federative reports ~4-8K params for this architecture (ESP32 target)."""
        model = build_sensor(n_classes=4, n_channels=6, window_samples=5000)
        assert model.count_params() < 20_000


class TestGateway:
    def test_dual_output_shapes(self):
        model = build_gateway(n_classes=3, n_channels=4, window_samples=5000)
        out = model(tf.zeros((2, 5000, 4)))
        assert out["classification"].shape == (2, 3)
        assert out["projection"].shape == (2, 32)  # BACKBONE_DIM, matches sensor's backbone_output
        assert np.allclose(np.sum(out["classification"].numpy(), axis=1), 1.0, atol=1e-5)

    def test_much_larger_than_sensor(self):
        """paper_federative reports ~200K params (RPi target) vs. sensor's ~4-8K."""
        model = build_gateway(n_classes=4, n_channels=6, window_samples=5000)
        assert model.count_params() > 50_000


class TestEnvelopeForecaster:
    def test_output_shape_is_bins_by_channels(self):
        model = build_envelope_forecaster(n_channels=4, n_bins=10, window_samples=5000)
        out = model(tf.zeros((2, 5000, 4)))
        assert out.shape == (2, 10, 4)

    def test_output_is_nonnegative(self):
        """softplus head -- envelope/RMS magnitude can't be negative."""
        model = build_envelope_forecaster(n_channels=3, n_bins=5, window_samples=5000)
        out = model(tf.random.normal((4, 5000, 3)))
        assert np.all(out.numpy() >= 0)

    def test_shares_backbone_shape_with_sensor(self):
        """Both use the same DS-CNN backbone -- same param count up to the head."""
        sensor = build_sensor(n_classes=3, n_channels=4, window_samples=5000)
        forecaster = build_envelope_forecaster(n_channels=4, n_bins=10, window_samples=5000)
        sensor_backbone_params = sum(np.prod(w.shape) for l in sensor.layers if l.name != "head" for w in l.get_weights())
        forecaster_backbone_params = sum(
            np.prod(w.shape) for l in forecaster.layers if l.name != "envelope_head" for w in l.get_weights()
        )
        assert sensor_backbone_params == forecaster_backbone_params
