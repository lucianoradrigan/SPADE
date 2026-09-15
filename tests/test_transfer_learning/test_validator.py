"""Tests for ModelValidator (Phase 2): each check exercised with concrete inputs, plus an
end-to-end validate() call against a real (tiny) trained classifier."""

import keras
import numpy as np
import pytest

from driveflow.ai.transfer.validator import ModelValidator
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig


def _tiny_classifier():
    # Weight init draws from Keras' own global RNG, not seeded by the local np.random.default_rng
    # used for data below -- found via a real flake: this test passed alone but failed inside the
    # full file (a different, less separable random init after other tests' model constructions
    # had already advanced Keras' global RNG state). Seeding it here makes init deterministic too.
    keras.utils.set_random_seed(0)
    config = ClassifierConfig(domain="dc_motor", tier="pc", input_window=16, num_classes=2, blocks=(ConvBlockConfig(filters=4, kernel_size=3),))
    model = build_classifier(config, n_channels=2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


class TestCheckAccuracyThreshold:
    def test_uses_min_accuracy_when_given(self):
        validator = ModelValidator(baseline_metrics={"accuracy": 0.5}, thresholds={"min_accuracy": 0.90})
        assert validator.check_accuracy_threshold(0.92) is True
        assert validator.check_accuracy_threshold(0.89) is False

    def test_falls_back_to_max_accuracy_drop_against_baseline(self):
        validator = ModelValidator(baseline_metrics={"accuracy": 0.895}, thresholds={"max_accuracy_drop": 0.05})
        assert validator.check_accuracy_threshold(0.86) is True
        assert validator.check_accuracy_threshold(0.80) is False

    def test_passes_when_no_threshold_or_baseline_given(self):
        validator = ModelValidator(baseline_metrics={}, thresholds={})
        assert validator.check_accuracy_threshold(0.1) is True


class TestCheckConvergence:
    def test_true_when_loss_decreased(self):
        validator = ModelValidator({}, {})
        assert validator.check_convergence([0.9, 0.6, 0.3, 0.24]) is True

    def test_false_when_loss_increased(self):
        validator = ModelValidator({}, {})
        assert validator.check_convergence([0.2, 0.3, 0.5]) is False

    def test_false_for_nan_or_inf(self):
        validator = ModelValidator({}, {})
        assert validator.check_convergence([0.5, float("nan"), 0.1]) is False
        assert validator.check_convergence([0.5, float("inf")]) is False

    def test_false_for_a_single_point(self):
        validator = ModelValidator({}, {})
        assert validator.check_convergence([0.5]) is False


class TestCheckModeCollapse:
    def test_true_when_more_than_one_class_predicted(self):
        validator = ModelValidator({}, {})
        assert validator.check_mode_collapse([0, 1, 0, 1, 1]) is True

    def test_false_when_every_prediction_is_the_same_class(self):
        validator = ModelValidator({}, {})
        assert validator.check_mode_collapse([1, 1, 1, 1]) is False


def _learnable_classification_data(n=40, seed=0):
    """Unlike pure noise, class is a deterministic function of the window's own mean -- something
    a 1-conv-block classifier can actually learn in a few epochs, so its predictions span both
    classes (mode collapse would be the CORRECT outcome on pure noise, not a validator bug)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    offsets = np.where(y == 1, 3.0, -3.0)
    X = rng.normal(size=(n, 16, 2)).astype("float32") + offsets[:, None, None]
    return X.astype("float32"), y


class TestValidateEndToEnd:
    def test_passes_for_a_healthy_classifier(self):
        model = _tiny_classifier()
        X, y = _learnable_classification_data(n=40, seed=0)
        model.fit(X, y, epochs=8, batch_size=8, verbose=0)

        validator = ModelValidator(baseline_metrics={"accuracy": 0.0}, thresholds={"min_accuracy": 0.0})
        passed, results = validator.validate(model, {"X": X, "y": y})
        assert passed is True
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_fails_when_accuracy_below_threshold(self):
        model = _tiny_classifier()  # untrained -- near-chance accuracy
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 16, 2)).astype("float32")
        y = np.array([0, 1] * 10)

        validator = ModelValidator(baseline_metrics={}, thresholds={"min_accuracy": 1.1})  # impossible to meet
        passed, results = validator.validate(model, {"X": X, "y": y})
        assert passed is False
        assert results["accuracy_ok"] is False
