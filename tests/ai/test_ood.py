"""OOD detector (ai/ood.py) and its place in the decision (ai/calibration.decide) -- TODO A1."""

import numpy as np
import pytest

from driveflow.ai.calibration import UNCERTAIN, UNKNOWN, decide
from driveflow.ai.ood import energy_score, fit_mahalanobis, fit_ood, head_logits, mahalanobis_score, scores, window_score
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig

pytestmark = pytest.mark.ai

CLASSES = ["inner_race", "normal", "outer_race"]


def test_energy_is_minus_logsumexp():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]])
    expected = -np.log(np.exp(z[0]).sum())
    assert energy_score(z)[0] == pytest.approx(expected)
    assert np.isfinite(energy_score(z)[1]) and energy_score(z)[1] == pytest.approx(-1000.0)


def test_mahalanobis_separates_far_points_from_training_clusters():
    rng = np.random.default_rng(0)
    means = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    y = np.repeat(np.arange(3), 300)
    f = means[y] + rng.normal(size=(900, 3))
    params = fit_mahalanobis(f, y, 3)
    s_in = mahalanobis_score(means[y] + rng.normal(size=(900, 3)), params)
    s_far = mahalanobis_score(np.array([[0.0, 0.0, 12.0], [-10.0, -10.0, 0.0]]), params)
    assert np.all(s_far > np.quantile(s_in, 0.999))
    # Three dimensions: in-distribution squared distances follow ~chi2(3), mean ~3.
    assert 2.0 < np.mean(s_in) < 4.0


def _tiny_model():
    cfg = ClassifierConfig(domain="dc_motor", tier="pc", input_window=32, num_classes=3,
                           blocks=(ConvBlockConfig(filters=4, kernel_size=3),), dense_units=(8,), dropout=0.0)
    return build_classifier(cfg, n_channels=2)


def test_head_logits_reproduce_the_softmax():
    model = _tiny_model()
    x = np.random.default_rng(1).normal(size=(5, 32, 2)).astype(np.float32)
    from driveflow.ai.ood import feature_extractor

    feats = np.asarray(feature_extractor(model)(x))
    logits = head_logits(model, feats)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(p / p.sum(axis=1, keepdims=True), model.predict(x, verbose=0), rtol=1e-4, atol=1e-6)


def test_fit_ood_thresholds_keep_the_target_share_of_validation_windows():
    model = _tiny_model()
    rng = np.random.default_rng(2)
    x_train, y_train = rng.normal(size=(120, 32, 2)).astype(np.float32), np.repeat(np.arange(3), 40)
    x_val = rng.normal(size=(200, 32, 2)).astype(np.float32)
    ood = fit_ood(model, x_train, y_train, x_val, target_coverage=0.95)
    assert ood["fitted_on"] == {"statistics": "train", "thresholds": "val"}
    s = scores(model, x_val, ood)
    for m in ("mahalanobis", "energy"):
        assert np.mean(s[m] <= ood[m]["threshold"]) == pytest.approx(0.95, abs=0.01)
    assert window_score(model, x_val[0], ood) == pytest.approx(s["mahalanobis"][0], rel=1e-4)
    with pytest.raises(ValueError):
        fit_ood(model, x_train, y_train, x_val, method="softmax")


class TestDecide:
    METRICS = {"calibration": {"temperature": 1.0, "abstain_below": 0.6}, "ood": {"method": "mahalanobis", "mahalanobis": {"threshold": 10.0}}}

    def test_flagged_window_is_unknown_even_when_confident(self):
        label, _p = decide(np.array([0.01, 0.01, 0.98]), CLASSES, self.METRICS, ood_score=50.0)
        assert label == UNKNOWN

    def test_familiar_window_keeps_the_class_or_abstains(self):
        assert decide(np.array([0.01, 0.01, 0.98]), CLASSES, self.METRICS, ood_score=1.0)[0] == "outer_race"
        assert decide(np.array([0.4, 0.3, 0.3]), CLASSES, self.METRICS, ood_score=1.0)[0] == UNCERTAIN

    def test_runs_without_an_ood_block_are_unchanged(self):
        metrics = {"calibration": self.METRICS["calibration"]}
        assert decide(np.array([0.01, 0.01, 0.98]), CLASSES, metrics, ood_score=1e9)[0] == "outer_race"
        assert decide(np.array([0.01, 0.01, 0.98]), CLASSES, metrics)[0] == "outer_race"
