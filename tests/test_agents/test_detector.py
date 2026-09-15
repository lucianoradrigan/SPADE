"""Tests for AnomalyDetector (Phase 4): real DTW/Euclidean/Mahalanobis, checked against known
correctness properties -- not just "doesn't raise"."""

import numpy as np
import pytest

from driveflow.agents.detector import AnomalyDetector


class TestDtwDistance:
    def test_identical_sequences_have_zero_distance(self):
        seq = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert AnomalyDetector.dtw_distance(seq, seq) == pytest.approx(0.0, abs=1e-9)

    def test_tolerates_different_lengths(self):
        # Not a palindrome (unlike an earlier version of this test, which used
        # [0,1,2,3,2,1,0] -- reversing that gives the SAME sequence, so both distances below
        # came out trivially equal and the test asserted nothing real).
        base = [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 1.0]
        # A "stretched" version of the same shape (each point repeated) -- DTW should align them
        # with near-zero cost, unlike a plain point-by-point (Euclidean) comparison which can't
        # even be computed for mismatched lengths.
        stretched = [v for v in base for _ in range(2)]
        assert AnomalyDetector.dtw_distance(base, stretched) < AnomalyDetector.dtw_distance(base, list(reversed(stretched)))

    def test_more_different_sequences_have_larger_distance(self):
        base = [0.0] * 10
        close = [0.1] * 10
        far = [5.0] * 10
        assert AnomalyDetector.dtw_distance(base, close) < AnomalyDetector.dtw_distance(base, far)

    def test_supports_multi_feature_2d_sequences(self):
        a = np.zeros((5, 2))
        b = np.ones((5, 2))
        d = AnomalyDetector.dtw_distance(a, b)
        assert d > 0

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            AnomalyDetector.dtw_distance([], [1.0, 2.0])


class TestEuclideanDistance:
    def test_identical_sequences_have_zero_distance(self):
        seq = [1.0, 2.0, 3.0]
        assert AnomalyDetector.euclidean_distance(seq, seq) == pytest.approx(0.0)

    def test_matches_manual_l2_norm(self):
        a = [0.0, 0.0, 0.0]
        b = [3.0, 4.0, 0.0]
        assert AnomalyDetector.euclidean_distance(a, b) == pytest.approx(5.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            AnomalyDetector.euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])


class TestMahalanobisDistance:
    def test_reduces_to_euclidean_with_identity_covariance(self):
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        identity = np.eye(2)
        assert AnomalyDetector.mahalanobis_distance(a, b, identity) == pytest.approx(AnomalyDetector.euclidean_distance(a, b))

    def test_matches_scipy_reference_for_a_known_covariance(self):
        from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

        a = [1.0, 2.0]
        b = [4.0, 1.0]
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        expected = scipy_mahalanobis(a, b, np.linalg.inv(cov))
        assert AnomalyDetector.mahalanobis_distance(a, b, cov) == pytest.approx(expected)

    def test_wrong_covariance_shape_raises(self):
        with pytest.raises(ValueError):
            AnomalyDetector.mahalanobis_distance([1.0, 2.0], [3.0, 4.0], np.eye(3))
