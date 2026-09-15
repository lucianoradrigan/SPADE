"""Phase 1 scaffolding smoke tests for AnomalyDetector."""

import pytest

from driveflow.agents.detector import AnomalyDetector


def test_detector_methods_exist():
    assert hasattr(AnomalyDetector, "dtw_distance")
    assert hasattr(AnomalyDetector, "euclidean_distance")
    assert hasattr(AnomalyDetector, "mahalanobis_distance")


def test_detector_missing_implementation():
    with pytest.raises(NotImplementedError):
        AnomalyDetector.dtw_distance([1, 2, 3], [1, 2, 3])
