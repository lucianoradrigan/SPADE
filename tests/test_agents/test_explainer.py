"""Tests for AnomalyExplainer (Phase 4)."""

import numpy as np

from driveflow.agents.explainer import AnomalyExplainer


class TestFeatureImportance:
    def test_weights_sum_to_one(self):
        explainer = AnomalyExplainer(anomaly_score=0.8, diagnosis="fault")
        telemetry = {"a": np.zeros(10), "b": np.full(10, 5.0)}
        simulation = {"a": np.full(10, 1.0), "b": np.zeros(10)}
        importance = explainer.feature_importance(telemetry, simulation)
        assert set(importance) == {"a", "b"}
        assert sum(importance.values()) == 1.0

    def test_the_feature_with_the_bigger_gap_gets_more_weight(self):
        explainer = AnomalyExplainer(anomaly_score=0.8, diagnosis="fault")
        telemetry = {"barely_off": np.full(10, 0.1), "way_off": np.full(10, 10.0)}
        simulation = {"barely_off": np.zeros(10), "way_off": np.zeros(10)}
        importance = explainer.feature_importance(telemetry, simulation)
        assert importance["way_off"] > importance["barely_off"]

    def test_only_shared_features_are_considered(self):
        explainer = AnomalyExplainer(anomaly_score=0.5, diagnosis="fault")
        telemetry = {"a": np.zeros(5), "only_in_telemetry": np.zeros(5)}
        simulation = {"a": np.ones(5), "only_in_simulation": np.ones(5)}
        importance = explainer.feature_importance(telemetry, simulation)
        assert set(importance) == {"a"}

    def test_no_shared_features_returns_empty(self):
        explainer = AnomalyExplainer(anomaly_score=0.5, diagnosis="fault")
        assert explainer.feature_importance({"a": np.zeros(5)}, {"b": np.zeros(5)}) == {}

    def test_tolerates_mismatched_lengths_via_dtw_fallback(self):
        # zeros vs. zeros (even at different lengths) would have zero distance regardless of
        # which metric is used -- an earlier version of this test used exactly that and asserted
        # importance["a"] == 1.0, which can never be true when the (correctly, in this case)
        # computed distance is 0. Use values that actually differ, so there's a real non-zero
        # distance for the DTW fallback (euclidean_distance would raise ValueError here, since
        # 5 != 8) to compute.
        explainer = AnomalyExplainer(anomaly_score=0.5, diagnosis="fault")
        importance = explainer.feature_importance({"a": np.zeros(5)}, {"a": np.full(8, 3.0)})
        assert importance["a"] == 1.0


class TestExplain:
    def test_includes_score_diagnosis_and_top_feature(self):
        explainer = AnomalyExplainer(anomaly_score=0.9, diagnosis="moderate")
        telemetry = {"way_off": np.full(10, 10.0), "barely_off": np.full(10, 0.1)}
        simulation = {"way_off": np.zeros(10), "barely_off": np.zeros(10)}
        result = explainer.explain(telemetry, simulation)
        assert result["anomaly_score"] == 0.9
        assert result["diagnosis"] == "moderate"
        assert result["most_responsible_feature"] == "way_off"
        assert "way_off" in result["summary"]

    def test_handles_no_shared_features_gracefully(self):
        explainer = AnomalyExplainer(anomaly_score=0.1, diagnosis="healthy")
        result = explainer.explain({"a": np.zeros(5)}, {"b": np.zeros(5)})
        assert result["most_responsible_feature"] is None
        assert result["feature_importance"] == {}
