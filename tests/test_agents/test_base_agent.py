"""Tests for SimulationBasedAgent (Phase 4): detect_anomaly's real shared scoring algorithm,
exercised against a small fake concrete subclass (fast, deterministic -- no real physics
simulation) plus the real DCMotorAnomalyDetector/VSCDPCAnomalyDetector for at least one true
end-to-end check each."""

import numpy as np
import pytest

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent
from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector
from driveflow.datagen.runner import TAU


class _FakeAgent(SimulationBasedAgent):
    """A constant-signal fake domain: simulate_scenario returns `n` copies of `level`, so
    detect_anomaly's own scoring logic can be tested precisely without running real physics."""

    HYPOTHESES = {"healthy": {"level": 0.0}, "fault": {"level": 10.0}}

    def __init__(self, detector_metric=DetectorMetric.EUCLIDEAN):
        super().__init__(domain="fake", detector_metric=detector_metric, monitored_features=["signal"])
        self.simulate_calls = 0

    def simulate_scenario(self, scenario_config: dict) -> dict:
        self.simulate_calls += 1
        n = round(scenario_config["duration_s"] / TAU)
        return {"signal": np.full(n, scenario_config["level"])}


def test_cannot_instantiate_the_abstract_base_class():
    with pytest.raises(TypeError):
        SimulationBasedAgent(domain="dc_motor")


def test_detector_metric_enum():
    assert DetectorMetric.DTW.value == "dtw"
    assert DetectorMetric.EUCLIDEAN.value == "euclidean"
    assert DetectorMetric.MAHALANOBIS.value == "mahalanobis"


class TestDetectAnomalyScoring:
    def test_telemetry_matching_healthy_scores_low(self):
        agent = _FakeAgent()
        telemetry = {"signal": np.zeros(20)}
        score, closest = agent.detect_anomaly(telemetry)
        assert closest == "healthy"
        assert score < 0.5

    def test_telemetry_matching_fault_scores_high(self):
        agent = _FakeAgent()
        telemetry = {"signal": np.full(20, 10.0)}
        score, closest = agent.detect_anomaly(telemetry)
        assert closest == "fault"
        assert score > 0.5

    def test_score_is_always_in_0_1(self):
        agent = _FakeAgent()
        for level in (-5.0, 0.0, 3.0, 10.0, 25.0):
            score, _ = agent.detect_anomaly({"signal": np.full(20, level)})
            assert 0.0 <= score <= 1.0

    def test_simulations_are_cached_per_hypothesis_and_duration(self):
        agent = _FakeAgent()
        agent.detect_anomaly({"signal": np.zeros(20)})
        agent.detect_anomaly({"signal": np.zeros(20)})  # same duration -- both hypotheses cached
        assert agent.simulate_calls == 2  # one per HYPOTHESES entry, not per call

        agent.detect_anomaly({"signal": np.zeros(40)})  # different duration -- cache miss again
        assert agent.simulate_calls == 4

    def test_missing_monitored_feature_raises(self):
        agent = _FakeAgent()
        with pytest.raises(ValueError, match="none of the monitored features"):
            agent.detect_anomaly({"unrelated": np.zeros(20)})

    def test_hypotheses_without_healthy_entry_raises(self):
        class _NoHealthy(_FakeAgent):
            HYPOTHESES = {"fault": {"level": 10.0}}

        with pytest.raises(ValueError, match="'healthy'"):
            _NoHealthy().detect_anomaly({"signal": np.zeros(20)})

    def test_clear_cache_forces_resimulation(self):
        agent = _FakeAgent()
        agent.detect_anomaly({"signal": np.zeros(20)})
        assert agent.simulate_calls == 2
        agent.clear_cache()
        agent.detect_anomaly({"signal": np.zeros(20)})
        assert agent.simulate_calls == 4


class TestComputeDistance:
    def test_mahalanobis_is_rejected_without_a_covariance_matrix(self):
        agent = _FakeAgent(detector_metric=DetectorMetric.MAHALANOBIS)
        with pytest.raises(ValueError, match="covariance matrix"):
            agent.compute_distance([1.0, 2.0], [3.0, 4.0])


class TestRealAgentsEndToEnd:
    """Not fakes -- the actual DCMotorAnomalyDetector/VSCDPCAnomalyDetector, run against real
    (if tiny) driveflow.datagen simulations both for the "telemetry" and the hypotheses."""

    def test_dc_motor_agent_flags_a_real_faulty_run_as_closer_to_a_fault_hypothesis(self):
        from driveflow.datagen import Scenario, run_scenario

        agent = DCMotorAnomalyDetector()
        duration_s = 20 * TAU
        records = run_scenario(Scenario(scenario_id="test_telemetry", fault_type="outer_race", electrical_severity=8.0, mechanical_severity=0.05, duration_s=duration_s, seed=99))
        import pandas as pd

        df = pd.DataFrame.from_records(records)
        telemetry = {feat: df[feat].to_numpy(dtype=float) for feat in agent.monitored_features if feat in df.columns}

        score, closest = agent.detect_anomaly(telemetry)
        assert closest in agent.HYPOTHESES
        assert 0.0 <= score <= 1.0

    def test_vsc_dpc_agent_runs_end_to_end(self):
        from driveflow.datagen import Scenario, run_scenario

        agent = VSCDPCAnomalyDetector()
        duration_s = 20 * TAU
        records = run_scenario(Scenario(scenario_id="test_telemetry", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=duration_s, seed=0))
        import pandas as pd

        df = pd.DataFrame.from_records(records)
        telemetry = {feat: df[feat].to_numpy(dtype=float) for feat in agent.monitored_features if feat in df.columns}

        score, closest = agent.detect_anomaly(telemetry)
        assert closest in agent.HYPOTHESES
        assert 0.0 <= score <= 1.0
        # Telemetry generated at the network's own trained load_resistance_ohm should read as
        # close to "healthy" -- the whole point of that hypothesis being the training value.
        assert closest == "healthy"
