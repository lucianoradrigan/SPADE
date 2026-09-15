"""Phase 1 scaffolding smoke tests for SimulationBasedAgent and its concrete subclasses."""

import pytest

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent
from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        SimulationBasedAgent(domain="dc_motor")


def test_detector_metric_enum():
    assert DetectorMetric.DTW.value == "dtw"
    assert DetectorMetric.EUCLIDEAN.value == "euclidean"
    assert DetectorMetric.MAHALANOBIS.value == "mahalanobis"


def test_dc_motor_agent_has_the_real_channel_names():
    agent = DCMotorAnomalyDetector()
    assert agent.domain == "dc_motor"
    assert agent.monitored_features == ["i_a", "rpm", "acc_x", "acc_y", "acc_z"]


def test_vsc_agent_has_the_real_channel_names_not_dq_frame():
    """Regression guard: an earlier draft of this module used v_d/v_q/i_d/i_q (PMSM FOC naming,
    an unrelated system) instead of vsc_dpc's actual channels."""
    agent = VSCDPCAnomalyDetector()
    assert agent.domain == "vsc_dpc"
    assert agent.monitored_features == ["vc_real", "vc_imag", "i_f_real", "i_f_imag"]
