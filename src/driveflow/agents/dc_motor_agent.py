"""DC motor (domain dc_motor) simulation-based anomaly detection agent.

Phase 1 scaffolding only.
"""

from __future__ import annotations

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent

#: Same channel vocabulary as models.common.windowing.CANDIDATE_CHANNELS' physical subset --
#: current, speed, and 3-axis vibration.
_DEFAULT_FEATURES = ["i_a", "rpm", "acc_x", "acc_y", "acc_z"]


class DCMotorAnomalyDetector(SimulationBasedAgent):
    """Detecta fallas de rodamiento (outer_race, inner_race, ball, cage) comparando telemetría
    real contra simulaciones de Scenario(plant_config_id="dc_perm_ex_v1", ...) a distinta
    severidad."""

    def __init__(self, detector_metric: DetectorMetric = DetectorMetric.DTW):
        super().__init__(domain="dc_motor", detector_metric=detector_metric, monitored_features=list(_DEFAULT_FEATURES))

    def simulate_scenario(self, scenario_config: dict) -> dict:
        raise NotImplementedError("Phase 4")

    def detect_anomaly(self, telemetry_window: dict) -> tuple[float, str]:
        raise NotImplementedError("Phase 4")
