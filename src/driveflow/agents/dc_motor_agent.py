"""DC motor (domain dc_motor) simulation-based anomaly detection agent.

Phase 4: HYPOTHESES uses concrete electrical_severity (Nm)/mechanical_severity (Module B's own
units) pairs -- the same ranges the dashboard's own sidebar sliders expose (0-20 Nm, 0-0.2) --
not a generic 0-1 "fault_severity" scale, because dc_motor's own Scenario has no such field (see
datagen/scenario.py: severity is two independent physical parameters, not one normalized knob).
"""

from __future__ import annotations

import pandas as pd

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent
from driveflow.datagen import Scenario, run_scenario

#: Same channel vocabulary as models.common.windowing.CANDIDATE_CHANNELS' physical subset --
#: current, speed, and 3-axis vibration.
_DEFAULT_FEATURES = ["i_a", "rpm", "acc_x", "acc_y", "acc_z"]

_HYPOTHESES = {
    "healthy": {"fault_type": None},
    "incipient": {"fault_type": "outer_race", "electrical_severity": 3.0, "mechanical_severity": 0.02},
    "moderate": {"fault_type": "outer_race", "electrical_severity": 8.0, "mechanical_severity": 0.05},
}


class DCMotorAnomalyDetector(SimulationBasedAgent):
    """Detecta fallas de rodamiento (outer_race, inner_race, ball, cage) comparando telemetría
    real contra simulaciones de Scenario(plant_config_id="dc_perm_ex_v1", ...) sanas vs. con
    falla a distinta severidad."""

    HYPOTHESES = _HYPOTHESES

    def __init__(self, detector_metric: DetectorMetric = DetectorMetric.DTW):
        super().__init__(domain="dc_motor", detector_metric=detector_metric, monitored_features=list(_DEFAULT_FEATURES))

    def simulate_scenario(self, scenario_config: dict) -> dict:
        scenario = Scenario(scenario_id="agent_hypothesis", **scenario_config)
        records = run_scenario(scenario)
        df = pd.DataFrame.from_records(records)
        return {feat: df[feat].to_numpy(dtype=float) for feat in self.monitored_features if feat in df.columns}
