"""DC motor (domain dc_motor) simulation-based anomaly detection agent.

Phase 4: HYPOTHESES uses concrete electrical_severity (Nm)/mechanical_severity (Module B's own
units) pairs -- the same ranges the dashboard's own sidebar sliders expose (0-20 Nm, 0-0.2) --
not a generic 0-1 "fault_severity" scale, because dc_motor's own Scenario has no such field (see
datagen/scenario.py: severity is two independent physical parameters, not one normalized knob).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent
from driveflow.datagen import Scenario, run_scenario

#: Same channel vocabulary as models.common.windowing.CANDIDATE_CHANNELS' physical subset --
#: current, speed, and 3-axis vibration. The armature current is the "current_r" column of the
#: telemetry schema (datagen/runner.py); this list used to say "i_a", a name no telemetry frame
#: carries, so detect_anomaly() silently dropped the current channel (AI_TECHNICAL_AUDIT AI-N05).
_DEFAULT_FEATURES = ["current_r", "rpm", "acc_x", "acc_y", "acc_z"]

_HYPOTHESES = {
    "healthy": {"fault_type": None},
    "incipient": {"fault_type": "outer_race", "electrical_severity": 3.0, "mechanical_severity": 0.02},
    "moderate": {"fault_type": "outer_race", "electrical_severity": 8.0, "mechanical_severity": 0.05},
    # Inner-race hypotheses (AI-N10): with outer_race ones only, inner_race runs scored 0.64-0.71,
    # half of them under the 0.70 alert threshold (docs/audit/ai/anomaly_agent_roc.json).
    "inner_race_incipient": {"fault_type": "inner_race", "electrical_severity": 3.0, "mechanical_severity": 0.01},
    "inner_race_moderate": {"fault_type": "inner_race", "electrical_severity": 8.0, "mechanical_severity": 0.02},
}


class DCMotorAnomalyDetector(SimulationBasedAgent):
    """Detecta fallas de rodamiento (outer_race, inner_race, ball, cage) comparando telemetría
    real contra simulaciones de Scenario(plant_config_id="dc_perm_ex_v1", ...) sanas vs. con
    falla a distinta severidad."""

    HYPOTHESES = _HYPOTHESES
    #: The bearing signature lives in the spectrum of vibration and current (AI-N10); the speed
    #: trajectory is compared sample by sample.
    SPECTRAL_FEATURES = ("acc_x", "acc_y", "acc_z", "current_r")
    #: rpm is smooth (the mechanical time constant is tens of ms), so its DTW runs on every 10th
    #: sample: 100x fewer cells for the same trajectory shape.
    TIME_DOMAIN_DECIMATION = 10

    def __init__(self, detector_metric: DetectorMetric = DetectorMetric.DTW):
        super().__init__(domain="dc_motor", detector_metric=detector_metric, monitored_features=list(_DEFAULT_FEATURES))

    def operating_point(self, telemetry_window: dict) -> dict:
        """Speed setpoint estimated from the last 10 % of the rpm trajectory, rounded to 1 rad/s
        so repeated runs at one setpoint share cached hypotheses (AI-N09). The PI loop holds the
        speed at its reference in steady state, faulted or not."""
        rpm = np.asarray(telemetry_window.get("rpm", []), dtype=float)
        rpm = rpm[np.isfinite(rpm)]
        if len(rpm) == 0:
            return {}
        tail = rpm[-max(1, len(rpm) // 10):]
        return {"omega_ref_rad_s": float(round(float(np.mean(tail)) * np.pi / 30.0))}

    def simulate_scenario(self, scenario_config: dict) -> dict:
        scenario = Scenario(scenario_id="agent_hypothesis", **scenario_config)
        records = run_scenario(scenario)
        df = pd.DataFrame.from_records(records)
        return {feat: df[feat].to_numpy(dtype=float) for feat in self.monitored_features if feat in df.columns}
