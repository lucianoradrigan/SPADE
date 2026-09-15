"""VSC/DPC (domain vsc_dpc) simulation-based anomaly detection agent.

Phase 4: this domain has no fault model at all -- pure power electronics, no bearing, no rotating
machinery (see docs/design_ai_layer_transversal.md and dashboard.py's own "About this platform").
"Anomaly" here means the load resistance drifting away from the ~8.0Ω point the trained DPC
network was validated on -- the same off-distribution robustness concept Fase B's own sidebar
already exposes as a slider (control/dpc/controller.py feeds load_resistance_ohm directly into
the network's input row) -- not a fault severity.
"""

from __future__ import annotations

import pandas as pd

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _VSC_R_OHM
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM

#: Same channel vocabulary as models.common.windowing.VSC_DPC_CANDIDATE_CHANNELS -- this domain's
#: real state channels (NOT the dq-frame v_d/v_q/i_d/i_q naming, which belongs to the unrelated
#: PMSM FOC/MTPA system, System 2, unconnected to this pipeline).
_DEFAULT_FEATURES = ["vc_real", "vc_imag", "i_f_real", "i_f_imag"]

#: "healthy" = the network's own training value (Scenario's default, so no override needed here);
#: "incipient"/"moderate" step progressively further from it but stay above
#: MIN_STABLE_LOAD_RESISTANCE_OHM -- below it the plant itself is open-loop unstable regardless
#: of the controller (sim/vsc_system.py), a different, already-covered failure mode
#: (monitoring/rules/vsc_dpc.yaml's hard-threshold rule), not this agent's job.
_HYPOTHESES = {
    "healthy": {},
    "incipient": {"load_resistance_ohm": float(_VSC_R_OHM) * 1.5},
    "moderate": {"load_resistance_ohm": max(float(_VSC_R_OHM) * 2.5, MIN_STABLE_LOAD_RESISTANCE_OHM + 1.0)},
}


class VSCDPCAnomalyDetector(SimulationBasedAgent):
    """Detecta desviaciones (resistencia de carga fuera de distribución) comparando telemetría
    real contra simulaciones de Scenario(controller_type="DPC", plant_config_id="vsc_dpc_v1",
    ...) a distinta resistencia de carga."""

    HYPOTHESES = _HYPOTHESES

    def __init__(self, detector_metric: DetectorMetric = DetectorMetric.EUCLIDEAN):
        super().__init__(domain="vsc_dpc", detector_metric=detector_metric, monitored_features=list(_DEFAULT_FEATURES))

    def simulate_scenario(self, scenario_config: dict) -> dict:
        scenario = Scenario(scenario_id="agent_hypothesis", controller_type="DPC", plant_config_id="vsc_dpc_v1", **scenario_config)
        records = run_scenario(scenario)
        df = pd.DataFrame.from_records(records)
        return {feat: df[feat].to_numpy(dtype=float) for feat in self.monitored_features if feat in df.columns}
