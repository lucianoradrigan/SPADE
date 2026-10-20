"""VSC/DPC (domain vsc_dpc) simulation-based anomaly detection agent.

Pure power electronics, no bearing, no rotating machinery. Two kinds of anomaly hypotheses:
- the load resistance drifting away from the ~8.0 ohm point the trained DPC network was validated
  on -- the off-distribution robustness concept the VSC page exposes as a slider;
- since 2026-10-20, a degraded filter capacitor (80 % and 70 % of the nominal 15 uF;
  sim/vsc_faults.py). This fault definition is SYNTHETIC-ONLY: no real reference data exists.
  Separability passed a pre-registered cut even with +-10 % R/L/R_f uncertainty
  (docs/audit/ai/vsc_capacitor_separability.md); experiments/report_vsc_agent_capacitor.py
  measures this agent on it.
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
    # Synthetic filter-capacitor degradation (A4 follow-up, sim/vsc_faults.py): separability passed
    # a pre-registered cut even with +-10 % R/L/R_f uncertainty (docs/audit/ai/
    # vsc_capacitor_separability.md). SYNTHETIC-ONLY -- no real reference data for this fault.
    "capacitor_80pct": {"vsc_capacitor_scale": 0.8},
    "capacitor_70pct": {"vsc_capacitor_scale": 0.7},
}


class VSCDPCAnomalyDetector(SimulationBasedAgent):
    """Detecta desviaciones (resistencia de carga fuera de distribución, o capacitor del filtro
    degradado -- sintético) comparando telemetría real contra simulaciones de
    Scenario(controller_type="DPC", plant_config_id="vsc_dpc_v1", ...)."""

    HYPOTHESES = _HYPOTHESES

    def __init__(self, detector_metric: DetectorMetric = DetectorMetric.EUCLIDEAN):
        super().__init__(domain="vsc_dpc", detector_metric=detector_metric, monitored_features=list(_DEFAULT_FEATURES))

    def simulate_scenario(self, scenario_config: dict) -> dict:
        scenario = Scenario(scenario_id="agent_hypothesis", controller_type="DPC", plant_config_id="vsc_dpc_v1", **scenario_config)
        records = run_scenario(scenario)
        df = pd.DataFrame.from_records(records)
        return {feat: df[feat].to_numpy(dtype=float) for feat in self.monitored_features if feat in df.columns}
