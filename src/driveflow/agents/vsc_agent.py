"""VSC/DPC (domain vsc_dpc) simulation-based anomaly detection agent.

Phase 1 scaffolding only.
"""

from __future__ import annotations

from driveflow.agents.base import DetectorMetric, SimulationBasedAgent

#: Same channel vocabulary as models.common.windowing.VSC_DPC_CANDIDATE_CHANNELS -- this domain's
#: real state channels (NOT the dq-frame v_d/v_q/i_d/i_q naming, which belongs to the unrelated
#: PMSM FOC/MTPA system, System 2, unconnected to this pipeline).
_DEFAULT_FEATURES = ["vc_real", "vc_imag", "i_f_real", "i_f_imag"]


class VSCDPCAnomalyDetector(SimulationBasedAgent):
    """Detecta desviaciones (ej. resistencia de carga fuera de distribución) comparando
    telemetría real contra simulaciones de Scenario(controller_type="DPC",
    plant_config_id="vsc_dpc_v1", ...) a distinta resistencia de carga."""

    def __init__(self, detector_metric: DetectorMetric = DetectorMetric.EUCLIDEAN):
        super().__init__(domain="vsc_dpc", detector_metric=detector_metric, monitored_features=list(_DEFAULT_FEATURES))

    def simulate_scenario(self, scenario_config: dict) -> dict:
        raise NotImplementedError("Phase 4")

    def detect_anomaly(self, telemetry_window: dict) -> tuple[float, str]:
        raise NotImplementedError("Phase 4")
