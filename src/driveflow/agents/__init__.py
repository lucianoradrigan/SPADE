"""Simulation-Based Agents for SPADE/driveflow.

Detectan anomalías comparando una ventana de telemetría en vivo contra simulaciones hipotéticas
(baseline sano vs. escenarios de falla a distinta severidad) -- distinto de
driveflow.monitoring.agents, que evalúa reglas de umbral duro sobre telemetría, sin simular nada.
Los dos tipos de agente son complementarios: monitoring.agents.ServerAgent es quien terminaría
agregando las alertas de ambos (Phase 6 conecta esa integración; no está hecha todavía).

Phase 4: implementación real -- detector.py tiene DTW/Euclidean/Mahalanobis de verdad (sin
librería externa, este proyecto no depende de ninguna), y dc_motor_agent.py/vsc_agent.py simulan
hipótesis reales (Scenario/run_scenario) para compararlas contra telemetría. SimulationCache
(cache.py) es el mecanismo de caché real que usa SimulationBasedAgent -- no un dict plano
paralelo. Sin UI todavía (Phase 5) ni conexión con monitoring.agents.ServerAgent (Phase 6).
"""

from driveflow.agents.base import SimulationBasedAgent
from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector
from driveflow.agents.detector import AnomalyDetector
from driveflow.agents.explainer import AnomalyExplainer

__all__ = [
    "SimulationBasedAgent",
    "DCMotorAnomalyDetector",
    "VSCDPCAnomalyDetector",
    "AnomalyDetector",
    "AnomalyExplainer",
]
