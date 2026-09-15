"""Base class for simulation-based anomaly-detection agents.

Phase 1 scaffolding only -- concrete subclasses (dc_motor_agent.py, vsc_agent.py) are also stubs;
detect_anomaly/simulate_scenario raise NotImplementedError until Phase 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class DetectorMetric(Enum):
    DTW = "dtw"
    EUCLIDEAN = "euclidean"
    MAHALANOBIS = "mahalanobis"


class SimulationBasedAgent(ABC):
    """Compara una ventana de telemetría real contra simulaciones de escenarios hipotéticos
    (sano, incipiente, moderado, severo) para estimar qué tan anómala es. Cada dominio
    (dc_motor, vsc_dpc) tiene su propia subclase concreta -- qué features monitorea y cómo
    simula un escenario es específico de cada planta."""

    def __init__(self, domain: str, detector_metric: DetectorMetric = DetectorMetric.DTW, monitored_features: list | None = None):
        self.domain = domain
        self.detector_metric = detector_metric
        self.monitored_features = monitored_features or []
        self.cache: dict = {}

    @abstractmethod
    def simulate_scenario(self, scenario_config: dict) -> dict:
        """Corre una simulación de un escenario hipotético (vía Scenario/run_scenario del
        dominio correspondiente) y devuelve las trayectorias de monitored_features."""

    @abstractmethod
    def detect_anomaly(self, telemetry_window: dict) -> tuple[float, str]:
        """Devuelve (anomaly_score en [0,1], diagnóstico) para una ventana de telemetría real."""

    def compute_distance(self, trajectory_1: list, trajectory_2: list) -> float:
        """Distancia entre dos trayectorias según self.detector_metric (ver detector.py)."""
        raise NotImplementedError("Phase 4")

    def clear_cache(self) -> None:
        self.cache.clear()
