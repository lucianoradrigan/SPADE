"""Base class for simulation-based anomaly-detection agents.

Phase 4: detect_anomaly is now a CONCRETE, shared algorithm -- not abstract per subclass like
Phase 1's scaffolding had it. It compares a telemetry window against each entry of
self.HYPOTHESES (a class attribute each subclass defines: name -> partial Scenario kwargs for one
hypothesis), scored by how much closer the telemetry sits to the "healthy" hypothesis than to the
nearest non-healthy one. Only simulate_scenario stays abstract, since running a hypothesis
simulation is genuinely domain-specific (a different Scenario shape per domain, see
dc_motor_agent.py/vsc_agent.py) -- the comparison/scoring logic above it is not, and an earlier
draft duplicated that same logic in every subclass with no domain-specific content in it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from driveflow.agents.cache import SimulationCache
from driveflow.datagen.runner import TAU


class DetectorMetric(Enum):
    DTW = "dtw"
    EUCLIDEAN = "euclidean"
    MAHALANOBIS = "mahalanobis"


class SimulationBasedAgent(ABC):
    """Compara una ventana de telemetría real contra simulaciones de escenarios hipotéticos
    (sano + N variantes anómalas) para estimar qué tan anómala es. Cada dominio (dc_motor,
    vsc_dpc) tiene su propia subclase concreta -- qué features monitorea, cómo simula un
    escenario, y cuáles son sus hipótesis concretas es específico de cada planta."""

    #: name -> partial Scenario kwargs para una hipótesis (duration_s/seed los agrega
    #: detect_anomaly). DEBE incluir una entrada "healthy" -- el score de abajo es relativo a
    #: ella. Las subclases la definen (ver dc_motor_agent.py/vsc_agent.py).
    HYPOTHESES: dict = {}

    def __init__(self, domain: str, detector_metric: DetectorMetric = DetectorMetric.DTW, monitored_features: list | None = None, cache: SimulationCache | None = None):
        self.domain = domain
        self.detector_metric = detector_metric
        self.monitored_features = monitored_features or []
        self.cache = cache if cache is not None else SimulationCache()

    @abstractmethod
    def simulate_scenario(self, scenario_config: dict) -> dict:
        """Corre una hipótesis (una llamada real a Scenario/run_scenario de este dominio) y
        devuelve {nombre_de_feature: np.ndarray} para self.monitored_features."""

    def detect_anomaly(self, telemetry_window: dict) -> tuple:
        """(anomaly_score en [0,1], nombre de la hipótesis más cercana). score = 0 -> tan cerca
        de "healthy" como de cualquier hipótesis de falla (o más cerca de "healthy"); score
        cercano a 1 -> mucho más cerca de una hipótesis de falla que de "healthy". La duración de
        cada simulación de hipótesis se infiere de len(telemetry_window) * TAU (mismo paso fijo
        que usa el simulador real en todo el proyecto), para comparar trayectorias del mismo
        largo. Resultados por (hipótesis, duración) se cachean en self.cache -- simular es caro."""
        if "healthy" not in self.HYPOTHESES:
            raise ValueError(f"{type(self).__name__}.HYPOTHESES must include a 'healthy' entry")
        present = [f for f in self.monitored_features if f in telemetry_window]
        if not present:
            raise ValueError(f"telemetry_window has none of the monitored features {self.monitored_features}")
        n_samples = len(np.asarray(telemetry_window[present[0]]))
        duration_s = n_samples * TAU

        distances = {}
        for name, overrides in self.HYPOTHESES.items():
            cache_key = (name, round(duration_s, 6))
            sim = self.cache.get(cache_key)
            if sim is None:
                sim = self.simulate_scenario({**overrides, "duration_s": duration_s, "seed": 0})
                self.cache.set(cache_key, sim)
            per_feature = [self.compute_distance(telemetry_window[f], sim[f]) for f in present if f in sim]
            distances[name] = float(np.mean(per_feature)) if per_feature else float("inf")

        closest = min(distances, key=distances.get)
        healthy_d = distances["healthy"]
        fault_distances = [d for name, d in distances.items() if name != "healthy"]
        fault_d = min(fault_distances) if fault_distances else healthy_d
        total = healthy_d + fault_d
        score = healthy_d / total if total > 0 else 0.0
        return float(np.clip(score, 0.0, 1.0)), closest

    def compute_distance(self, trajectory_1, trajectory_2) -> float:
        """Distancia entre dos trayectorias según self.detector_metric (ver detector.py).
        MAHALANOBIS no pasa por acá -- necesita una matriz de covarianza que este método no
        recibe; llamar AnomalyDetector.mahalanobis_distance directamente para esa métrica."""
        from driveflow.agents.detector import AnomalyDetector

        if self.detector_metric is DetectorMetric.DTW:
            return AnomalyDetector.dtw_distance(trajectory_1, trajectory_2)
        if self.detector_metric is DetectorMetric.EUCLIDEAN:
            return AnomalyDetector.euclidean_distance(trajectory_1, trajectory_2)
        raise ValueError(f"{self.detector_metric} needs a covariance matrix -- call AnomalyDetector.mahalanobis_distance directly, not through compute_distance")

    def clear_cache(self) -> None:
        """Limpiar caché de simulaciones."""
        self.cache.clear()
