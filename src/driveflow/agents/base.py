"""Base class for simulation-based anomaly-detection agents.

Phase 4: detect_anomaly is now a CONCRETE, shared algorithm -- not abstract per subclass like
Phase 1's scaffolding had it. It compares a telemetry window against each entry of
self.HYPOTHESES (a class attribute each subclass defines: name -> partial Scenario kwargs for one
hypothesis), scored by how much closer the telemetry sits to the "healthy" hypothesis than to the
nearest non-healthy one. Only simulate_scenario stays abstract, since running a hypothesis
simulation is genuinely domain-specific (a different Scenario shape per domain, see
dc_motor_agent.py/vsc_agent.py) -- the comparison/scoring logic above it is not, and an earlier
draft duplicated that same logic in every subclass with no domain-specific content in it.

docs/AI_TECHNICAL_AUDIT.md fixes:
- AI-N09: hypotheses are simulated at the OBSERVED operating point (operating_point(), e.g. the
  DC motor's speed), not always at the Scenario defaults -- a healthy motor at 120 rad/s used to
  match the "moderate" fault simulated at 150 rad/s. The operating point is part of the cache key.
- AI-N10: each feature is z-scored with the healthy hypothesis's own mean and standard deviation
  before the distance, so a feature in large units (rpm, ~1e3) no longer outweighs the vibration
  channels that carry the bearing signature. Features listed in SPECTRAL_FEATURES are compared by
  their log power spectral density over the trailing SPECTRAL_TAIL_FRACTION of the window instead
  of sample by sample: two vibration records are independent noise realizations, so a time-domain
  DTW between them measured the noise (the score sat at 0.50 for healthy and faulty runs alike),
  while the bearing fault adds spectral energy that a PSD comparison sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.signal import welch

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
    #: Features compared through their log-PSD (see module docstring); others sample by sample.
    SPECTRAL_FEATURES: tuple = ()
    SPECTRAL_TAIL_FRACTION = 0.5
    #: Keep every k-th sample of the non-spectral features before the time-domain distance.
    TIME_DOMAIN_DECIMATION = 1

    def __init__(self, domain: str, detector_metric: DetectorMetric = DetectorMetric.DTW, monitored_features: list | None = None, cache: SimulationCache | None = None):
        self.domain = domain
        self.detector_metric = detector_metric
        self.monitored_features = monitored_features or []
        self.cache = cache if cache is not None else SimulationCache()
        self.last_operating_point: dict = {}

    @abstractmethod
    def simulate_scenario(self, scenario_config: dict) -> dict:
        """Corre una hipótesis (una llamada real a Scenario/run_scenario de este dominio) y
        devuelve {nombre_de_feature: np.ndarray} para self.monitored_features."""

    def operating_point(self, telemetry_window: dict) -> dict:
        """Scenario kwargs describing where the telemetry was recorded (AI-N09); merged into every
        hypothesis. The base class has none; subclasses estimate it from the telemetry."""
        return {}

    @staticmethod
    def _feature_scale(reference: np.ndarray) -> tuple:
        """(mean, scale) of the healthy hypothesis's trajectory for one feature. A (near-)
        constant trajectory has no usable standard deviation; its magnitude (or 1) is used."""
        ref = np.asarray(reference, dtype=float)
        mu, sd = float(np.mean(ref)), float(np.std(ref))
        if sd > 1e-6 * (abs(mu) + 1.0):
            return mu, sd
        return mu, abs(mu) if abs(mu) > 0 else 1.0

    def _log_psd(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[len(x) - max(2, int(len(x) * self.SPECTRAL_TAIL_FRACTION)):]
        _f, p = welch(x - x.mean(), fs=1.0 / TAU, nperseg=min(512, len(x)))
        return np.log10(p + 1e-20)

    def _feature_distance(self, feature: str, telemetry, simulated, scale: tuple) -> float:
        if feature in self.SPECTRAL_FEATURES:
            a, b = self._log_psd(telemetry), self._log_psd(simulated)
            return float(np.sqrt(np.mean((a - b) ** 2)))
        mu, sc = scale
        k = self.TIME_DOMAIN_DECIMATION
        return self.compute_distance((np.asarray(telemetry, dtype=float)[::k] - mu) / sc, (np.asarray(simulated, dtype=float)[::k] - mu) / sc)

    def _hypothesis_key(self, name: str, duration_s: float, op: dict) -> tuple:
        return (name, round(duration_s, 6), tuple(sorted(op.items())))

    def hypothesis_simulation(self, name: str, n_samples: int, operating_point: dict | None = None):
        """The cached simulation detect_anomaly compared against (None if never simulated)."""
        op = self.last_operating_point if operating_point is None else operating_point
        return self.cache.get(self._hypothesis_key(name, n_samples * TAU, op or {}))

    def detect_anomaly(self, telemetry_window: dict, operating_point: dict | None = None) -> tuple:
        """(anomaly_score en [0,1], nombre de la hipótesis más cercana). score = 0 -> tan cerca
        de "healthy" como de cualquier hipótesis de falla (o más cerca de "healthy"); score
        cercano a 1 -> mucho más cerca de una hipótesis de falla que de "healthy". La duración de
        cada simulación de hipótesis se infiere de len(telemetry_window) * TAU (mismo paso fijo
        que usa el simulador real en todo el proyecto), para comparar trayectorias del mismo
        largo. Resultados por (hipótesis, duración, punto de operación) se cachean en self.cache --
        simular es caro. operating_point: kwargs de Scenario del punto de operación observado; None
        lo estima self.operating_point(telemetry_window) (AI-N09)."""
        if "healthy" not in self.HYPOTHESES:
            raise ValueError(f"{type(self).__name__}.HYPOTHESES must include a 'healthy' entry")
        present = [f for f in self.monitored_features if f in telemetry_window]
        if not present:
            raise ValueError(f"telemetry_window has none of the monitored features {self.monitored_features}")
        n_samples = len(np.asarray(telemetry_window[present[0]]))
        duration_s = n_samples * TAU
        op = dict(self.operating_point(telemetry_window) if operating_point is None else operating_point)
        self.last_operating_point = op

        sims = {}
        for name, overrides in self.HYPOTHESES.items():
            cache_key = self._hypothesis_key(name, duration_s, op)
            sim = self.cache.get(cache_key)
            if sim is None:
                sim = self.simulate_scenario({**overrides, **op, "duration_s": duration_s, "seed": 0})
                self.cache.set(cache_key, sim)
            sims[name] = sim
        scales = {f: self._feature_scale(sims["healthy"][f]) for f in present if f in sims["healthy"]}

        distances = {}
        for name, sim in sims.items():
            per_feature = [self._feature_distance(f, telemetry_window[f], sim[f], scales[f]) for f in present if f in sim and f in scales]
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
