"""Distance-metric implementations backing SimulationBasedAgent.compute_distance.

Phase 1 scaffolding only.
"""

from __future__ import annotations


class AnomalyDetector:
    """Funciones de distancia entre dos trayectorias (listas de valores en el tiempo), usadas
    para comparar una ventana de telemetría real contra una simulación hipotética."""

    @staticmethod
    def dtw_distance(seq1: list, seq2: list) -> float:
        raise NotImplementedError("Phase 4")

    @staticmethod
    def euclidean_distance(seq1: list, seq2: list) -> float:
        raise NotImplementedError("Phase 4")

    @staticmethod
    def mahalanobis_distance(seq1: list, seq2: list, cov_matrix: list) -> float:
        raise NotImplementedError("Phase 4")
