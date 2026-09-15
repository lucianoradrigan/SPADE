"""Explains why a simulation-based agent flagged a given anomaly score.

Phase 1 scaffolding only.
"""

from __future__ import annotations


class AnomalyExplainer:
    """A partir de una telemetría real y la simulación más cercana que se le comparó, calcula
    qué features explican más la distancia (Sec. 1.2's "Explicabilidad")."""

    def __init__(self, anomaly_score: float, diagnosis: str):
        self.anomaly_score = anomaly_score
        self.diagnosis = diagnosis

    def explain(self) -> dict:
        raise NotImplementedError("Phase 4")

    def feature_importance(self, telemetry: dict, simulation: dict) -> dict:
        raise NotImplementedError("Phase 4")
