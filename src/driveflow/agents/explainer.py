"""Explains why a simulation-based agent flagged a given anomaly score.

Phase 4: explain()/feature_importance() both take the telemetry window and the matching
simulated hypothesis they were compared against -- Phase 1's stub had explain() take no
arguments at all, which can't actually explain anything beyond restating anomaly_score/diagnosis
already known from __init__. Extended here: the whole point of this module (Sec. 1.1's
"Explicabilidad (feature importance)") needs that data to say anything real.
"""

from __future__ import annotations


class AnomalyExplainer:
    """A partir de una telemetría real y la simulación de la hipótesis más cercana que se le
    comparó (SimulationBasedAgent.detect_anomaly ya decide cuál es), calcula qué features
    explican más la distancia entre ambas."""

    def __init__(self, anomaly_score: float, diagnosis: str):
        self.anomaly_score = anomaly_score
        self.diagnosis = diagnosis

    def explain(self, telemetry: dict, simulation: dict) -> dict:
        importance = self.feature_importance(telemetry, simulation)
        top_feature = max(importance, key=importance.get) if importance else None
        summary = f"Closest to hypothesis '{self.diagnosis}' (score {self.anomaly_score:.2f})"
        if top_feature:
            summary += f" -- '{top_feature}' contributes the most to that."
        return {
            "anomaly_score": self.anomaly_score,
            "diagnosis": self.diagnosis,
            "feature_importance": importance,
            "most_responsible_feature": top_feature,
            "summary": summary,
        }

    def feature_importance(self, telemetry: dict, simulation: dict) -> dict:
        """Peso (sumando 1.0) de cuánto contribuye cada feature compartida a la distancia total
        entre telemetry y simulation -- euclidean_distance cuando los largos coinciden (el caso
        normal: ambos vienen de la misma duration_s), dtw_distance si no (tolera el desfasaje)."""
        from driveflow.agents.detector import AnomalyDetector

        shared_features = [f for f in telemetry if f in simulation]
        if not shared_features:
            return {}
        raw = {}
        for feat in shared_features:
            try:
                raw[feat] = AnomalyDetector.euclidean_distance(telemetry[feat], simulation[feat])
            except ValueError:
                raw[feat] = AnomalyDetector.dtw_distance(telemetry[feat], simulation[feat])
        total = sum(raw.values())
        if total <= 0:
            return dict.fromkeys(raw, 0.0)
        return {feat: float(d / total) for feat, d in raw.items()}
