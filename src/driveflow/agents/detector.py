"""Distance-metric implementations backing SimulationBasedAgent.compute_distance.

Phase 4: real implementations. No DTW library is a project dependency (checked
pyproject.toml -- no dtaidistance/tslearn/fastdtw), so dtw_distance is a standard O(n*m)
dynamic-programming DTW written directly against numpy, not a hand-wave. It accepts both 1D
(single-feature) and 2D (n_samples, n_features) sequences, unlike euclidean_distance, which
requires equal-length/shape sequences (DTW's whole point is tolerating length mismatches; a plain
Euclidean distance has no sensible definition when the two sequences disagree in length).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import mahalanobis


class AnomalyDetector:
    """Funciones de distancia entre dos trayectorias (arrays 1D de un solo feature, o 2D
    (n_samples, n_features)), usadas para comparar una ventana de telemetría real contra una
    simulación hipotética."""

    @staticmethod
    def dtw_distance(seq1, seq2) -> float:
        """Dynamic Time Warping -- distancia acumulada del camino de menor costo alineando seq1
        contra seq2, tolerante a que tengan distinto largo (a diferencia de euclidean_distance)."""
        a = np.asarray(seq1, dtype=float)
        b = np.asarray(seq2, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            raise ValueError("dtw_distance needs non-empty sequences")

        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            diffs = a[i - 1] - b  # (m, n_features) -- vectorized over j
            point_costs = np.linalg.norm(diffs, axis=-1)
            for j in range(1, m + 1):
                cost[i, j] = point_costs[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return float(cost[n, m])

    @staticmethod
    def euclidean_distance(seq1, seq2) -> float:
        a = np.asarray(seq1, dtype=float)
        b = np.asarray(seq2, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"euclidean_distance needs equal-shape sequences, got {a.shape} vs {b.shape} -- use dtw_distance for sequences of different length")
        return float(np.linalg.norm(a - b))

    @staticmethod
    def mahalanobis_distance(seq1, seq2, cov_matrix) -> float:
        """Distancia de Mahalanobis entre dos puntos (seq1/seq2 se aplanan a un vector 1D cada
        uno -- Mahalanobis está definida entre puntos, no entre trayectorias, así que compararla
        con dtw_distance/euclidean_distance es comparar dos nociones distintas de "distancia").
        cov_matrix debe ser cuadrada y de tamaño igual al vector aplanado; se usa su pseudo-
        inversa (np.linalg.pinv), no la inversa exacta, para no fallar en una covarianza
        casi-singular (ej. muy pocas muestras para estimarla bien)."""
        a = np.asarray(seq1, dtype=float).ravel()
        b = np.asarray(seq2, dtype=float).ravel()
        cov = np.asarray(cov_matrix, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"mahalanobis_distance needs equal-size sequences once flattened, got {a.shape} vs {b.shape}")
        if cov.shape != (a.size, a.size):
            raise ValueError(f"cov_matrix must be ({a.size}, {a.size}) to match the flattened sequences, got {cov.shape}")
        return float(mahalanobis(a, b, np.linalg.pinv(cov)))
