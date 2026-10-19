"""Distance-metric implementations backing SimulationBasedAgent.compute_distance.

Phase 4: real implementations. No DTW library is a project dependency (checked
pyproject.toml -- no dtaidistance/tslearn/fastdtw), so dtw_distance is a standard O(n*m)
dynamic-programming DTW written directly against numpy, vectorized per row (AI-N10). It accepts both 1D
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

        # Row-vectorized exact DTW (AI_TECHNICAL_AUDIT AI-N10; the double Python loop took ~1 s per
        # 1500x1500 pair). Within row i the recurrence
        #   C[i, j] = d[j] + min(C[i-1, j], C[i-1, j-1], C[i, j-1])
        # unrolls, with t[j] = d[j] + min(C[i-1, j], C[i-1, j-1]) and S = cumsum(d), to
        #   C[i, j] = S[j] + min_{k <= j} (t[k] - S[k]),
        # a running minimum (np.minimum.accumulate) -- same values, O(m) numpy work per row.
        prev = np.full(m + 1, np.inf)
        prev[0] = 0.0
        for i in range(1, n + 1):
            d = np.linalg.norm(a[i - 1] - b, axis=-1)
            t = d + np.minimum(prev[1:], prev[:-1])
            csum = np.cumsum(d)
            row = np.empty(m + 1)
            row[0] = np.inf
            row[1:] = csum + np.minimum.accumulate(t - csum)
            prev = row
        return float(prev[m])

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
