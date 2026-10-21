"""Particle swarm optimisation in numpy -- expansion task E7 (decision D1: own implementation, no
pyswarms). Idea from Fund. ``1_MHA/Single_Objective_MHA/`` (which uses pyswarms); written from the
standard global-best PSO (Kennedy & Eberhart 1995, inertia weight of Shi & Eberhart 1998), no code
copied.

Minimises ``f(x)`` over a box. Positions are clipped to the bounds and velocities to a fraction of
the box width. Every evaluation is recorded in order, so the GUI can draw best-so-far against
evaluation count. Seeded: the same seed and the same function give the same history.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class OptimResult:
    x_best: np.ndarray
    f_best: float
    X: np.ndarray          # every evaluated point, in evaluation order
    F: np.ndarray          # its objective value
    method: str
    seed: int

    @property
    def best_so_far(self) -> np.ndarray:
        return np.minimum.accumulate(self.F)

    @property
    def n_evaluations(self) -> int:
        return len(self.F)


def check_bounds(bounds) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2 or not np.all(np.isfinite(b)) or np.any(b[:, 0] >= b[:, 1]):
        raise ValueError(f"bounds must be a (d, 2) array of finite [low, high] with low < high, got {bounds!r}")
    return b


def pso(f, bounds, n_particles: int = 16, iterations: int = 30, seed: int = 0, inertia: float = 0.72, c_cognitive: float = 1.49,
        c_social: float = 1.49, v_max_frac: float = 0.2, callback=None) -> OptimResult:
    """Global-best PSO. ``callback(i_eval, x, fx)`` is called after every evaluation (progress,
    cancellation: raise to stop)."""
    b = check_bounds(bounds)
    if n_particles < 2 or iterations < 1:
        raise ValueError("need n_particles >= 2 and iterations >= 1")
    rng = np.random.default_rng(seed)
    lo, hi = b[:, 0], b[:, 1]
    span = hi - lo
    v_max = v_max_frac * span
    x = lo + rng.random((n_particles, len(lo))) * span
    v = (rng.random(x.shape) * 2 - 1) * v_max
    X, F = [], []

    def evaluate(points):
        vals = []
        for p in points:
            fx = float(f(p))
            X.append(p.copy())
            F.append(fx)
            if callback is not None:
                callback(len(F), p, fx)
            vals.append(fx)
        return np.array(vals)

    fx = evaluate(x)
    p_best, p_val = x.copy(), fx.copy()
    g = int(np.argmin(p_val))
    for _ in range(iterations - 1):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = inertia * v + c_cognitive * r1 * (p_best - x) + c_social * r2 * (p_best[g] - x)
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, lo, hi)
        fx = evaluate(x)
        better = fx < p_val
        p_best[better], p_val[better] = x[better], fx[better]
        g = int(np.argmin(p_val))
    return OptimResult(p_best[g].copy(), float(p_val[g]), np.array(X), np.array(F), "pso", seed)
