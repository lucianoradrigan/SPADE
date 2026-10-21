"""NSGA-II in numpy -- expansion task E7 (decision D1: own implementation, no pymoo).

Idea from Fund. ``1_MHA/Multi_Objective_MHA/`` (which uses pymoo); written from Deb et al. (2002),
"A fast and elitist multiobjective genetic algorithm: NSGA-II", no code copied:
- fast non-dominated sorting and crowding distance;
- binary tournament on (rank, crowding);
- simulated binary crossover (SBX, η_c = 15) and polynomial mutation (η_m = 20, probability 1/d);
- elitist (μ + λ) survival.

Minimises a vector objective ``f(x) -> array(m)``. Seeded; every evaluation is recorded in order.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from driveflow.optim.pso import check_bounds


@dataclasses.dataclass
class ParetoResult:
    X: np.ndarray            # every evaluated point, in order
    F: np.ndarray            # (n, m) objectives
    front_idx: np.ndarray    # indices (into X/F) of the final non-dominated set
    method: str
    seed: int

    @property
    def front_X(self) -> np.ndarray:
        return self.X[self.front_idx]

    @property
    def front_F(self) -> np.ndarray:
        return self.F[self.front_idx]


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    n = len(F)
    dominated_by = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                dominated_by[i].append(j)
                n_dom[j] += 1
            elif dominates(F[j], F[i]):
                dominated_by[j].append(i)
                n_dom[i] += 1
    fronts, current = [], np.flatnonzero(n_dom == 0)
    while len(current):
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        current = np.array(sorted(nxt), dtype=int)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[order[-1], k] - F[order[0], k]
        if span > 0:
            d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return d


def _sbx(p1, p2, lo, hi, rng, eta=15.0):
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    swap = rng.random(len(p1)) < 0.5  # uniform exchange per gene, as in the reference implementation
    c1[swap], c2[swap] = c2[swap].copy(), c1[swap].copy()
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def _mutate(x, lo, hi, rng, eta=20.0):
    x = x.copy()
    for i in range(len(x)):
        if rng.random() < 1.0 / len(x):
            u = rng.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x[i] = np.clip(x[i] + delta * (hi[i] - lo[i]), lo[i], hi[i])
    return x


def nsga2(f, bounds, population: int = 24, generations: int = 20, seed: int = 0, callback=None) -> ParetoResult:
    b = check_bounds(bounds)
    if population < 4 or population % 2:
        raise ValueError("population must be an even number >= 4")
    rng = np.random.default_rng(seed)
    lo, hi = b[:, 0], b[:, 1]
    X_all, F_all = [], []

    def evaluate(P):
        out = []
        for p in P:
            fx = np.asarray(f(p), dtype=float)
            X_all.append(p.copy())
            F_all.append(fx)
            if callback is not None:
                callback(len(F_all), p, fx)
            out.append(fx)
        return np.array(out)

    P = lo + rng.random((population, len(lo))) * (hi - lo)
    FP = evaluate(P)
    idx_P = np.arange(population)

    def ranks_and_crowd(F):
        rank = np.empty(len(F), dtype=int)
        crowd = np.empty(len(F))
        for r, front in enumerate(non_dominated_sort(F)):
            rank[front] = r
            crowd[front] = crowding_distance(F[front])
        return rank, crowd

    for _ in range(generations - 1):
        rank, crowd = ranks_and_crowd(FP)

        def tournament():
            i, j = rng.integers(population, size=2)
            if rank[i] != rank[j]:
                return i if rank[i] < rank[j] else j
            return i if crowd[i] >= crowd[j] else j

        children = []
        while len(children) < population:
            c1, c2 = _sbx(P[tournament()], P[tournament()], lo, hi, rng)
            children += [_mutate(c1, lo, hi, rng), _mutate(c2, lo, hi, rng)]
        Q = np.array(children[:population])
        FQ = evaluate(Q)
        idx_Q = np.arange(len(F_all) - population, len(F_all))
        R, FR, idx_R = np.vstack([P, Q]), np.vstack([FP, FQ]), np.concatenate([idx_P, idx_Q])
        keep = []
        for front in non_dominated_sort(FR):
            if len(keep) + len(front) <= population:
                keep += list(front)
            else:
                cd = crowding_distance(FR[front])
                keep += list(front[np.argsort(-cd)[: population - len(keep)]])
                break
        keep = np.array(keep)
        P, FP, idx_P = R[keep], FR[keep], idx_R[keep]
    F_arr = np.array(F_all)
    final_front = idx_P[non_dominated_sort(FP)[0]]
    return ParetoResult(np.array(X_all), F_arr, np.sort(final_front), "nsga2", seed)


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 test problem (n variables in [0, 1]); the true front is f2 = 1 - sqrt(f1)."""
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:]) if len(x) > 1 else 1.0
    return np.array([f1, g * (1 - np.sqrt(f1 / g))])


def igd_zdt1(front: np.ndarray, n_ref: int = 200) -> float:
    """Inverted generational distance of a front to ZDT1's true front (lower is better)."""
    f1 = np.linspace(0, 1, n_ref)
    ref = np.stack([f1, 1 - np.sqrt(f1)], axis=1)
    d = np.sqrt(((ref[:, None, :] - front[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(d.mean())
