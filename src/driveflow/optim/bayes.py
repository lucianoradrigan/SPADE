"""Bayesian optimisation with a Gaussian process -- expansion task E7 (decision D1).

Idea from Fund. ``2_Classic_ML/gaussian_process_bayesian_optimization.ipynb``; written on
``sklearn.gaussian_process.GaussianProcessRegressor`` (already a dependency), no code copied.

- Surrogate: GP with a constant × Matérn(ν = 2.5) kernel plus a white-noise term, on inputs scaled
  to [0, 1] and standardised outputs.
- Acquisition: expected improvement (minimisation), maximised over a seeded random candidate set,
  ``n_candidates`` points per step.
- Start: a seeded Latin-hypercube design of ``n_init`` points.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from driveflow.optim.pso import OptimResult, check_bounds


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma
    return (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayes_opt(f, bounds, n_init: int = 8, iterations: int = 24, seed: int = 0, n_candidates: int = 2000, xi: float = 0.01, callback=None) -> OptimResult:
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    b = check_bounds(bounds)
    if n_init < 2 or iterations < 0:
        raise ValueError("need n_init >= 2 and iterations >= 0")
    rng = np.random.default_rng(seed)
    lo, span = b[:, 0], b[:, 1] - b[:, 0]
    d = len(lo)
    U = list(qmc.LatinHypercube(d=d, seed=seed).random(n_init))
    X, F = [], []

    def evaluate(u):
        x = lo + u * span
        fx = float(f(x))
        X.append(x)
        F.append(fx)
        if callback is not None:
            callback(len(F), x, fx)

    for u in U:
        evaluate(u)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 10.0), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
    for _ in range(iterations):
        Fa = np.array(F)
        mu_f, sd_f = Fa.mean(), Fa.std() or 1.0
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=int(rng.integers(2**31)), n_restarts_optimizer=2)
        with warnings.catch_warnings():  # lbfgs hyper-parameter fits on tiny data warn routinely
            warnings.simplefilter("ignore")
            gp.fit(np.array(U), (Fa - mu_f) / sd_f)
        cand = rng.random((n_candidates, d))
        mu, sigma = gp.predict(cand, return_std=True)
        ei = expected_improvement(mu, sigma, (Fa.min() - mu_f) / sd_f, xi)
        u = cand[int(np.argmax(ei))]
        U.append(u)
        evaluate(u)
    F_arr = np.array(F)
    k = int(np.argmin(F_arr))
    return OptimResult(np.array(X[k]), float(F_arr[k]), np.array(X), F_arr, "bayes", seed)
