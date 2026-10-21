"""Own optimisers of expansion task E7 (decision D1) on reference functions, seeded."""

import numpy as np
import pytest

from driveflow.optim.bayes import bayes_opt
from driveflow.optim.multiobjective import crowding_distance, igd_zdt1, non_dominated_sort, nsga2, zdt1
from driveflow.optim.pso import check_bounds, pso


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def rastrigin(x):
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def test_pso_minimises_sphere_and_is_reproducible():
    a = pso(sphere, [[-5, 5]] * 4, n_particles=20, iterations=60, seed=3)
    b = pso(sphere, [[-5, 5]] * 4, n_particles=20, iterations=60, seed=3)
    c = pso(sphere, [[-5, 5]] * 4, n_particles=20, iterations=60, seed=4)
    assert a.f_best < 1e-4 and np.array_equal(a.F, b.F) and not np.array_equal(a.F, c.F)
    assert a.n_evaluations == 20 * 60 and np.all(np.diff(a.best_so_far) <= 0)
    assert np.all((a.X >= -5) & (a.X <= 5))


def test_pso_on_rastrigin_finds_a_low_basin():
    r = pso(rastrigin, [[-5.12, 5.12]] * 2, n_particles=30, iterations=100, seed=0)
    assert r.f_best < 1.0  # global minimum 0; local minima are at integer points with f >= 1


def test_bayes_minimises_sphere_in_few_evaluations_and_is_reproducible():
    a = bayes_opt(sphere, [[-5, 5]] * 2, n_init=6, iterations=20, seed=1, n_candidates=1000)
    b = bayes_opt(sphere, [[-5, 5]] * 2, n_init=6, iterations=20, seed=1, n_candidates=1000)
    assert a.n_evaluations == 26 and a.f_best < 0.1 and np.array_equal(a.F, b.F)


def test_nsga2_approaches_the_zdt1_front_and_is_reproducible():
    a = nsga2(zdt1, [[0, 1]] * 6, population=40, generations=60, seed=2)
    b = nsga2(zdt1, [[0, 1]] * 6, population=40, generations=60, seed=2)
    assert np.array_equal(a.F, b.F) and np.array_equal(a.front_idx, b.front_idx)
    assert igd_zdt1(a.front_F) < 0.05 and len(a.front_idx) >= 10
    random_front = np.array([zdt1(x) for x in np.random.default_rng(0).random((2400, 6))])
    nd = random_front[non_dominated_sort(random_front)[0]]
    assert igd_zdt1(a.front_F) < igd_zdt1(nd) / 3  # far better than random search with the same budget


def test_sorting_and_crowding():
    F = np.array([[1, 4], [2, 3], [3, 3], [4, 1], [5, 5]])
    fronts = non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1, 3] and fronts[-1].tolist() == [4]
    cd = crowding_distance(F[fronts[0]])
    assert np.isinf(cd).sum() == 2


@pytest.mark.parametrize("bounds", [[[0, 0]], [[1, 0]], [[0, np.inf]], [0, 1]])
def test_bad_bounds(bounds):
    with pytest.raises(ValueError):
        check_bounds(bounds)
