"""E6: the averaged buck plant against its analytic steady state and against solve_ivp."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from driveflow.sim.buck_system import BuckParams, BuckState, BuckSystem, continuous_matrices

pytestmark = pytest.mark.simulation


def _run(system, duty, n):
    system.reset()
    return np.array([[s.i_l, s.v_out] for s in (system.simulate(duty) for _ in range(n))])


@pytest.mark.parametrize("duty", [0.25, 0.5, 0.8])
def test_steady_state_is_d_times_vin_without_winding_resistance(duty):
    p = BuckParams(r_l_ohm=0.0)
    x = _run(BuckSystem(p), duty, 40_000)  # 0.8 s, many times the 1/(zeta*w0) decay time
    assert x[-1, 1] == pytest.approx(duty * p.v_in_v, rel=1e-9)
    assert x[-1, 0] == pytest.approx(duty * p.v_in_v / p.load_resistance_ohm, rel=1e-9)


def test_steady_state_with_winding_resistance():
    p = BuckParams()
    x = _run(BuckSystem(p), 0.5, 40_000)
    i_ss, v_ss = p.steady_state(0.5)
    assert x[-1, 1] == pytest.approx(v_ss, rel=1e-9) and x[-1, 0] == pytest.approx(i_ss, rel=1e-9)


def test_transient_matches_solve_ivp():
    """Duty step 0 -> 0.5 from rest, 20 ms (14 L-C periods): measured max errors 3.0e-10 V and
    3.2e-10 A (docstring of sim/buck_system.py); the bound leaves a margin for other BLAS builds."""
    p = BuckParams()
    system = BuckSystem(p)
    n = 1000
    x = _run(system, 0.5, n)
    a, b = continuous_matrices(p, p.load_resistance_ohm)
    t = np.arange(1, n + 1) * system.tau
    sol = solve_ivp(lambda _t, y: a @ y + b * 0.5, (0.0, t[-1]), [0.0, 0.0], t_eval=t, method="Radau", rtol=1e-10, atol=1e-12)
    err = np.abs(sol.y.T - x).max(axis=0)
    assert err[0] < 1e-8 and err[1] < 1e-8, err


def test_load_step_rediscretizes():
    p = BuckParams()
    system = BuckSystem(p)
    system.reset(*reversed(p.steady_state(0.5)))
    system.set_load(3.0)
    for _ in range(40_000):
        system.simulate(0.5)
    i_ss, v_ss = p.steady_state(0.5, 3.0)
    assert system.state.v_out == pytest.approx(v_ss, rel=1e-9) and system.state.i_l == pytest.approx(i_ss, rel=1e-9)


def test_duty_is_clipped_and_bad_parameters_rejected():
    a, b = BuckSystem(), BuckSystem()
    a.reset(), b.reset()
    assert a.simulate(1.7) == b.simulate(1.0)
    assert a.simulate(-1.0) == b.simulate(0.0)
    assert isinstance(a.state, BuckState)
    for bad in ({"l_h": 0.0}, {"c_f": -1e-6}, {"r_l_ohm": -0.1}, {"v_in_v": float("nan")}):
        with pytest.raises(ValueError):
            BuckParams(**bad)
    with pytest.raises(ValueError):
        BuckSystem().set_load(0.0)
