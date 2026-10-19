"""Numerical stability of the PMSM / FOC model -- docs/SIMULATION_STABILITY_REPORT.md Sec. 3.

Key result (S-01): with forward Euler at tau = 1e-4 s the open-loop dq electrical dynamics leave
the method's stability region above ~266 rad/s mechanical -- inside the motor's own nominal speed
(314 rad/s). The physical system is damped there (RK45 converges to the analytic short-circuit
current); the Euler trajectory grows without bound. A numerical, not a physical, instability.
"""

import numpy as np
import pytest

from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS, generate_mtpa_vs_naive_cloud, mtpa_id_iq
from driveflow.sim.solvers import ScipySolveIvpSolver
from driveflow.validation import euler_max_stable_speed_pmsm, euler_spectral_radius, pmsm_dq_matrix, run_pmsm_current_step, run_pmsm_open_loop
from driveflow.validation.benches import pmsm_short_circuit_current

pytestmark = pytest.mark.simulation

TAU = 1e-4
NOMINAL_SPEED = 3e3 * np.pi / 30  # PermanentMagnetSynchronousMotor._default_nominal_values["omega"]


def _rk45():
    return ScipySolveIvpSolver(method="RK45", rtol=1e-8, atol=1e-8)


def _tail_magnitude(r, n=500):
    mag = np.hypot(r["i_d"], r["i_q"])[-n:]
    return mag.min(), mag.max()


def test_physical_dq_dynamics_are_damped_at_every_speed():
    for w in (0.0, 100.0, 300.0, NOMINAL_SPEED):
        a = pmsm_dq_matrix(w * MOTOR_PARAMS["p"], MOTOR_PARAMS["r_s"], MOTOR_PARAMS["l_d"], MOTOR_PARAMS["l_q"])
        assert np.all(np.linalg.eigvals(a).real < 0)


def test_euler_stability_limit_is_inside_the_nominal_speed_range():
    w_max = euler_max_stable_speed_pmsm(MOTOR_PARAMS, TAU)
    assert 250.0 < w_max < NOMINAL_SPEED
    a = pmsm_dq_matrix(1.05 * w_max * MOTOR_PARAMS["p"], MOTOR_PARAMS["r_s"], MOTOR_PARAMS["l_d"], MOTOR_PARAMS["l_q"])
    assert euler_spectral_radius(a, TAU) > 1.0


def test_rk45_reaches_the_analytic_short_circuit_current_at_300_rad_s():
    i_d, i_q = pmsm_short_circuit_current(300.0)
    lo, hi = _tail_magnitude(run_pmsm_open_loop(300.0, ode_solver=_rk45()))
    assert lo == pytest.approx(np.hypot(i_d, i_q), rel=2e-3) and hi == pytest.approx(np.hypot(i_d, i_q), rel=2e-3)


def test_euler_diverges_at_300_rad_s_although_the_physics_is_stable():
    lo, hi = _tail_magnitude(run_pmsm_open_loop(300.0))
    analytic = np.hypot(*pmsm_short_circuit_current(300.0))
    assert hi > 5 * analytic  # measured ~5700 A vs 178 A analytic


def test_euler_is_accurate_well_inside_its_stability_region():
    """At 100 rad/s (rho(I + tau A) well below 1) Euler and the analytic value agree to 1 %."""
    lo, hi = _tail_magnitude(run_pmsm_open_loop(100.0, duration_s=0.6))
    analytic = np.hypot(*pmsm_short_circuit_current(100.0))
    assert lo == pytest.approx(analytic, rel=0.01) and hi == pytest.approx(analytic, rel=0.01)


def test_closed_loop_current_step_euler_matches_rk45():
    """The demo regime (rotor held near standstill by LOAD_INERTIA): closed-loop Euler vs RK45
    differ by < 0.5 % of the commanded current magnitude over the whole transient."""
    i_s = 200.0
    i_d, i_q = mtpa_id_iq(i_s)
    e = run_pmsm_current_step(i_d, i_q)
    k = run_pmsm_current_step(i_d, i_q, ode_solver=_rk45())
    assert np.max(np.abs(e["i_d"] - k["i_d"])) < 0.005 * i_s
    assert np.max(np.abs(e["i_q"] - k["i_q"])) < 0.005 * i_s


def test_mtpa_cloud_is_reproducible_and_finite():
    a = generate_mtpa_vs_naive_cloud(i_s_values=[100, 300], duration_s=0.01)
    b = generate_mtpa_vs_naive_cloud(i_s_values=[100, 300], duration_s=0.01)
    assert a["mtpa"] == b["mtpa"] and a["naive"] == b["naive"]
    assert np.all(np.isfinite(np.asarray(a["mtpa"])))
