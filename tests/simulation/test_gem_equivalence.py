"""Equivalence of driveflow.sim with gym-electric-motor 3.0.3 beyond test_scml_system_regression.py
-- TODO_CLAUDE_CODE.md V4.

The reference trajectories in tests/data/ref_data_v4.npz were generated once against the unmodified
GEM 3.0.3 clone by tests/data/gen_ref_data_v4.py (see its docstring for the plants and inputs).
This file never imports gym_electric_motor.

- Same plant, same solver: the port must reproduce GEM's trajectory to rounding (Euler) or to the
  solver tolerance (solve_ivp, whose scipy version may differ between the two environments).
- Different solvers: GEM's RK45 run (rtol = atol = 1e-10) is the reference truth, and the errors are
  measured, not assumed.
"""

from pathlib import Path

import numpy as np
import pytest

from driveflow.control.classical.pmsm_foc import default_ode_solver
from driveflow.sim import (
    ConstantSpeedLoad,
    ContB6BridgeConverter,
    ContTwoQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PermanentMagnetSynchronousMotor,
    PolynomialStaticLoad,
    ScipySolveIvpSolver,
    SynchronousMotorSystem,
)

pytestmark = pytest.mark.simulation

REF = Path(__file__).resolve().parents[1] / "data" / "ref_data_v4.npz"
IVP = dict(method="RK45", rtol=1e-10, atol=1e-10)


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip(f"{REF} not found -- regenerate with tests/data/gen_ref_data_v4.py")
    return np.load(REF)


def _dc(solver):
    return DcMotorSystem(converter=ContTwoQuadrantConverter(), motor=DcPermanentlyExcitedMotor(),
                         load=PolynomialStaticLoad(load_parameter=dict(a=0.05, b=0.02, c=0.001, j_load=0.01)),
                         supply=IdealVoltageSupply(u_nominal=420.0), ode_solver=solver, tau=1e-4)


def _pmsm(solver, omega=400.0):
    motor = PermanentMagnetSynchronousMotor(motor_parameter=dict(p=3, l_d=0.37e-3, l_q=1.2e-3, r_s=18e-3, psi_p=66e-3), nominal_values=dict(omega=4e3 * np.pi / 30))
    return SynchronousMotorSystem(converter=ContB6BridgeConverter(), motor=motor, load=ConstantSpeedLoad(omega_fixed=omega),
                                  supply=IdealVoltageSupply(u_nominal=330.0), ode_solver=solver, tau=1e-4, control_space="dq")


def _run(system, actions):
    system.seed(np.random.SeedSequence(4321))
    states = [system.reset()]
    with np.errstate(over="ignore", invalid="ignore"):
        for a in actions:
            states.append(system.simulate(a))
    return np.array(states), list(system.state_names)


@pytest.mark.parametrize("case, build, solver, rtol", [
    ("dc_euler", _dc, EulerSolver, 1e-12),
    ("dc_solve_ivp", _dc, lambda: ScipySolveIvpSolver(**IVP), 1e-8),
    ("pmsm_euler1", _pmsm, EulerSolver, 1e-12),
    ("pmsm_solve_ivp", _pmsm, lambda: ScipySolveIvpSolver(**IVP), 1e-8),
])
def test_same_solver_reproduces_gem(ref, case, build, solver, rtol):
    # Measured 2026-10-20: identical to the last bit in all four cases (same scipy minor version).
    states, names = _run(build(solver()), ref[f"{case}_actions"])
    assert names == [str(n) for n in ref[f"{case}_state_names"]]
    expected = ref[f"{case}_states"]
    if case == "pmsm_euler1":  # diverges by design (S-01): compare while the values are still moderate
        keep = np.all(np.abs(expected) < 1e3, axis=1)
        states, expected = states[keep], expected[keep]
    np.testing.assert_allclose(states, expected, rtol=rtol, atol=1e-12)


def test_dc_euler_against_gem_rk45(ref):
    # Measured 2026-10-20: max error 0.19 % of the peak speed and 2.6 % of the peak current (the
    # current error sits on the four voltage steps).
    states, names = _run(_dc(EulerSolver()), ref["dc_euler_actions"])
    truth = ref["dc_solve_ivp_states"]
    for state, bound in (("omega", 0.005), ("i", 0.03)):
        j = names.index(state)
        assert np.max(np.abs(states[:, j] - truth[:, j])) / np.max(np.abs(truth[:, j])) < bound, state


def _i_s(states, names):
    return np.hypot(states[:, names.index("i_sd")], states[:, names.index("i_sq")])


def test_gem_single_step_euler_diverges_at_400_rad_s_where_rk45_does_not(ref):
    # S-01, checked with GEM's own code: 1 Euler step per 1e-4 s is unstable above 265.8 rad/s.
    names = [str(n) for n in ref["pmsm_euler1_state_names"]]
    euler, truth = _i_s(ref["pmsm_euler1_states"], names), _i_s(ref["pmsm_solve_ivp_states"], names)
    assert np.max(euler) > 100 * np.max(truth)


def test_default_pmsm_solver_stays_bounded_at_400_rad_s(ref):
    states, names = _run(_pmsm(default_ode_solver()), ref["pmsm_euler1_actions"])
    truth = ref["pmsm_solve_ivp_states"]
    assert np.all(np.isfinite(states)) and np.max(_i_s(states, names)) < 2 * np.max(_i_s(truth, names))


def test_default_pmsm_solver_is_close_to_rk45_at_nominal_speed():
    """S-13: with 3 Euler substeps the open-loop RMS i_s error against RK45 was 24.6 % at
    314 rad/s; the default since 2026-10-20 (EULER_NSTEPS = 10) measures 5.3 %."""
    actions = np.repeat(np.array([(-0.05, 0.10), (-0.15, 0.30), (0.0, -0.20), (-0.30, 0.05)]), 500, axis=0)
    est, names = _run(_pmsm(default_ode_solver(), omega=314.0), actions)
    tru, _ = _run(_pmsm(ScipySolveIvpSolver(**IVP), omega=314.0), actions)
    a, b = names.index("i_sd"), names.index("i_sq")
    err = np.hypot(est[:, a] - tru[:, a], est[:, b] - tru[:, b])
    assert np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(_i_s(tru, names) ** 2)) < 0.06
