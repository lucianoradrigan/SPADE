"""One-off script that generates tests/data/ref_data_v4.npz against the UNMODIFIED gym-electric-motor
clone (v3.0.3, ../../gym-electric-motor relative to this repo) -- TODO_CLAUDE_CODE.md V4.

Same pattern as gen_ref_data.py: not part of the driveflow package, not run by the test suite, and
GEM is not a driveflow dependency. tests/simulation/test_gem_equivalence.py only reads the .npz.

Cases (2000 steps of tau = 1e-4 s each unless noted):
- dc_euler / dc_solve_ivp: permanently excited DC motor, two-quadrant converter, a polynomial load
  different from gen_ref_data.py's, 420 V supply, and four voltage steps (+0.3, +0.8, -0.4, +0.6
  duty), integrated with GEM's EulerSolver and with ScipySolveIvpSolver (RK45, rtol = atol = 1e-10).
- pmsm_euler1 / pmsm_solve_ivp: the PMSM of control/classical/pmsm_foc.py (GEM's default
  parameters), B6 bridge, 330 V, dq control space, rotor held at 400 rad/s by ConstantSpeedLoad (GEM's nominal limit is 4000 rpm ~ 419 rad/s), and
  a fixed sequence of dq voltage steps; integrated with GEM's single-step EulerSolver and with
  ScipySolveIvpSolver (RK45, rtol = atol = 1e-10) as the reference truth. 400 rad/s is above the
  single-step Euler limit (265.8 rad/s) and below the 3-substep one (~460 rad/s), so this checks
  S-01 independently of SPADE's code.

Run with (Python 3.11, GEM's own dependencies):

    uv run --python 3.11 --with matplotlib --with "gymnasium>=0.29.1" --with numpy --with scipy \\
        tests/data/gen_ref_data_v4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gym-electric-motor" / "src"))

import numpy as np
import scipy
from gym_electric_motor.physical_systems import converters as cv
from gym_electric_motor.physical_systems import electric_motors as em
from gym_electric_motor.physical_systems import mechanical_loads as ml
from gym_electric_motor.physical_systems import physical_systems as ps
from gym_electric_motor.physical_systems import solvers as sv
from gym_electric_motor.physical_systems import voltage_supplies as vs

SEED = 4321
TAU = 1e-4
N_STEPS = 2000
IVP = dict(method="RK45", rtol=1e-10, atol=1e-10)
DC_LOAD = dict(a=0.05, b=0.02, c=0.001, j_load=0.01)
DC_SUPPLY_V = 420.0
PMSM_PARAMS = dict(p=3, l_d=0.37e-3, l_q=1.2e-3, r_s=18e-3, psi_p=66e-3)
PMSM_SUPPLY_V = 330.0
PMSM_OMEGA = 400.0
#: ConstantSpeedLoad's initial speed must lie inside the motor's *nominal* omega (default 3000 rpm),
#: so the nominal speed is raised to the default limit (4000 rpm); limits, and so the normalization of
#: the recorded states, are unchanged.
PMSM_NOMINAL = dict(omega=4e3 * np.pi / 30)


def dc_actions():
    return np.repeat([0.3, 0.8, -0.4, 0.6], N_STEPS // 4)[:, None]


def pmsm_actions():
    steps = [(-0.05, 0.10), (-0.15, 0.30), (0.0, -0.20), (-0.30, 0.05)]
    return np.repeat(np.array(steps), N_STEPS // 4, axis=0)


def dc_system(solver):
    return ps.DcMotorSystem(converter=cv.ContTwoQuadrantConverter(), motor=em.DcPermanentlyExcitedMotor(),
                            load=ml.PolynomialStaticLoad(load_parameter=DC_LOAD), supply=vs.IdealVoltageSupply(u_nominal=DC_SUPPLY_V),
                            ode_solver=solver, tau=TAU)


def pmsm_system(solver):
    return ps.SynchronousMotorSystem(converter=cv.ContB6BridgeConverter(), motor=em.PermanentMagnetSynchronousMotor(motor_parameter=PMSM_PARAMS, nominal_values=PMSM_NOMINAL),
                                     load=ml.ConstantSpeedLoad(omega_fixed=PMSM_OMEGA), supply=vs.IdealVoltageSupply(u_nominal=PMSM_SUPPLY_V),
                                     ode_solver=solver, tau=TAU, control_space="dq")


def run(system, actions):
    system.seed(np.random.SeedSequence(SEED))
    states = np.zeros((len(actions) + 1, len(system.state_names)))
    states[0] = system.reset()
    with np.errstate(over="ignore", invalid="ignore"):
        for k, a in enumerate(actions):
            states[k + 1] = system.simulate(a)
    return states, np.array(system.state_names), system.limits


def main():
    out = {"seed": SEED, "tau": TAU, "n_steps": N_STEPS, "scipy_version": scipy.__version__, "numpy_version": np.__version__}
    for name, build, actions, solver in (
        ("dc_euler", dc_system, dc_actions(), lambda: sv.EulerSolver()),
        ("dc_solve_ivp", dc_system, dc_actions(), lambda: sv.ScipySolveIvpSolver(**IVP)),
        ("pmsm_euler1", pmsm_system, pmsm_actions(), lambda: sv.EulerSolver()),
        ("pmsm_solve_ivp", pmsm_system, pmsm_actions(), lambda: sv.ScipySolveIvpSolver(**IVP)),
    ):
        states, names, limits = run(build(solver()), actions)
        out[f"{name}_states"], out[f"{name}_actions"] = states, actions
        out[f"{name}_state_names"], out[f"{name}_limits"] = names, limits
        print(name, states.shape, "finite" if np.all(np.isfinite(states)) else "NON-FINITE")
    path = Path(__file__).parent / "ref_data_v4.npz"
    np.savez(path, **out)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
