"""One-off script that generates tests/data/ref_data.npz against the UNMODIFIED gym-electric-motor
clone (../../gym-electric-motor relative to this repo, i.e. a sibling of driveflow/).

This script is NOT part of the driveflow package and does not run as part of the test suite; it is
kept here only so the reference trajectory used by test_scml_system_regression.py is reproducible.
It needs gym_electric_motor's own dependencies (gymnasium, numpy, scipy, matplotlib), which are
deliberately NOT part of driveflow's own dependencies -- run it with e.g.:

    uv run --python 3.11 --with matplotlib --with "gymnasium>=0.29.1" --with numpy --with scipy \\
        tests/data/gen_ref_data.py

Re-run only if the reference plant configuration below changes; the regression test itself must
keep an identical configuration in driveflow's own ported classes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gym-electric-motor" / "src"))

import numpy as np
from gym_electric_motor.physical_systems import converters as cv
from gym_electric_motor.physical_systems import electric_motors as em
from gym_electric_motor.physical_systems import mechanical_loads as ml
from gym_electric_motor.physical_systems import physical_systems as ps
from gym_electric_motor.physical_systems import solvers as sv
from gym_electric_motor.physical_systems import voltage_supplies as vs

SEED = 1234
N_STEPS = 200
TAU = 1e-4


def build_system():
    converter = cv.ContOneQuadrantConverter()
    motor = em.DcPermanentlyExcitedMotor()
    load = ml.PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025))
    supply = vs.IdealVoltageSupply(u_nominal=560.0)
    solver = sv.EulerSolver()
    return ps.DcMotorSystem(converter=converter, motor=motor, load=load, supply=supply, ode_solver=solver, tau=TAU)


def main():
    system = build_system()
    system.seed(np.random.SeedSequence(SEED))
    initial_state = system.reset()

    rng = np.random.default_rng(SEED)
    t = np.arange(N_STEPS) * TAU
    actions = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * 50 * t) + 0.05 * rng.standard_normal(N_STEPS), 0.0, 1.0)

    states = np.zeros((N_STEPS + 1, len(system.state_names)))
    states[0] = initial_state
    for k in range(N_STEPS):
        states[k + 1] = system.simulate(np.array([actions[k]]))

    out_path = Path(__file__).parent / "ref_data.npz"
    np.savez(
        out_path,
        states=states,
        actions=actions,
        state_names=np.array(system.state_names),
        limits=system.limits,
        seed=SEED,
        tau=TAU,
        n_steps=N_STEPS,
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
