"""Regression test for the sim/ port: A.2 criterion of INSTRUCTIONS.md.

Verifies that ``driveflow.sim``'s SCMLSystem produces the exact same trajectory as the
unmodified ``gym-electric-motor`` package, for an identical plant configuration, action
sequence and seed. The reference trajectory was generated once (with a fixed seed) against
the pristine GEM clone by ``tests/data/gen_ref_data.py`` and stored in ``tests/data/ref_data.npz``
-- this test never imports ``gym_electric_motor`` itself, only the recorded reference.
"""

from pathlib import Path

import numpy as np
import pytest

from driveflow.sim import (
    ContOneQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PolynomialStaticLoad,
)

REF_DATA_PATH = Path(__file__).parent / "data" / "ref_data.npz"


def build_system(tau):
    converter = ContOneQuadrantConverter()
    motor = DcPermanentlyExcitedMotor()
    load = PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025))
    supply = IdealVoltageSupply(u_nominal=560.0)
    solver = EulerSolver()
    return DcMotorSystem(converter=converter, motor=motor, load=load, supply=supply, ode_solver=solver, tau=tau)


@pytest.fixture(scope="module")
def ref_data():
    if not REF_DATA_PATH.exists():
        pytest.skip(
            f"{REF_DATA_PATH} not found. Regenerate it against the unmodified GEM clone with "
            "tests/data/gen_ref_data.py before running this regression test."
        )
    return np.load(REF_DATA_PATH)


def test_state_names_match_gem(ref_data):
    system = build_system(tau=float(ref_data["tau"]))
    assert list(system.state_names) == list(ref_data["state_names"])


def test_limits_match_gem(ref_data):
    system = build_system(tau=float(ref_data["tau"]))
    np.testing.assert_allclose(system.limits, ref_data["limits"])


def test_trajectory_matches_gem(ref_data):
    """The core A.2 regression check: same seed, same actions -> same state trajectory."""
    tau = float(ref_data["tau"])
    seed = int(ref_data["seed"])
    actions = ref_data["actions"]
    expected_states = ref_data["states"]

    system = build_system(tau=tau)
    system.seed(np.random.SeedSequence(seed))
    initial_state = system.reset()

    states = np.zeros_like(expected_states)
    states[0] = initial_state
    for k, action in enumerate(actions):
        states[k + 1] = system.simulate(np.array([action]))

    np.testing.assert_allclose(states, expected_states, rtol=1e-10, atol=1e-12)


def test_reset_then_simulate_is_deterministic_for_fixed_seed(ref_data):
    """A second reset+run with the same seed must reproduce the same trajectory (used by
    datagen/runner.py in a later phase to make simulated runs reproducible)."""
    tau = float(ref_data["tau"])
    seed = int(ref_data["seed"])
    actions = ref_data["actions"]

    def run_once():
        system = build_system(tau=tau)
        system.seed(np.random.SeedSequence(seed))
        state = system.reset()
        trajectory = [state]
        for action in actions:
            trajectory.append(system.simulate(np.array([action])))
        return np.array(trajectory)

    first_run = run_once()
    second_run = run_once()
    np.testing.assert_array_equal(first_run, second_run)
