"""E7 follow-up (2026-10-21): J_effort, the peak-current constraint, the noisy evaluation grid and the
optional speed-sensor noise of the closed-loop bench."""

import math

import numpy as np
import pytest

from driveflow.jobs.handlers import _validate_tuning
from driveflow.optim import objectives as ob
from driveflow.runs.schemas import ExperimentSpec, SpadeError
from driveflow.validation.benches import run_dc_closed_loop


def test_zero_noise_leaves_the_loop_byte_identical():
    a = run_dc_closed_loop("PI", 120.0, duration_s=0.05)
    b = run_dc_closed_loop("PI", 120.0, duration_s=0.05, speed_noise_std_rad_s=0.0)
    assert np.array_equal(a["omega"], b["omega"]) and np.array_equal(a["u"], b["u"])


def test_noise_reaches_the_controller_only_and_is_seeded():
    a = run_dc_closed_loop("PI", 120.0, duration_s=0.05, speed_noise_std_rad_s=0.6, noise_seed=3)
    b = run_dc_closed_loop("PI", 120.0, duration_s=0.05, speed_noise_std_rad_s=0.6, noise_seed=3)
    c = run_dc_closed_loop("PI", 120.0, duration_s=0.05)
    assert np.array_equal(a["u"], b["u"]) and not np.array_equal(a["u"], c["u"])


def test_baseline_scores_four_and_the_constraint_penalises():
    base = ob.baseline_tuning_summary()
    assert ob.j_effort(base) == pytest.approx(4.0)
    over = dict(base, max_step_peak_i_a=ob.I_NOMINAL_A * 1.1)
    assert ob.j_effort(over) == pytest.approx(4.0 + 1.0)
    assert ob.j_effort(dict(base, finite=False)) == math.inf


def test_noisy_grid_is_the_evaluation_grid_with_noise_and_does_not_overlap_tuning():
    assert [c.name for c in ob.EVALUATION_GRID_NOISY] == [c.name + "_noisy" for c in ob.EVALUATION_GRID]
    assert all(c.noise_std_rad_s == ob.NOISE_STD_RAD_S for c in ob.EVALUATION_GRID_NOISY)
    assert not ob.grids_overlap(ob.TUNING_GRID, ob.EVALUATION_GRID_NOISY)


def test_rated_current_matches_the_motor():
    from driveflow.validation.benches import build_dc_system

    system = build_dc_system(None, 1e-4, None, None, 0.0, np.inf)
    assert system.electrical_motor.nominal_values["i"] == ob.I_NOMINAL_A


def test_unknown_objective_is_refused():
    spec = ExperimentSpec.create("controller_tuning", algorithm="pso", seed=0, params={"bounds": [list(b) for b in ob.BOUNDS], "objective": "nope"}, budget={"evaluations": 8})
    with pytest.raises(SpadeError):
        _validate_tuning(spec)
    _validate_tuning(ExperimentSpec.create("controller_tuning", algorithm="pso", seed=0, params={"bounds": [list(b) for b in ob.BOUNDS], "objective": "J_effort"}, budget={"evaluations": 8}))
