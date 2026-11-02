"""Round Assistant, task F0.6: PI tuning on another DC motor. With the stock motor every number
must stay as before (docs/audit/assistant/f0_6_stock_before.json, recorded before the change)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from driveflow.jobs.handlers import _validate_tuning
from driveflow.optim import objectives as ob
from driveflow.optim.pso import pso
from driveflow.runs.schemas import ExperimentSpec, SpadeError

BEFORE = json.loads((Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f0_6_stock_before.json").read_text())
CASE_B = ob.BaseMotor(motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05), supply_v=220.0,
                      limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
                      nominal_overrides=dict(u=220.0, i=25.0, omega=1500 * 2 * math.pi / 60, torque=1.146 * 25.0))


def test_stock_tuning_grid_is_unchanged():
    summary, rows = ob.evaluate(ob.BASELINE, ob.TUNING_GRID)
    expected = BEFORE["TUNING_GRID"]
    for key, value in expected["summary"].items():
        assert summary[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
    assert [r["condition"] for r in rows] == [r["condition"] for r in expected["results"]]
    assert ob.j_effort(summary) == pytest.approx(BEFORE["j_effort_baseline_on_tuning"], rel=1e-12)


def test_stock_drift_is_the_former_absolute_dict():
    assert ob.STOCK.plant_parameters(ob.DRIFT_FACTORS) == ob.DRIFT
    assert ob.STOCK.bench_kwargs() == {}


def test_drift_is_relative_to_the_base_motor():
    plant = CASE_B.plant_parameters(ob.DRIFT_FACTORS)
    assert plant["psi_e"] == pytest.approx(1.146 * 0.9) and plant["j_rotor"] == pytest.approx(0.05 * 1.5) and plant["r_a"] == 1.6
    assert CASE_B.bench_kwargs()["design_motor_parameter"] == CASE_B.motor_parameter_overrides


def test_short_tuning_on_the_220_v_motor_runs():
    scalar, _vector, _grid, _text, score = ob.objective_functions("J", CASE_B)
    res = pso(scalar, [list(b) for b in ob.BOUNDS], n_particles=2, iterations=1, seed=0)
    assert np.isfinite(res.f_best)
    summary, _ = ob.evaluate(ob.as_kwargs(res.x_best), ob.TUNING_GRID, CASE_B)
    assert summary["finite"] and score(summary) == summary["J"]
    assert summary["current_constraint_ok"] is not None  # rated current 25 A from nominal_overrides


def test_j_effort_without_a_rated_current_is_refused():
    no_rating = ob.BaseMotor(motor_parameter_overrides=dict(CASE_B.motor_parameter_overrides), supply_v=220.0)
    with pytest.raises(ob.MotorSpecError, match="rated current"):
        ob.objective_functions("J_effort", no_rating)
    text = ob.objective_functions("J_effort", CASE_B)[3]
    assert "25 A" in text and "97 A" not in text  # both mentions of the rated current use the base motor
    spec = ExperimentSpec(experiment_id="f06", kind="controller_tuning", algorithm="pso", seed=1, budget={"evaluations": 4},
                          params={"bounds": [list(b) for b in ob.BOUNDS], "objective": "J_effort",
                                  "motor": {"motor_parameter_overrides": dict(CASE_B.motor_parameter_overrides), "dc_supply_v": 220.0}})
    with pytest.raises(SpadeError, match="rated current"):
        _validate_tuning(spec)


@pytest.mark.parametrize("block, match", [
    ({"dc_supply_v": -5.0}, "dc_supply_v"),
    ({"voltage": 220.0}, "unknown key"),
    ({"motor_parameter_overrides": {"kt": 1.0}}, "unknown parameter"),
])
def test_the_motor_block_is_validated_like_a_scenario(block, match):
    with pytest.raises((ValueError, ob.MotorSpecError), match=match):
        ob.motor_from_spec(block)


def test_grids_are_scaled_to_the_base_motor_ratings():
    """0b D2: speeds, steps and noise scale with omega_rated / 300, loads with torque_rated / 16;
    the stock motor keeps the module's own grids (same objects)."""
    assert ob.grids_for(ob.STOCK) == (ob.TUNING_GRID, ob.EVALUATION_GRID, ob.EVALUATION_GRID_NOISY)
    assert ob.grids_for(ob.STOCK)[0] is ob.TUNING_GRID
    tuning, evaluation, noisy = ob.grids_for(CASE_B)
    ks, kt = CASE_B.nominal_overrides["omega"] / 300.0, CASE_B.nominal_overrides["torque"] / 16.0
    for scaled, stock in zip(evaluation + noisy, ob.EVALUATION_GRID + ob.EVALUATION_GRID_NOISY):
        assert scaled.omega_rad_s == pytest.approx(stock.omega_rad_s * ks) and scaled.step_rad_s == pytest.approx(stock.step_rad_s * ks)
        assert scaled.load_nm == pytest.approx(stock.load_nm * kt) and scaled.noise_std_rad_s == pytest.approx(stock.noise_std_rad_s * ks)
        assert scaled.drift == stock.drift and scaled.name == stock.name + "_scaled"
    assert max(c.omega_rad_s for c in evaluation) <= CASE_B.nominal_overrides["omega"] * 200.0 / 300.0 + 1e-9


def test_another_motor_without_rated_speed_and_torque_cannot_be_tuned():
    only_current = ob.BaseMotor(motor_parameter_overrides=dict(CASE_B.motor_parameter_overrides), supply_v=220.0, nominal_overrides={"i": 25.0})
    with pytest.raises(ob.MotorSpecError, match="rated speed and torque"):
        ob.objective_functions("J", only_current)
