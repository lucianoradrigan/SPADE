"""Round Assistant 0b, D1 (user decision 2026-11-02): the default load of a DC motor other than the
stock one is a friction of 2 % of its rated torque at rated speed. The stock motor keeps its load."""

import math

import numpy as np
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import STOCK_DC_LOAD_PARAMETER, build_plant, dc_default_load_parameter
from driveflow.optim import objectives as ob

OMEGA_B = 1500 * 2 * math.pi / 60
CASE_B = dict(motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05), dc_supply_v=220.0,
              dc_limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
              dc_nominal_overrides=dict(u=220.0, i=25.0, omega=OMEGA_B, torque=1.146 * 25.0))


def _load(**kwargs) -> dict:
    return build_plant(Scenario("s", **kwargs)).mechanical_load.base_load.load_parameter


def test_stock_motor_keeps_the_stock_load():
    assert _load() == STOCK_DC_LOAD_PARAMETER
    assert _load(dc_nominal_overrides=dict(torque=16.0, omega=300.0)) == STOCK_DC_LOAD_PARAMETER  # motor unchanged
    assert _load(motor_parameter_overrides={"psi_e": 0.15}) == STOCK_DC_LOAD_PARAMETER  # ratings unknown: stock load


def test_another_motor_with_ratings_gets_two_percent_of_rated_torque():
    load = _load(**CASE_B)
    t_rated = 1.146 * 25.0
    assert load["a"] + load["b"] * OMEGA_B == pytest.approx(0.02 * t_rated)
    assert load["a"] == pytest.approx(0.01 * t_rated) and load["c"] == 0.0 and load["j_load"] == 0.0025
    assert _load(load_torque_nm=5.0, **CASE_B)["a"] == pytest.approx(0.01 * t_rated + 5.0)


def test_tuning_uses_the_same_rule():
    motor = ob.BaseMotor(CASE_B["motor_parameter_overrides"], 220.0, CASE_B["dc_limit_overrides"], CASE_B["dc_nominal_overrides"])
    assert motor.bench_kwargs()["load_parameter"] == dc_default_load_parameter(CASE_B["motor_parameter_overrides"], CASE_B["dc_nominal_overrides"])
    assert "load_parameter" not in ob.STOCK.bench_kwargs()


def test_220_v_motor_now_carries_rated_load_close_to_rated_speed():
    """At rated torque the needed voltage, psi_e*omega + R_a*I = 180.0 + 40.8 V, is 0.4 % above the
    220 V supply (psi_e was derived from P/(omega*I), not from the EMF), so the speed settles within
    1 % of rated instead of 6 % below it with the stock friction."""
    recs = run_scenario(Scenario("b", omega_ref_rad_s=OMEGA_B, load_torque_nm=1.146 * 25.0, duration_s=1.5, **CASE_B))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    assert abs(omega[-1] - OMEGA_B) < 0.01 * OMEGA_B
