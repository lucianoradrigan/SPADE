"""Round Assistant, task F0.2: DC supply voltage, motor limits and nominal values, and the
controller-design check. Defaults must stay bit for bit (existing regression tests unchanged)."""

import math

import numpy as np
import pytest

from driveflow.control.design_check import controller_design_check, mpc_design_check
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import build_plant

#: Case B of docs/assistant/PLAN_AGENTE_ASISTENTE_2026-11-02.md: 220 V, 25 A, 1500 rpm, 4.5 kW,
#: with the parameters the round instructions derive (R_a 1.6 ohm, L_a 16 mH, psi_e 1.146 V*s,
#: J 0.05 kg*m^2). Limits: twice the rated current, the supply voltage.
OMEGA_RATED = 1500 * 2 * math.pi / 60
CASE_B = dict(
    motor_parameter_overrides=dict(r_a=1.6, l_a=16e-3, psi_e=1.146, j_rotor=0.05),
    dc_supply_v=220.0,
    dc_limit_overrides=dict(u=220.0, i=50.0, omega=200.0, torque=60.0),
    dc_nominal_overrides=dict(u=220.0, i=25.0, omega=OMEGA_RATED, torque=1.146 * 25.0),
)
KEYS = ("current_r", "rpm", "torque_nm", "voltage_v", "acc_x")


def _records(**kwargs):
    return run_scenario(Scenario("s", duration_s=0.2, fault_type="inner_race", **kwargs))


def _same(a, b):
    return all(np.array_equal(np.array([r[k] for r in a]), np.array([r[k] for r in b])) for k in KEYS)


@pytest.mark.parametrize("controller", ["PI", "MPC"])
def test_explicit_stock_values_are_bit_identical_to_the_default(controller):
    default = _records(controller_type=controller)
    assert _same(default, _records(controller_type=controller, dc_supply_v=65.0))
    assert _same(default, _records(controller_type=controller, dc_limit_overrides=dict(omega=400, torque=38.0, i=210, u=60),
                                   dc_nominal_overrides=dict(omega=300, torque=16.0, i=97, u=60)))


def test_overrides_reach_the_plant():
    plant = build_plant(Scenario("s", **CASE_B))
    assert plant.supply.u_nominal == 220.0
    assert plant.electrical_motor.limits["i"] == 50.0 and plant.electrical_motor.limits["u"] == 220.0
    assert plant.electrical_motor.nominal_values["i"] == 25.0


def test_an_explicit_current_limit_is_kept_and_parameters_alone_keep_the_stock_limit():
    """GEM derives i = u/r_a only for a missing or zero limit, so an explicit i is respected, and
    changing r_a without limits keeps the stock 210 A (documented in the Scenario docstring)."""
    assert build_plant(Scenario("s", dc_limit_overrides={"i": 150.0})).electrical_motor.limits["i"] == 150.0
    assert build_plant(Scenario("s", motor_parameter_overrides={"r_a": 1.6})).electrical_motor.limits["i"] == 210


@pytest.mark.parametrize("controller", ["PI", "MPC"])
def test_the_220_v_motor_reaches_its_rated_speed(controller):
    recs = run_scenario(Scenario("b", controller_type=controller, omega_ref_rad_s=OMEGA_RATED, duration_s=1.0, **CASE_B))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    assert np.all(np.isfinite(omega))
    assert abs(omega[-1] - OMEGA_RATED) < 0.005 * OMEGA_RATED  # MPC has no integrator: small offset
    assert max(r["current_r"] for r in recs) <= 0.86 * 50.0  # current cap from the overridden limit


@pytest.mark.parametrize("kwargs, match", [
    (dict(dc_supply_v=0.0), "dc_supply_v"),
    (dict(dc_supply_v=math.nan), "dc_supply_v"),
    (dict(dc_limit_overrides={"volts": 10.0}), "unknown key"),
    (dict(dc_nominal_overrides={"i": -1.0}), r"dc_nominal_overrides\['i'\]"),
    (dict(dc_limit_overrides={"i": 0.0}), r"dc_limit_overrides\['i'\]"),
    (dict(dc_limit_overrides=[("i", 5.0)]), "must be a dict"),
    (dict(dc_supply_v=48.0, dc_limit_overrides={"u": 60.0}), "must not exceed dc_supply_v"),
])
def test_invalid_values_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Scenario("s", **kwargs)


def test_other_plants_reject_the_fields():
    for field, value in (("dc_supply_v", 48.0), ("dc_limit_overrides", {"i": 10.0}), ("dc_nominal_overrides", {"i": 10.0})):
        with pytest.raises(ValueError, match="applies only to plant_config_id='dc_perm_ex_v1'"):
            Scenario("s", controller_type="DPC", plant_config_id="vsc_dpc_v1", **{field: value})


def test_design_check_ok_on_the_stock_motor_and_warning_when_the_time_constants_are_close():
    stock = build_plant(Scenario("s"))
    mp, j = stock.electrical_motor.motor_parameter, stock.mechanical_load.j_total
    ok = controller_design_check(mp, j, stock.tau)
    assert ok["status"] == "ok" and ok["tau_m_over_tau_a"] == pytest.approx(13.61, rel=1e-3) and ok["reasons"] == []
    assert mpc_design_check(mp, j, stock.tau)["status"] == "ok"
    # Constructed case: l_a x 2 halves tau_m/tau_a to 6.8 < 10.
    slow = controller_design_check({**mp, "l_a": 2 * mp["l_a"]}, j, stock.tau)
    assert slow["status"] == "warning" and "tau_m/tau_a" in slow["reasons"][0]
    case_b = build_plant(Scenario("b", **CASE_B))
    b = controller_design_check(case_b.electrical_motor.motor_parameter, case_b.mechanical_load.j_total, case_b.tau)
    assert b["status"] == "warning" and b["tau_m_over_tau_a"] == pytest.approx(6.40, rel=1e-2)
    assert mpc_design_check(case_b.electrical_motor.motor_parameter, case_b.mechanical_load.j_total, case_b.tau)["horizon_over_tau_a"] == pytest.approx(0.2)
