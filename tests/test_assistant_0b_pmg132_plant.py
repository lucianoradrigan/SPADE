"""Round Assistant 0b, D3 (user decision 2026-11-02): plant dc_pmg132_v1, the DC motor with the
Perm-Motor PMG 132 datasheet values at 60 V (S-20). dc_perm_ex_v1 (GEM's port) is unchanged."""

import math

import numpy as np
import pytest

from driveflow.control.design_check import controller_design_check
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import PMG132_LIMITS, PMG132_MOTOR_PARAMETER, PMG132_NOMINAL_VALUES, build_plant

OMEGA_RATED = 2870 * 2 * math.pi / 60


def test_the_plant_carries_the_datasheet_values():
    p = build_plant(Scenario("p", plant_config_id="dc_pmg132_v1"))
    m = p.electrical_motor
    assert m.motor_parameter == PMG132_MOTOR_PARAMETER and m.motor_parameter["psi_e"] == 0.181
    assert m.nominal_values["i"] == 110.0 and m.nominal_values["torque"] == 20.0 and m.nominal_values["omega"] == pytest.approx(OMEGA_RATED)
    assert m.limits["i"] == 200.0 and m.limits["torque"] == 38.0
    load = p.mechanical_load.base_load.load_parameter
    assert load["a"] + load["b"] * OMEGA_RATED == pytest.approx(0.02 * 20.0)  # D1 rule
    assert p.supply.u_nominal == 65.0
    assert PMG132_NOMINAL_VALUES["u"] == PMG132_LIMITS["u"] == 60.0


@pytest.mark.parametrize("controller", ["PI", "MPC"])
def test_rated_load_at_rated_speed(controller):
    """Rated torque (20 N*m) at 2870 min^-1: current about 110 A plus the 2 % friction, voltage below the 65 V supply."""
    recs = run_scenario(Scenario("p", plant_config_id="dc_pmg132_v1", controller_type=controller, omega_ref_rad_s=OMEGA_RATED, load_torque_nm=20.0, duration_s=1.5))
    omega = np.array([r["rpm"] for r in recs]) * 2 * math.pi / 60
    current = np.mean([r["current_r"] for r in recs[-2000:]])
    assert abs(omega[-1] - OMEGA_RATED) < 0.002 * OMEGA_RATED
    assert current == pytest.approx((0.4 + 20.0) / 0.181, rel=5e-3)
    assert recs[0]["plant_config_id"] == "dc_pmg132_v1"


def test_overrides_merge_over_the_datasheet_values():
    p = build_plant(Scenario("p", plant_config_id="dc_pmg132_v1", motor_parameter_overrides={"psi_e": 0.194}, dc_limit_overrides={"i": 150.0}, load_torque_nm=5.0))
    assert p.electrical_motor.motor_parameter["psi_e"] == 0.194 and p.electrical_motor.motor_parameter["r_a"] == 0.016
    assert p.electrical_motor.limits["i"] == 150.0 and p.electrical_motor.limits["torque"] == 38.0
    assert p.mechanical_load.base_load.load_parameter["a"] == pytest.approx(0.2 + 5.0)


def test_the_gem_plant_is_unchanged():
    p = build_plant(Scenario("s"))
    assert p.electrical_motor.motor_parameter["psi_e"] == 0.165 and p.electrical_motor.nominal_values["i"] == 97
    assert p.mechanical_load.base_load.load_parameter == dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)


def test_design_check_and_plant_rules():
    p = build_plant(Scenario("p", plant_config_id="dc_pmg132_v1"))
    assert controller_design_check(p.electrical_motor.motor_parameter, p.mechanical_load.j_total, p.tau)["status"] == "ok"
    with pytest.raises(NotImplementedError):
        Scenario("p", plant_config_id="dc_pmg132_v1", controller_type="DPC")
    with pytest.raises(ValueError, match="pmsm_v1"):
        Scenario("p", plant_config_id="dc_pmg132_v1", pmsm_motor_parameters={"p": 4})
