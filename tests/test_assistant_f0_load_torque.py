"""Round Assistant, task F0.1: Scenario.load_torque_nm, a constant passive load torque on the DC
motor (dc_perm_ex_v1). The default (None) must leave every existing result bit for bit; the
existing datagen and tests/test_scml_system_regression.py tests check that and are not modified."""

import math

import numpy as np
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import build_plant

PSI_E = 0.165  # DcPermanentlyExcitedMotor stock psi_e (V*s/rad = N*m/A)
A_STOCK, B_STOCK = 0.01, 0.05  # runner.build_plant's PolynomialStaticLoad a (N*m), b (N*m*s/rad)


def _arrays(records, *keys):
    return [np.array([r[k] for r in records], dtype=float) for k in keys]


def test_none_and_zero_load_build_the_stock_load():
    stock = build_plant(Scenario("s")).mechanical_load
    zero = build_plant(Scenario("s", load_torque_nm=0.0)).mechanical_load
    loaded = build_plant(Scenario("s", load_torque_nm=12.5)).mechanical_load
    base = lambda load: load.base_load  # noqa: E731 -- BearingFaultLoad wraps the PolynomialStaticLoad
    assert base(stock).load_parameter == dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)
    assert base(zero).load_parameter == base(stock).load_parameter
    assert base(loaded).load_parameter["a"] == pytest.approx(12.51)


def test_zero_load_is_bit_identical_to_the_default():
    a = run_scenario(Scenario("s", duration_s=0.2, fault_type="outer_race"))
    b = run_scenario(Scenario("s", duration_s=0.2, fault_type="outer_race", load_torque_nm=0.0))
    for key in ("current_r", "rpm", "torque_nm", "voltage_v", "acc_x"):
        assert np.array_equal(*[np.array([r[key] for r in recs]) for recs in (a, b)]), key


@pytest.mark.parametrize("load_nm", [4.0, 12.0])
def test_steady_state_current_balances_the_load(load_nm):
    """Speed PI at 150 rad/s: in steady state psi_e * i = a + b*omega + load (the current stays
    below the PI's 168 A cap at these loads). Declared tolerance: 0.1 % of the predicted current."""
    rpm, current = _arrays(run_scenario(Scenario("s", omega_ref_rad_s=150.0, load_torque_nm=load_nm, duration_s=1.2)), "rpm", "current_r")
    omega = rpm[-2000:] * 2 * math.pi / 60
    predicted = (A_STOCK + B_STOCK * omega.mean() + load_nm) / PSI_E
    assert abs(omega[-1] - 150.0) < 0.01
    assert current[-2000:].mean() == pytest.approx(predicted, rel=1e-3)


def test_the_load_also_applies_in_torque_control():
    """torque_ref_nm = load + friction at standstill is not enough to accelerate past the friction:
    a larger load torque gives a lower speed for the same torque reference."""
    free, = _arrays(run_scenario(Scenario("s", torque_ref_nm=10.0, duration_s=0.3)), "rpm")
    loaded, = _arrays(run_scenario(Scenario("s", torque_ref_nm=10.0, load_torque_nm=6.0, duration_s=0.3)), "rpm")
    assert loaded[-1] < free[-1]


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, True, "5"])
def test_invalid_values_are_rejected(value):
    with pytest.raises(ValueError, match="load_torque_nm"):
        Scenario("s", load_torque_nm=value)


@pytest.mark.parametrize("kwargs", [
    dict(controller_type="DPC", plant_config_id="vsc_dpc_v1"),
    dict(controller_type="FOC_MTPA", plant_config_id="pmsm_v1", pmsm_current_ref_a=50.0),
    dict(controller_type="PI", plant_config_id="buck_v1"),
])
def test_other_plants_reject_the_field(kwargs):
    with pytest.raises(ValueError, match="applies only to plant_config_id='dc_perm_ex_v1'"):
        Scenario("s", load_torque_nm=5.0, **kwargs)
