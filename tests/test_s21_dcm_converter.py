"""S21-2: optional discontinuous-conduction converter model (Scenario.dc_converter_model).

The default plant is byte-identical; "dcm" never carries a negative current and removes the S-21
speed rise; with one substep and no zero crossing it is the default plant. The convergence and the
control are measured in docs/audit/assistant/s21_2_dcm_convergence.md (pre-registered)."""

import numpy as np
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import DCM_DEFAULT_SUBSTEPS, build_plant
from driveflow.sim.dc_dcm import DcmDcMotorSystem

KEYS = ("rpm", "current_r", "voltage_v", "torque_nm", "acc_x", "acc_y", "acc_z")


def rec(**kw):
    r = run_scenario(Scenario("t", **kw))
    return {k: np.array([x[k] for x in r]) for k in KEYS}


def test_explicit_gem_is_byte_identical_to_the_default():
    for kw in (dict(omega_ref_rad_s=120.0, duration_s=0.1, fault_type="outer_race"), dict(plant_config_id="dc_pmg132_v1", omega_ref_rad_s=120.0, duration_s=0.1)):
        a, b = rec(**kw), rec(dc_converter_model="gem", **kw)
        assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in KEYS)


def test_dcm_removes_the_s21_speed_rise_and_never_carries_negative_current():
    g = rec(plant_config_id="dc_pmg132_v1", omega_ref_rad_s=120.0, duration_s=0.5)
    d = rec(plant_config_id="dc_pmg132_v1", omega_ref_rad_s=120.0, duration_s=0.5, dc_converter_model="dcm", dc_plant_substeps=20)
    w = lambda r: r["rpm"] * 2 * np.pi / 60  # noqa: E731
    assert g["current_r"].min() < -50 and w(g)[-1] > 150  # the artifact
    assert d["current_r"].min() >= 0.0 and abs(w(d)[-1] - 120.0) < 0.5


def test_one_substep_without_zero_crossing_is_the_default_plant():
    kw = dict(omega_ref_rad_s=150.0, duration_s=0.2, load_torque_nm=5.0)
    a, b = rec(**kw), rec(dc_converter_model="dcm", dc_plant_substeps=1, **kw)
    for k in ("rpm", "current_r", "torque_nm"):
        assert np.max(np.abs(a[k] - b[k])) <= 1e-8 * max(1.0, np.max(np.abs(a[k])))


def test_build_plant_and_defaults():
    assert type(build_plant(Scenario("t"))).__name__ == "DcMotorSystem"
    p = build_plant(Scenario("t", dc_converter_model="dcm"))
    assert isinstance(p, DcmDcMotorSystem) and p.substeps == DCM_DEFAULT_SUBSTEPS
    assert build_plant(Scenario("t", dc_converter_model="dcm", dc_plant_substeps=7)).substeps == 7


@pytest.mark.parametrize("kw, message", [
    (dict(dc_converter_model="spice"), "must be one of"),
    (dict(dc_converter_model="dcm", dc_plant_solver="rk45"), "rk45"),
    (dict(dc_converter_model="dcm", plant_config_id="pmsm_v1", controller_type="FOC_MTPA", pmsm_current_ref_a=1.0), "DC motor plants"),
])
def test_validation(kw, message):
    with pytest.raises(ValueError, match=message):
        Scenario("t", **kw)


def test_catalog_lists_the_field():
    from driveflow.datagen.plant_catalog import describe_plant

    f = {e["name"]: e for e in describe_plant("dc_perm_ex_v1")["fields"]}["dc_converter_model"]
    assert f["allowed"] == ["gem", "dcm"] and f["default"] is None
