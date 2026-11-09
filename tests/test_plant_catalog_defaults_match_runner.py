"""S-23 (2026-11-08): for every native plant, the field defaults that the plant catalog (and so the assistant)
shows must equal the parameters the runner really builds. Compared against the built plant objects, not
against the constants the catalog reads, so a wrong mapping in either place is caught."""

import pytest

from driveflow.datagen.plant_catalog import NATIVE_PLANTS, describe_plant
from driveflow.datagen.runner import build_buck, build_plant
from driveflow.datagen.scenario import Scenario

DC = [p for p in NATIVE_PLANTS if p in Scenario.DC_PLANTS]


def _keys(plant: str, field: str) -> dict:
    f = {x["name"]: x for x in describe_plant(plant)["fields"]}[field]
    return {k: v["default"] for k, v in f["keys"].items()}


@pytest.mark.parametrize("plant", DC)
def test_dc_catalog_defaults_equal_the_built_motor(plant):
    system = build_plant(Scenario("catalog_check", plant_config_id=plant))
    motor = system.electrical_motor
    assert _keys(plant, "motor_parameter_overrides") == pytest.approx({k: motor.motor_parameter[k] for k in _keys(plant, "motor_parameter_overrides")})
    assert _keys(plant, "dc_limit_overrides") == pytest.approx({k: motor.limits[k] for k in _keys(plant, "dc_limit_overrides")})
    assert _keys(plant, "dc_nominal_overrides") == pytest.approx({k: motor.nominal_values[k] for k in _keys(plant, "dc_nominal_overrides")})
    stock = describe_plant(plant)["stock_motor"]
    assert stock["supply_v"] == pytest.approx(float(system.supply.u_nominal))
    assert stock["motor_parameter"] == pytest.approx(dict(motor.motor_parameter))


@pytest.mark.parametrize("v2, v1", sorted(Scenario.DC_V2_BASE.items()))
def test_v2_plants_have_the_defaults_of_their_v1_base(v2, v1):
    for field in ("motor_parameter_overrides", "dc_limit_overrides", "dc_nominal_overrides"):
        assert _keys(v2, field) == _keys(v1, field), (v2, field)


def test_pmsm_catalog_defaults_equal_the_built_plant():
    from driveflow.control.classical.pmsm_foc import build_pmsm_system

    system = build_pmsm_system()
    keys = _keys("pmsm_v1", "pmsm_motor_parameters")
    mp = system.electrical_motor.motor_parameter
    assert {k: v for k, v in keys.items() if k != "u_supply"} == pytest.approx({k: mp[k] for k in keys if k != "u_supply"})
    assert keys["u_supply"] == pytest.approx(float(system.supply.u_nominal))


def test_buck_catalog_defaults_equal_the_built_plant():
    system, _v_ref, _step = build_buck(Scenario("catalog_check", plant_config_id="buck_v1", controller_type="PI"))
    keys = _keys("buck_v1", "buck_parameter_overrides")
    assert keys == pytest.approx({k: getattr(system.params, k) for k in keys})
