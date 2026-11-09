"""S-24 (decision C6, 2026-11-09, for v0.2.1): the DC voltage limit is checked against the EFFECTIVE supply,
the given dc_supply_v or the stock 65 V when it is not given."""

import pytest

from driveflow.datagen import Scenario
from driveflow.datagen.runner import build_plant

PLANTS = ("dc_perm_ex_v1", "dc_pmg132_v1", "dc_perm_ex_v2", "dc_pmg132_v2")


@pytest.mark.parametrize("plant", PLANTS)
def test_a_limit_above_the_stock_supply_is_rejected_when_no_supply_is_given(plant):
    with pytest.raises(ValueError, match=r"must not exceed the supply: dc_supply_v is not given, so it is the stock 65 V"):
        Scenario("x", plant_config_id=plant, dc_limit_overrides={"u": 100.0})


@pytest.mark.parametrize("plant", PLANTS)
def test_a_limit_above_a_given_supply_is_rejected_as_before(plant):
    with pytest.raises(ValueError, match=r"must not exceed dc_supply_v \(50.0 V\)"):
        Scenario("x", plant_config_id=plant, dc_supply_v=50.0, dc_limit_overrides={"u": 60.0})


@pytest.mark.parametrize("plant", PLANTS)
def test_limits_within_the_effective_supply_are_accepted(plant):
    Scenario("x", plant_config_id=plant, dc_limit_overrides={"u": 65.0})
    Scenario("x", plant_config_id=plant, dc_supply_v=120.0, dc_limit_overrides={"u": 100.0})


def test_the_stock_supply_is_the_runners():
    assert float(build_plant(Scenario("x", plant_config_id="dc_perm_ex_v1")).supply.u_nominal) == Scenario.DC_STOCK_SUPPLY_V
    assert float(build_plant(Scenario("x", plant_config_id="dc_pmg132_v1")).supply.u_nominal) == Scenario.DC_STOCK_SUPPLY_V
