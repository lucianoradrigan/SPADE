"""AI-N31 (fixed 2026-11-09): the tuning proposal card names an unmodified motor by its plant and says
"custom" only when the user changed the motor."""

import pytest

from driveflow.assistant import objective as O
from driveflow.assistant.cards import card_from_tool
from driveflow.assistant.core.policy import SessionState
from driveflow.assistant.draft import Ledger
from driveflow.datagen.scenario import Scenario


def _card(plant, block):
    res = {"plant": plant, "base_motor": block or "stock motor", "base_motor_label": O.base_motor_label(plant, block), "requirements": []}
    return [ln for ln in card_from_tool("propose_tuning", {}, res)[0]["lines"] if ln.startswith("base motor:")][0]


@pytest.mark.parametrize("plant", Scenario.DC_PLANTS)
def test_unmodified_motor_of_every_dc_plant_is_named_by_its_plant(plant):
    assert _card(plant, O.default_motor_block(plant)) == f"base motor: {plant} (unmodified motor)"


@pytest.mark.parametrize("plant", Scenario.DC_PLANTS)
def test_a_changed_motor_is_custom(plant):
    block = dict(O.default_motor_block(plant) or {})
    block["dc_supply_v"] = 48.0
    line = _card(plant, block)
    assert line.startswith("base motor: custom (") and "dc_supply_v" in line


def test_the_proposal_of_the_assistant_uses_the_label():
    for plant in ("dc_perm_ex_v2", "dc_pmg132_v2"):
        s = SessionState(mode="objective")
        s.ledger = Ledger()
        s.ledger.add_user_message(1, "sobrepaso menor a 5 %")
        assert s.draft.patch([{"op": "set_plant", "plant": plant}, {"op": "set_controller", "controller_type": "PI"}], s.ledger)["ok"]
        O.update_objective(s, requirements=[{"metric": "overshoot_pct", "op": "<=", "value": 5, "unit": "%"}], use_defaults=True)
        res = O.propose_tuning(s)
        lines = card_from_tool("propose_tuning", {}, res)[0]["lines"]
        assert f"base motor: {plant} (unmodified motor)" in lines, lines
