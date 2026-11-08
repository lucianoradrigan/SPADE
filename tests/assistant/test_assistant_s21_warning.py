"""S21-3 (approved by Luciano 2026-11-02, ordered 2026-11-08): the S-21 warning in controller_design_check,
describe_plant and validate_draft (warning, not blocking) for dc_pmg132_v1 and for the motors with the
0b-D1 friction braking at light load; the assistant shows it on the proposal card.

Decision E3 (Luciano, 2026-11-08): the warning stays for the v1 plants only (frozen GEM-faithful mode); S-21 is
corrected in the v2 plants, which are the only ones the assistant offers. The recorded v1 conversation of
2026-11-08 (docs/audit/assistant/s21_3_cassette_v1_pmg132.json) is kept as evidence; it can no longer be
replayed, because the assistant now refuses v1 plants."""

import json
from pathlib import Path

import pytest

from driveflow.assistant import objective, tools
from driveflow.assistant.cards import card_from_tool
from driveflow.assistant.core import llm
from driveflow.assistant.core.policy import SessionState
from driveflow.assistant.draft import Draft, Ledger
from driveflow.control.design_check import S21_MESSAGE, controller_design_check
from driveflow.datagen.plant_catalog import describe_plant
from driveflow.datagen.s21_warning import condition_warning, free_speed_rad_s, motor_reason

CHANGED = {"motor_parameter_overrides": {"r_a": 0.02, "l_a": 2e-4, "psi_e": 0.2, "j_rotor": 0.02}, "dc_nominal_overrides": {"torque": 10.0, "omega": 250.0}}


def ledger(*texts):
    lg = Ledger()
    for i, t in enumerate(texts, 1):
        lg.add_user_message(i, t)
    return lg


# ------------------------------------------------------------------------------------- the rule
def test_which_motors():
    assert motor_reason("dc_pmg132_v1").startswith("dc_pmg132_v1")
    assert motor_reason("dc_perm_ex_v1") is None  # stock motor, stock friction: not covered by S21-3
    assert "0b-D1" in motor_reason("dc_perm_ex_v1", CHANGED["motor_parameter_overrides"], CHANGED["dc_nominal_overrides"])
    assert motor_reason("dc_perm_ex_v1", CHANGED["motor_parameter_overrides"], None) is None  # no ratings: stock friction kept
    assert motor_reason("dc_pmg132_v1", dc_converter_model="dcm") is None and motor_reason("pmsm_v1") is None


@pytest.mark.parametrize("cond, warned", [
    ({"plant_config_id": "dc_pmg132_v1", "omega_ref_rad_s": 120.0}, True),
    ({"plant_config_id": "dc_pmg132_v1", "omega_ref_rad_s": 120.0, "load_torque_nm": 0.5}, True),
    ({"plant_config_id": "dc_pmg132_v1", "omega_ref_rad_s": 120.0, "load_torque_nm": 1.0}, False),  # 1 N*m: not light
    ({"plant_config_id": "dc_pmg132_v1", "omega_ref_rad_s": 400.0}, False),  # above the free speed 65/0.181 = 359 rad/s
    ({"plant_config_id": "dc_pmg132_v1", "omega_ref_rad_s": 120.0, "dc_converter_model": "dcm"}, False),
    ({"plant_config_id": "dc_perm_ex_v1", "omega_ref_rad_s": 120.0, "torque_ref_nm": 5.0, **CHANGED}, False),  # torque control
    ({"plant_config_id": "dc_perm_ex_v1", "omega_ref_rad_s": 120.0, **CHANGED}, True),
    ({"plant_config_id": "dc_perm_ex_v1", "omega_ref_rad_s": 120.0}, False),
])
def test_which_experiments(cond, warned):
    w = condition_warning(cond)
    assert (w is not None) == warned
    if w:
        assert w["message"] == S21_MESSAGE and w["blocking"] is False and w["id"] == "S-21"


def test_free_speed_is_supply_over_psi_e():
    assert free_speed_rad_s({"plant_config_id": "dc_pmg132_v1"}) == pytest.approx(65.0 / 0.181)
    assert free_speed_rad_s({"plant_config_id": "dc_perm_ex_v1", "dc_supply_v": 48.0, **CHANGED}) == pytest.approx(48.0 / 0.2)


# ----------------------------------------------------------------- design check and describe_plant
def test_controller_design_check_warns_without_changing_its_status():
    mp = {"r_a": 16e-3, "l_a": 19e-6, "psi_e": 0.181}
    plain = controller_design_check(mp, 0.0275, 1e-4)
    flagged = controller_design_check(mp, 0.0275, 1e-4, s21_reason="dc_pmg132_v1")
    assert plain["warnings"] == [] and flagged["warnings"][0]["message"] == S21_MESSAGE
    assert {k: v for k, v in flagged.items() if k != "warnings"} == {k: v for k, v in plain.items() if k != "warnings"}


def test_catalog_describe_plant_carries_the_warning():
    d = describe_plant("dc_pmg132_v1")
    assert d["warnings"][0]["message"] == S21_MESSAGE
    assert d["stock_motor"]["controller_design_check"]["PI"]["status"] == "ok"  # unchanged (test_assistant_f0_plant_catalog)
    assert describe_plant("dc_perm_ex_v1")["warnings"] == []


def test_v2_plants_are_not_warned_at_any_level():
    for plant in ("dc_perm_ex_v2", "dc_pmg132_v2"):
        assert motor_reason(plant) is None and describe_plant(plant)["warnings"] == []
        assert condition_warning({"plant_config_id": plant, "omega_ref_rad_s": 120.0, **(CHANGED if plant == "dc_perm_ex_v2" else {})}) is None
    assert describe_plant("dc_pmg132_v1")["warnings"][0]["message"] == S21_MESSAGE  # v1 keeps it


def test_assistant_refuses_v1_and_describes_v2_without_warning():
    s = SessionState(mode="configuration")
    with pytest.raises(Exception, match="frozen GEM-faithful"):
        tools.describe_plant(s, "dc_pmg132_v1")
    assert tools.describe_plant(s, "dc_pmg132_v2")["warnings"] == []
    r = Draft().patch([{"op": "set_plant", "plant": "dc_perm_ex_v1"}], ledger("x"))
    assert not r["ok"] and "frozen GEM-faithful" in json.dumps(r)


def _v2_draft(text, ops_extra):
    lg = ledger(text)
    d = Draft()
    r = d.patch([{"op": "set_plant", "plant": "dc_pmg132_v2"}, {"op": "set_controller", "controller_type": "PI"}, *ops_extra], lg)
    assert r["ok"], r
    return d, lg


def test_v2_validate_draft_and_proposals_carry_no_s21_warning():
    s = SessionState(mode="configuration")
    s.draft, s.ledger = _v2_draft("PMG 132 con PI a 120 rad/s, 0,3 s", [{"op": "set", "field": "omega_ref_rad_s", "value": 120, "unit": "rad/s"},
                                                                       {"op": "set", "field": "duration_s", "value": 0.3, "unit": "s"}])
    st = s.draft.status()
    assert st["ready"] is True and st["warnings"] == []
    res = tools.propose_experiments(s)
    assert res["warnings"] == [] and not any(ln.startswith("WARNING") for ln in card_from_tool("propose_experiments", {}, res)[0]["lines"])
    t = SessionState(mode="objective")
    t.ledger = ledger("sobrepaso menor a 5 %")
    assert t.draft.patch([{"op": "set_plant", "plant": "dc_pmg132_v2"}, {"op": "set_controller", "controller_type": "PI"}], t.ledger)["ok"]
    objective.update_objective(t, requirements=[{"metric": "overshoot_pct", "op": "<=", "value": 5, "unit": "%"}], use_defaults=True)
    assert objective.propose_tuning(t)["warnings"] == []


def test_the_v1_warning_card_lines_still_render():
    """The card code that shows the warning is kept for v1 results (e.g. reproduced runs)."""
    from driveflow.control.design_check import s21_warning

    res = {"plant": "dc_pmg132_v1", "controller_type": "PI", "experiments": [{"omega_ref_rad_s": 120.0}], "warnings": [{**s21_warning("dc_pmg132_v1"), "experiments": [0]}]}
    lines = card_from_tool("propose_experiments", {}, res)[0]["lines"]
    assert any(ln.startswith("WARNING S-21: " + S21_MESSAGE) and "experiments 0" in ln for ln in lines)
