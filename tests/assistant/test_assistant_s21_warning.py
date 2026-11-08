"""S21-3 (approved by Luciano 2026-11-02, ordered 2026-11-08): the S-21 warning in controller_design_check,
describe_plant and validate_draft (warning, not blocking) for dc_pmg132_v1 and for the motors with the
0b-D1 friction braking at light load; the assistant shows it on the proposal card.

The recorded conversation (tests/assistant/cassettes/s21_warning_pmg132.json, qwen3.5:9b through
Ollama, recorded with experiments/report_s21_warning.py) is replayed strictly: the requests must be
the recorded ones, so a change of prompt or tools breaks this test on purpose."""

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

CASSETTE = Path(__file__).parent / "cassettes" / "s21_warning_pmg132.json"
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


def test_assistant_describe_plant_warns_for_pmg132_and_for_the_drafts_changed_motor():
    s = SessionState(mode="configuration")
    assert tools.describe_plant(s, "dc_pmg132_v1")["warnings"][0]["id"] == "S-21"
    assert tools.describe_plant(s, "dc_perm_ex_v1")["warnings"] == []
    text = "motor DC con r_a 0.02 ohm, l_a 0.0002 H, psi_e 0.2 V*s/rad, j_rotor 0.02 kg*m^2, par nominal 10 N*m y velocidad nominal 250 rad/s, a 120 rad/s"
    ops = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"},
           {"op": "set", "field": "motor_parameter_overrides.r_a", "value": 0.02, "unit": "ohm"},
           {"op": "set", "field": "motor_parameter_overrides.l_a", "value": 0.0002, "unit": "H"},
           {"op": "set", "field": "motor_parameter_overrides.psi_e", "value": 0.2, "unit": "V*s/rad"},
           {"op": "set", "field": "motor_parameter_overrides.j_rotor", "value": 0.02, "unit": "kg*m^2"},
           {"op": "set", "field": "dc_nominal_overrides.torque", "value": 10, "unit": "N*m"},
           {"op": "set", "field": "dc_nominal_overrides.omega", "value": 250, "unit": "rad/s"},
           {"op": "set", "field": "omega_ref_rad_s", "value": 120, "unit": "rad/s"}]
    s.ledger = ledger(text)
    r = s.draft.patch(ops, s.ledger)
    assert r["ok"], r
    w = tools.describe_plant(s, "dc_perm_ex_v1")["warnings"]
    assert w and "0b-D1" in w[0]["applies_to"]
    card = card_from_tool("describe_plant", {"plant": "dc_perm_ex_v1"}, tools.describe_plant(s, "dc_perm_ex_v1"))[0]
    assert any(ln.startswith("WARNING S-21: " + S21_MESSAGE) for ln in card["lines"])


# ------------------------------------------------------------------------- validate_draft and proposal
def _pmg_draft(text, ops_extra):
    lg = ledger(text)
    d = Draft()
    r = d.patch([{"op": "set_plant", "plant": "dc_pmg132_v1"}, {"op": "set_controller", "controller_type": "PI"}, *ops_extra], lg)
    assert r["ok"], r
    return d, lg


def test_validate_draft_marks_the_light_load_experiments_and_does_not_block():
    d, _ = _pmg_draft("PMG 132 con PI a 120 y 400 rad/s, 0,3 s", [{"op": "vary", "field": "omega_ref_rad_s", "values": [120, 400], "unit": "rad/s"},
                                                               {"op": "set", "field": "duration_s", "value": 0.3, "unit": "s"}])
    st = d.status()
    assert st["ready"] is True  # a warning never blocks
    assert [w["experiment"] for w in st["warnings"]] == [0] and st["warnings"][0]["message"] == S21_MESSAGE


def test_the_configuration_proposal_card_shows_the_warning():
    s = SessionState(mode="configuration")
    s.draft, s.ledger = _pmg_draft("PMG 132 con PI a 120 rad/s, 0,3 s", [{"op": "set", "field": "omega_ref_rad_s", "value": 120, "unit": "rad/s"},
                                                                         {"op": "set", "field": "duration_s", "value": 0.3, "unit": "s"}])
    res = tools.propose_experiments(s)
    assert res["warnings"][0]["experiments"] == [0]
    lines = card_from_tool("propose_experiments", {}, res)[0]["lines"]
    assert any(ln.startswith("WARNING S-21: " + S21_MESSAGE) and "experiments 0" in ln and "not blocking" in ln for ln in lines)


def test_the_tuning_proposal_card_shows_the_warning_for_pmg132_only():
    def proposal(plant):
        s = SessionState(mode="objective")
        s.ledger = ledger("sobrepaso menor a 5 %")
        assert s.draft.patch([{"op": "set_plant", "plant": plant}, {"op": "set_controller", "controller_type": "PI"}], s.ledger)["ok"]
        objective.update_objective(s, requirements=[{"metric": "overshoot_pct", "op": "<=", "value": 5, "unit": "%"}], use_defaults=True)
        return objective.propose_tuning(s)

    pmg = proposal("dc_pmg132_v1")
    assert pmg["warnings"] and "no load" in pmg["warnings"][0]["experiments"]
    assert any(ln.startswith("WARNING S-21") for ln in card_from_tool("propose_tuning", {}, pmg)[0]["lines"])
    assert proposal("dc_perm_ex_v1")["warnings"] == []


# ---------------------------------------------------------------------------- recorded conversation
@pytest.mark.skipif(not CASSETTE.exists(), reason="cassette not recorded")
def test_recorded_conversation_shows_the_warning_on_the_proposal_card():
    from driveflow.assistant.cli import build_agent
    from driveflow.assistant.config import load_config
    from driveflow.assistant.draft import audit_trace

    cas = json.loads(CASSETTE.read_text())
    meta = cas["meta"]
    a = build_agent(client=llm.ReplayClient(cas), config=load_config())  # as the CLI and the page build it
    cards = []
    for text in meta["user_turns"]:
        res = a.user_turn(text)
        cards += res.cards
    proposal = [c for c in cards if c["title"].startswith("Proposal:")]
    assert proposal, [c["title"] for c in cards]
    assert any(ln.startswith("WARNING S-21: " + S21_MESSAGE) for ln in proposal[-1]["lines"])
    assert a.state.proposal is not None and a.state.jobs == []  # proposed, not launched
    assert audit_trace(a.trace.draft_events) == []  # no invented numbers
