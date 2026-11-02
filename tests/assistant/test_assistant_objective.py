"""Phase 1, F1.6: objective mode (a) -- the user's thresholds (no invented numbers), tuning with J /
J_effort through controller_tuning, and met / not met with margins on the evaluation grid."""

import math

import pytest

from driveflow.assistant import objective as O
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import Agent, ToolFailure
from driveflow.assistant.core.policy import SessionState, visible_tools
from driveflow.assistant.draft import audit_trace
from driveflow.assistant.tools import CONFIG_TOOLS

REQS = [{"metric": "overshoot_pct", "op": "<", "value": 5, "unit": "%"}, {"metric": "settling_time_s", "op": "<", "value": 50, "unit": "ms"}]


def objective_state(text, plant="dc_perm_ex_v1", controller="PI", extra_ops=()):
    s = SessionState(mode="objective")
    s.last_user_seq = s.next_seq()
    s.ledger.add_user_message(1, text)
    r = s.draft.patch([{"op": "set_plant", "plant": plant}, {"op": "set_controller", "controller_type": controller}, *extra_ops], s.ledger)
    assert r["ok"], r["errors"]
    return s


def test_describe_tunable():
    d = O.describe_tunable(SessionState())
    assert d["tunable_from_chat"] and [p["name"] for p in d["parameters"]] == ["speed_bandwidth_rad_s", "speed_damping", "setpoint_weight"]
    assert d["defaults"]["evaluations"] == 48 and set(d["objectives"]) == {"J", "J_effort"}
    assert O.describe_tunable(SessionState(), "mpc")["tunable_from_chat"] is False


def test_thresholds_come_from_the_user_with_their_units():
    s = objective_state("sobrepaso menor a 5 % y asentamiento menor a 50 ms")
    r = O.update_objective(s, requirements=REQS, use_defaults=True)
    assert r["complete"] and r["missing"] == []
    assert [q["value_si"] for q in s.draft.objective["requirements"]] == [5, pytest.approx(0.05)]
    assert s.draft.objective["evaluations"] == {"value": 48, "origin": "default", "source": O.DEFAULTS_SOURCE}
    assert "propose_tuning" in visible_tools(s)


@pytest.mark.parametrize("req, message", [
    ({"metric": "overshoot_pct", "op": "<", "value": 3, "unit": "%"}, "did not write this threshold"),
    ({"metric": "settling_time_s", "op": "<", "value": 0.05, "unit": "ms"}, "did not write"),
    ({"metric": "settling_time_s", "op": "<", "value": 50, "unit": "A"}, "current"),
    ({"metric": "bandwidth", "op": "<", "value": 5, "unit": "%"}, "metric must be"),
    ({"metric": "overshoot_pct", "op": "=", "value": 5, "unit": "%"}, "op must be"),
])
def test_invented_or_malformed_thresholds_are_refused(req, message):
    s = objective_state("sobrepaso menor a 5 % y asentamiento menor a 50 ms")
    with pytest.raises(ToolFailure, match=message):
        O.update_objective(s, requirements=[req])
    assert s.draft.objective is None


def test_budget_from_the_user_or_the_default():
    s = objective_state("sobrepaso < 5 %, con 24 evaluaciones")
    O.update_objective(s, requirements=REQS[:1], evaluations=24, use_defaults=True)
    assert s.draft.objective["evaluations"]["value"] == 24 and s.draft.objective["evaluations"]["origin"] == "user"
    with pytest.raises(ToolFailure, match="did not write"):
        O.update_objective(s, evaluations=60)
    with pytest.raises(ToolFailure, match="default budget is 48"):
        O.update_objective(s, evaluations=60, evaluations_origin="default")
    with pytest.raises(ToolFailure, match="integer"):
        O.update_objective(s, evaluations=1)


@pytest.mark.parametrize("plant, controller, message", [("pmsm_v1", "FOC_MTPA", "only the DC-motor speed PI"), ("dc_perm_ex_v1", "MPC", "no tunable gains")])
def test_what_cannot_be_tuned_is_said(plant, controller, message):
    s = SessionState(mode="objective")
    s.ledger.add_user_message(1, "x")
    s.draft.patch([{"op": "set_plant", "plant": plant}, {"op": "set_controller", "controller_type": controller}], s.ledger)
    with pytest.raises(ToolFailure, match=message):
        O.update_objective(s, requirements=REQS)


def test_j_effort_needs_the_rated_current_of_another_motor():
    s = objective_state("psi_e 1,15 V*s/rad, sobrepaso menor a 5 %",
                        extra_ops=[{"op": "set", "field": "motor_parameter_overrides.psi_e", "value": 1.15, "unit": "V*s/rad"}])
    with pytest.raises(ToolFailure, match="rated current"):
        O.update_objective(s, requirements=REQS[:1], objective="J_effort", use_defaults=True)


def test_motor_block():
    assert O.motor_block(objective_state("x")) is None  # stock motor
    s = objective_state("220 V", extra_ops=[{"op": "set", "field": "dc_supply_v", "value": 220, "unit": "V"}])
    assert O.motor_block(s) == {"dc_supply_v": 220.0}


def test_pmg132_as_motor_values_gives_the_same_simulation():
    """The tuning benches use dc_perm_ex_v1; dc_pmg132_v1 is passed as its datasheet values."""
    from driveflow.datagen import Scenario, run_scenario

    block = O.motor_block(objective_state("PMG 132", plant="dc_pmg132_v1"))
    a = run_scenario(Scenario("a", plant_config_id="dc_pmg132_v1", omega_ref_rad_s=200.0, duration_s=0.2))
    b = run_scenario(Scenario("a", plant_config_id="dc_perm_ex_v1", omega_ref_rad_s=200.0, duration_s=0.2, **block))
    for key in ("rpm", "current_r", "voltage_v", "torque_nm"):
        assert [r[key] for r in a] == [r[key] for r in b], key


def test_requirement_evaluation_worst_case_and_never_settled():
    reqs = [{"metric": "settling_time_s", "op": "<=", "value_si": 0.05, "unit_si": "s"}, {"metric": "load_dip_rad_s", "op": "<=", "value_si": 2.0, "unit_si": "rad/s"},
            {"metric": "peak_current_a", "op": "<=", "value_si": 100.0, "unit_si": "A"}]
    rows = [{"condition": "s1", "kind": "step", "settling_ms": 30.0, "peak_i_a": 90.0}, {"condition": "s2", "kind": "step", "settling_ms": None, "peak_i_a": 50.0},
            {"condition": "l1", "kind": "load", "dip_rad_s": 1.5, "peak_i_a": 120.0}]
    st, dip, peak = O.evaluate_requirements(reqs, rows)
    assert st["met"] is False and st["worst_case"] is None and st["conditions"] == ["s1", "s2"]  # s2 never settled
    assert dip["met"] is True and dip["margin"] == pytest.approx(0.5) and dip["conditions"] == ["l1"]
    assert peak["met"] is False and peak["worst_case"] == 120.0 and peak["margin"] == pytest.approx(-20.0)


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg}}


def test_objective_conversation_tunes_and_reports_met_or_not():
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    ops = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"}]
    script = [reply("", call("set_mode", mode="objective"), call("update_draft", ops=ops)),
              reply("", call("update_objective", requirements=REQS, evaluations=4, use_defaults=True)), reply("", call("propose_tuning")),
              reply("Busco un PI con sobrepaso < 5 % y asentamiento < 50 ms, 4 evaluaciones. ¿Lanzo?"),
              reply("", call("run_tuning")), reply("Lanzado."), reply("", call("check_requirements")), reply("Resultado listo.")]
    a = Agent(llm.ReplayClient({"format": 1, "exchanges": script}, strict=False), CONFIG_TOOLS + O.OBJECTIVE_TOOLS)
    a.user_turn("Quiero un PI del motor DC con sobrepaso menor a 5 % y asentamiento menor a 50 ms; prueba con 4 evaluaciones")
    assert a.state.proposal is not None and not a.state.jobs
    a.user_turn("sí", confirmed_how="typed: sí")
    assert a.state.jobs and a.state.jobs[0]["kind"] == "controller_tuning"
    assert Worker(workspace_root()).serve(once=True, poll_s=0.05) == 1
    a.user_turn("¿y?")
    res = [e for e in a.trace.events if e["kind"] == "tool_result" and e["name"] == "check_requirements"][0]["result"]
    assert res["origin"] == "SIMULATED · AI-PLANNED" and res["status"] == "candidates only; no default changes"
    assert [r["label"] for r in res["results"]][0] == "baseline" and len(res["results"]) >= 2
    for r in res["results"]:
        assert [c["metric"] for c in r["checks"]] == ["overshoot_pct", "settling_time_s"]
        assert all(c["met"] in (True, False) for c in r["checks"])
    assert res["verdict"].startswith("candidate ") is res["all_met"] and (res["all_met"] or res["verdict"].startswith("no candidate meets all"))
    assert audit_trace(a.trace.draft_events) == []
    # The launched job carried the stock motor (no params.motor) and the full bounds.
    from driveflow.jobs.service import JobService
    spec = JobService().experiment(JobService().get(a.state.jobs[0]["job_id"]))
    assert "motor" not in spec.params and spec.params["objective"] == "J" and spec.algorithm == "pso" and spec.budget == {"evaluations": 4}


def test_check_requirements_needs_a_tuning_job():
    s = SessionState()
    s.jobs.append({"job_id": "j", "kind": "scenario_sweep"})
    with pytest.raises(ToolFailure, match="not a tuning job"):
        O.check_requirements(s)


def test_every_objective_tool_has_a_strict_object_schema():
    for t in O.OBJECTIVE_TOOLS:
        p = t.schema()["function"]["parameters"]
        assert p["type"] == "object" and p.get("additionalProperties") is False, t.name
    assert math.isclose(O.METRICS["settling_time_s"][3], 1e-3)
