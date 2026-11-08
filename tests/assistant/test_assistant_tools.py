"""Phase 1, F1.5: configuration-mode tools over existing SPADE code, and one full conversation
(scripted model) that configures, proposes, is confirmed, runs on a real worker and summarises."""

import math

import pytest

from driveflow.assistant import tools as T
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import Agent, ToolFailure
from driveflow.assistant.core.policy import SessionState
from driveflow.assistant.draft import audit_trace


def state_with(*texts):
    s = SessionState(mode="configuration")
    for i, t in enumerate(texts, 1):
        s.ledger.add_user_message(i, t)
        s.last_user_seq = s.seq = i
    return s


@pytest.mark.parametrize("text, plant", [
    ("quiero simular un motor DC de imanes permanentes", "dc_perm_ex_v2"),
    ("un PMG 132 de Perm-Motor", "dc_pmg132_v2"),
    ("a brushless servo (PMSM)", "pmsm_v1"),
    ("un inversor con filtro LC", "vsc_dpc_v1"),
    ("convertidor reductor buck de 48 a 24 V", "buck_v1"),
])
def test_recommend_plant(text, plant):
    r = T.recommend_plant(SessionState(), text)
    assert r["matches"][0]["plant"] == plant and "not_simulable" not in r


@pytest.mark.parametrize("text, substitute", [("un motor de inducción jaula de ardilla", "pmsm_v1"), ("a boost converter", "buck_v1")])
def test_recommend_plant_says_when_something_is_not_simulable(text, substitute):
    r = T.recommend_plant(SessionState(), text)
    assert r["matches"] == [] and r["not_simulable"]["closest_substitutes"][0]["plant"] == substitute


def test_recommend_plant_unknown_lists_the_plants():
    r = T.recommend_plant(SessionState(), "algo que no sé")
    assert r["matches"] == [] and len(r["plants"]) == 5


def test_describe_plant_is_compact_and_hides_faults():
    r = T.describe_plant(SessionState(), "dc_perm_ex_v2")
    names = {f["name"] for f in r["fields"]}
    assert "load_torque_nm" in names and "fault_type" not in names and "scenario_id" not in names
    assert r["stock_motor"]["nominal_values"]["torque"] == 16.0 and r["stock_motor"]["design_check_PI"] in ("ok", "warning")
    with pytest.raises(ToolFailure):
        T.describe_plant(SessionState(), "induction_v1")


def test_suggest_value_only_from_internal_sources():
    s = state_with("motor DC")
    s.draft.patch([{"op": "set_plant", "plant": "dc_perm_ex_v2"}], s.ledger)
    assert T.suggest_value(s, "omega_ref_rad_s") == {"field": "omega_ref_rad_s", "unit": "rad/s", "value": 300.0, "rule": "the motor's rated speed",
                                                       "source": "rated omega of the dc_perm_ex_v2 stock motor (describe_plant)"}
    assert T.suggest_value(s, "duration_s")["value"] == 1.0
    no = T.suggest_value(s, "motor_parameter_overrides.l_a")
    assert no["value"] is None and "not suggested" in no["why"]
    # Another motor without ratings: the stock rated speed does not apply, and nothing else backs a suggestion.
    s2 = state_with("motor con psi_e 1,15 V*s/rad")
    s2.draft.patch([{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set", "field": "motor_parameter_overrides.psi_e", "value": 1.15, "unit": "V*s/rad"}], s2.ledger)
    assert T.suggest_value(s2, "omega_ref_rad_s")["value"] == 150.0  # falls back to SPADE's default, labelled as such
    assert T.suggest_value(s2, "omega_ref_rad_s")["rule"] == "SPADE's default for this field"


def test_suggestion_round_trip_through_the_ledger():
    s = state_with("motor DC")
    s.draft.patch([{"op": "set_plant", "plant": "dc_perm_ex_v2"}], s.ledger)
    sug = T.suggest_value(s, "omega_ref_rad_s")
    s.ledger.add_tool_result("suggest_value", "c1", sug)
    r = T.update_draft(s, [{"op": "set", "field": "omega_ref_rad_s", "value": sug["value"], "unit": sug["unit"], "origin": "suggested"}])
    assert r["ok"] and s.draft.base["omega_ref_rad_s"].origin == "suggested"


def test_propose_requires_a_ready_draft():
    s = state_with("PMSM")
    s.draft.patch([{"op": "set_plant", "plant": "pmsm_v1"}], s.ledger)
    with pytest.raises(ToolFailure, match="not ready"):
        T.propose_experiments(s)


def test_kpis_of_each_plant_from_a_real_run():
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    cases = [({"plant_config_id": "dc_perm_ex_v2", "omega_ref_rad_s": 100.0, "duration_s": 0.3}, "speed_step"),
             ({"plant_config_id": "pmsm_v1", "controller_type": "FOC_MTPA", "pmsm_current_ref_a": 10.0, "duration_s": 0.05}, "i_q_step"),
             ({"plant_config_id": "buck_v1", "duration_s": 0.02}, "voltage_step"),
             ({"plant_config_id": "vsc_dpc_v1", "controller_type": "DPC", "duration_s": 0.04}, "rms_voltage_error_second_half_v")]
    for cond, key in cases:
        df = records_to_dataframe(run_scenario(Scenario("k", **cond)))
        k = T.condition_kpis(df, cond)
        assert key in k, (cond, k)
    dc = T.condition_kpis(records_to_dataframe(run_scenario(Scenario("k", plant_config_id="dc_perm_ex_v2", omega_ref_rad_s=100.0, duration_s=0.3))),
                          {"plant_config_id": "dc_perm_ex_v2", "omega_ref_rad_s": 100.0})
    assert abs(dc["final_speed_rad_s"] - 100.0) < 2.0 and dc["speed_step"]["finite"] is True


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"model": "scripted", "message": msg}}


def test_full_conversation_configures_runs_and_summarises():
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    ops = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"},
           {"op": "vary", "field": "omega_ref_rad_s", "values": [50, 100], "unit": "rad/s"}, {"op": "set", "field": "duration_s", "value": 0.3, "unit": "s"}]
    script = [reply("", call("set_mode", mode="configuration"), call("recommend_plant", description="motor DC")),
              reply("", call("update_draft", ops=ops)), reply("", call("propose_experiments")), reply("Propongo 2 experimentos de 0,3 s. ¿Lanzo?"),
              reply("", call("run_experiments")), reply("Lanzado."),
              reply("", call("get_job_status")), reply("", call("summarize_results")), reply("Resultados listos.")]
    a = Agent(llm.ReplayClient({"format": 1, "exchanges": script}, strict=False), T.CONFIG_TOOLS)
    a.user_turn("Simula el motor DC con PI a 50 y 100 rad/s, 0.3 s cada una")
    assert a.state.proposal is not None and a.state.jobs == []
    a.user_turn("sí", confirmed_how="typed: sí")
    assert len(a.state.jobs) == 1 and a.state.proposal is None
    assert Worker(workspace_root()).serve(once=True, poll_s=0.05) == 1
    a.user_turn("¿cómo salió?")
    results = [e for e in a.trace.events if e["kind"] == "tool_result" and e["name"] == "summarize_results"][0]["result"]
    assert results["run_status"] == "completed" and len(results["rows"]) == 2 and results["origin"] == "SIMULATED · AI-PLANNED"
    finals = [r["kpis"]["final_speed_rad_s"] for r in results["rows"]]
    assert abs(finals[0] - 50) < 2 and abs(finals[1] - 100) < 2
    assert all(math.isfinite(r["kpis"]["speed_step"]["rmse"]) for r in results["rows"])
    assert audit_trace(a.trace.draft_events) == []
    from driveflow.jobs.service import JobService
    spec = JobService().experiment(JobService().get(a.state.jobs[0]["job_id"]))
    assert spec.params == {"origin": T.AI_ORIGIN} and spec.plant_profile_id == "native:dc_perm_ex_v2"
    assert [c["omega_ref_rad_s"] for c in spec.conditions] == [50.0, 100.0]


def test_job_tools_need_a_job():
    with pytest.raises(ToolFailure, match="no job"):
        T.get_job_status(SessionState())
    s = SessionState()
    s.jobs.append({"job_id": "j1"})
    with pytest.raises(ToolFailure, match="not launched in this session"):
        T.summarize_results(s, "other")


def test_every_tool_has_a_strict_object_schema():
    for t in T.CONFIG_TOOLS:
        p = t.schema()["function"]["parameters"]
        assert p["type"] == "object" and p.get("additionalProperties") is False, t.name


def test_scenario_sweep_keeps_the_plain_label_unless_the_assistant_planned_it():
    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker
    from driveflow.runs.schemas import ExperimentSpec
    from driveflow.runs.store import workspace_root

    svc = JobService()
    jobs = [svc.submit(ExperimentSpec.create("scenario_sweep", conditions=[{"scenario_id": "o", "duration_s": 0.02}], params=p))
            for p in ({}, {"origin": "MEASURED · EXTERNAL DATASET"}, {"origin": T.AI_ORIGIN})]
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    assert [svc.run_for(svc.get(j.job_id)).origin for j in jobs] == ["SIMULATED", "SIMULATED", "SIMULATED · AI-PLANNED"]
