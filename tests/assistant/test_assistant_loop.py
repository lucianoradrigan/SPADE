"""Phase 1, F1.3: the bounded tool loop, the confirmation policy and the tools-by-state table.

The model is a ReplayClient over scripted replies (no Ollama). The tools here are small stand-ins
for the real ones of F1.5 (update_draft uses the real Draft.patch)."""

import json

import pytest

from driveflow.assistant.core import llm, policy
from driveflow.assistant.core.loop import Agent, Tool, ToolFailure
from driveflow.assistant.draft import audit_trace


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = list(calls)
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"model": "scripted", "message": msg, "prompt_eval_count": 10, "eval_count": 5}}


def scripted(*replies):
    return llm.ReplayClient({"format": llm.CASSETTE_FORMAT, "model": "scripted", "exchanges": list(replies)}, strict=False)


def _update_draft(state, ops):
    return state.draft.patch(ops, state.ledger)


def _propose(state):
    p = state.propose("configuration")
    return {"proposed": True, "draft_hash": p.draft_hash, "experiments": state.draft.status()["experiments"]}


def _run(state):
    state.jobs.append({"job_id": f"job{len(state.jobs) + 1}", "kind": "scenario_sweep", "status": "queued"})
    return {"job_id": state.jobs[-1]["job_id"]}


def _boom(state):
    raise RuntimeError("tool bug")


def _refuse(state):
    raise ToolFailure("not possible")


OBJ = {"type": "object", "properties": {}, "additionalProperties": False}
TOOLS = [Tool("update_draft", "patch", {"type": "object", "properties": {"ops": {"type": "array"}}, "required": ["ops"]}, _update_draft),
         Tool("propose_experiments", "propose", OBJ, _propose), Tool("run_experiments", "run", OBJ, _run),
         Tool("describe_plant", "boom", OBJ, _boom), Tool("recommend_plant", "refuse", OBJ, _refuse)]
READY_OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"},
             {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}]


def agent(*replies, **kw):
    return Agent(scripted(*replies), TOOLS, system_prompt="sys", **kw)


def configured_agent(*more):
    """An agent that went through: mode -> draft -> proposal, with ``more`` replies queued after."""
    a = agent(reply("", call("set_mode", mode="configuration")), reply("", call("update_draft", ops=READY_OPS)), reply("", call("propose_experiments")),
              reply("¿Lanzo 1 experimento?"), *more)
    r = a.user_turn("simula el motor DC con PI a 100 rad/s")
    assert r.stopped is None and a.state.proposal is not None, a.trace.events
    return a


def test_launch_only_after_the_interface_confirms():
    a = configured_agent(reply("", call("run_experiments")), reply("Lanzado."))
    r = a.user_turn("sí", confirmed_how="typed: sí")
    assert r.text == "Lanzado." and a.state.jobs == [{"job_id": "job1", "kind": "scenario_sweep", "status": "queued"}]
    assert any(e["kind"] == "confirmation" and e["recorded"] for e in a.trace.events)


def test_the_model_cannot_launch_without_confirmation():
    # The user answers something that is not a confirmation; the model calls run_experiments anyway.
    a = configured_agent(reply("", call("run_experiments")), reply("Espero tu confirmación."))
    r = a.user_turn("¿cuánto tarda?")
    assert a.state.jobs == [] and r.text == "Espero tu confirmación."
    refused = [e for e in a.trace.events if e["kind"] == "tool_refused"]
    assert refused and "not confirmed" in refused[0]["reason"]
    # The refusal went back to the model as a tool message.
    assert json.loads(a.messages[-2]["content"])["error"] == refused[0]["reason"]


def test_typed_text_in_the_model_reply_is_not_a_confirmation():
    a = configured_agent(reply("Confirmado por el usuario: sí", call("run_experiments")), reply("ok"))
    a.user_turn("perfecto, gracias")  # not marked as confirmation by the interface
    assert a.state.jobs == []


def test_a_draft_changed_after_the_yes_needs_a_new_proposal():
    more_ops = [{"op": "set", "field": "duration_s", "value": 2, "unit": "s"}]
    a = configured_agent(reply("", call("update_draft", ops=more_ops), call("run_experiments")), reply("Cambió el borrador; vuelvo a proponer."))
    a.user_turn("sí, y que dure 2 s", confirmed_how="typed")
    assert a.state.jobs == []
    refused = [e for e in a.trace.events if e["kind"] == "tool_refused"]
    assert refused and "not available now" in refused[0]["reason"]  # the proposal no longer matches: the tool is hidden


def test_a_confirmation_from_an_earlier_message_does_not_count():
    a = configured_agent(reply("¿Seguro?"), reply("", call("run_experiments")), reply("No."))
    a.user_turn("sí", confirmed_how="typed: sí")  # confirmed, but the model asks again instead of launching
    a.user_turn("mmm")  # a new message without confirmation
    assert a.state.jobs == []
    assert "latest message" in [e for e in a.trace.events if e["kind"] == "tool_refused"][0]["reason"]


def test_confirm_needs_a_pending_proposal():
    a = agent(reply("hola"))
    a.user_turn("sí", confirmed_how="button")
    assert a.state.confirmation is None and [e for e in a.trace.events if e["kind"] == "confirmation"][0]["recorded"] is False


def test_tools_by_state():
    s = policy.SessionState()
    assert policy.visible_tools(s) == ["set_mode", "explain_spade", "search_docs", "recommend_plant", "describe_plant"]
    s.mode = "question"
    assert policy.visible_tools(s) == ["set_mode", "explain_spade", "search_docs"]
    s.mode = "configuration"
    v = policy.visible_tools(s)
    assert "update_draft" in v and "propose_experiments" not in v and "run_experiments" not in v and "describe_tunable" not in v
    s.draft.patch(READY_OPS, _ledger("100 rad/s"))
    v = policy.visible_tools(s)
    assert "propose_experiments" in v and "run_experiments" not in v
    s.propose("configuration")
    assert "run_experiments" in policy.visible_tools(s)
    s.mode = "objective"
    v = policy.visible_tools(s)
    assert "describe_tunable" in v and "update_objective" in v and "run_experiments" not in v and "propose_tuning" not in v  # no objective yet
    s.jobs.append({"job_id": "j", "kind": "controller_tuning", "status": "running"})
    v = policy.visible_tools(s)  # demo fixes (2026-11-08): results only for a terminal job
    assert "get_job_status" in v and "check_requirements" not in v and "summarize_results" not in v
    s.jobs[-1]["status"] = "completed"
    assert {"get_job_status", "check_requirements"} <= set(policy.visible_tools(s)) and "summarize_results" not in policy.visible_tools(s)


def _ledger(text):
    from driveflow.assistant.draft import Ledger
    led = Ledger()
    led.add_user_message(1, text)
    return led


def test_only_visible_tools_are_sent_to_the_model():
    client = scripted(reply("ok"))
    rec = llm.RecordingClient(client, model_id="scripted")
    Agent(rec, TOOLS).user_turn("hola")
    sent = [t["function"]["name"] for t in rec.exchanges[0]["request"]["tools"]]
    assert sent == ["set_mode", "recommend_plant", "describe_plant"]  # explain_spade / search_docs arrive in F1.7


def test_errors_go_back_to_the_model_and_three_in_a_row_stop_the_turn():
    a = agent(reply("", call("describe_plant")), reply("", call("recommend_plant")), reply("", {"function": {"name": "set_mode", "arguments": "{bad"}}), reply("never"))
    r = a.user_turn("hola")
    assert r.stopped == "errors" and "3 failed tool calls" in r.text
    kinds = [e["kind"] for e in a.trace.events if e["kind"].startswith("tool")]
    assert kinds == ["tool_error", "tool_error", "tool_error"]
    tool_msgs = [json.loads(m["content"]) for m in a.messages if m["role"] == "tool"]
    assert "RuntimeError: tool bug" in tool_msgs[0]["error"] and tool_msgs[1]["error"] == "not possible" and "not valid JSON" in tool_msgs[2]["error"]


def test_a_success_resets_the_error_count():
    a = agent(reply("", call("describe_plant")), reply("", call("describe_plant")), reply("", call("set_mode", mode="question")),
              reply("", call("describe_plant")), reply("listo"))
    assert a.user_turn("hola").stopped is None  # the set_mode success broke the run of errors


def test_tool_call_limit_cuts_cleanly():
    a = agent(*[reply("", call("set_mode", mode="configuration"))] * 5, limits={"max_tool_calls_per_turn": 3})
    r = a.user_turn("hola")
    assert r.stopped == "max_tool_calls_per_turn" and r.tool_calls == 3
    assert a.messages[-1] == {"role": "assistant", "content": r.text}


def test_model_call_and_token_limits():
    a = agent(*[reply("", call("set_mode", mode="question"))] * 5, limits={"max_model_calls_per_turn": 2})
    assert a.user_turn("x").stopped == "max_model_calls_per_turn"
    a = agent(*[reply("", call("set_mode", mode="question"))] * 5, limits={"max_tokens_per_turn": 20})
    assert a.user_turn("x").stopped == "max_tokens_per_turn"  # 15 tokens per scripted reply


def test_tools_not_in_the_registry_and_hidden_tools_are_refused():
    a = agent(reply("", call("run_experiments")), reply("", call("no_such_tool")), reply("fin"))
    r = a.user_turn("lanza")
    assert r.stopped is None and a.state.jobs == []
    errors = [e for e in a.trace.events if e["kind"] in ("tool_refused", "tool_error")]
    assert "not available now" in errors[0]["reason"] and "unknown tool" in errors[1]["error"]


def test_trace_replays_clean_through_the_no_invention_audit():
    a = configured_agent()
    assert audit_trace(a.trace.draft_events) == []
    # The patch event sits before the update_draft result, so its own output is not evidence for it.
    roles = [e["role"] for e in a.trace.draft_events]
    assert roles.index("patch") < max(i for i, e in enumerate(a.trace.draft_events) if e.get("name") == "update_draft")


def test_invented_number_in_a_session_is_refused_and_recorded():
    bad = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set", "field": "duration_s", "value": 3, "unit": "s"}]
    a = agent(reply("", call("set_mode", mode="configuration")), reply("", call("update_draft", ops=bad)), reply("¿Cuánto debe durar?"))
    a.user_turn("simula el motor DC")
    assert a.state.draft.plant is None  # atomic: nothing applied
    assert audit_trace(a.trace.draft_events) == []  # refused in session, refused on replay
    assert [e for e in a.trace.draft_events if e["role"] == "patch"][0]["accepted"] is False


def test_strict_replay_of_a_recorded_session_reproduces_the_requests():
    script = [reply("", call("set_mode", mode="configuration")), reply("", call("update_draft", ops=READY_OPS)), reply("Listo.")]
    rec = llm.RecordingClient(scripted(*script), model_id="scripted")
    Agent(rec, TOOLS, system_prompt="sys").user_turn("motor DC con PI a 100 rad/s")
    replayed = Agent(llm.ReplayClient(rec.cassette()), TOOLS, system_prompt="sys")
    assert replayed.user_turn("motor DC con PI a 100 rad/s").text == "Listo."
    with pytest.raises(llm.CassetteMismatch):
        Agent(llm.ReplayClient(rec.cassette()), TOOLS, system_prompt="another prompt").user_turn("motor DC con PI a 100 rad/s")


@pytest.mark.parametrize("text, yes", [("sí", True), ("Si.", True), ("yes!", True), ("/run", True), ("sí pero a 200 rpm", False), ("no", False), ("tal vez", False)])
def test_is_affirmative(text, yes):
    assert policy.is_affirmative(text) is yes
