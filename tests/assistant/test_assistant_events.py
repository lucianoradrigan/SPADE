"""FE-1: the assistant's event contract (assistant/events.py), the loop's on_event, and the pure
event -> avatar-state mapping (viz/adapters/assistant_adapter.avatar_state)."""

import pytest

from driveflow.assistant import events as ev
from driveflow.assistant import objective as O
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import SET_MODE, Agent
from driveflow.assistant.tools import CONFIG_TOOLS
from driveflow.viz.adapters.assistant_adapter import PRIORITY, STATES, TOOL_LABELS, avatar_state


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


def agent(*replies, tools=CONFIG_TOOLS, seen=None):
    return Agent(llm.ReplayClient({"format": 1, "exchanges": list(replies)}, strict=False), tools, on_event=(seen.append if seen is not None else None))


OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"}, {"op": "set", "field": "duration_s", "value": 0.1, "unit": "s"}]


def states_along(events):
    return [avatar_state(events[: i + 1])[0] for i in range(len(events))]


def test_every_registered_tool_has_a_label():
    for t in [SET_MODE, *CONFIG_TOOLS, *O.OBJECTIVE_TOOLS]:
        assert t.name in TOOL_LABELS, t.name
    assert set(PRIORITY) <= set(STATES)


def test_events_are_frozen_dicts_with_known_types():
    for e in (ev.TurnStarted(1), ev.LlmCall("start"), ev.ToolCallEvent("end", "describe_plant", True), ev.ProposalReady("h", 2, 3.0), ev.UserConfirmed("h", 3),
              ev.JobEvent("j", "running", 1, 2), ev.ResultsReady("r"), ev.ErrorEvent("tool:x", "m"), ev.RuntimeUnavailableEvent("down"), ev.TurnEnded()):
        d = e.to_dict()
        assert d["type"] in ev.EVENT_TYPES
        with pytest.raises(Exception):
            e.type = "other"  # frozen


def test_configuration_conversation_states():
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    seen = []
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo?"),
              reply("", call("run_experiments")), reply("Lanzado."), reply("", call("get_job_status"), call("summarize_results")), reply("Listo."), seen=seen)
    a.user_turn("motor DC con PI, 0.1 s")
    assert avatar_state(seen) == ("awaiting_confirmation", "Waiting for your confirmation", None)
    n1 = len(seen)
    a.user_turn("sí", confirmed_how="typed: sí")
    second = seen[n1:]
    assert [e.type for e in second][:2] == ["turn_started", "user_confirmed"]
    assert avatar_state(seen[: n1 + 2])[0] == "confirmed"
    assert "using_tool" in states_along(seen)
    assert avatar_state(seen)[0] == "running" and avatar_state(seen)[1] == "Simulation running: 0/1"
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    a.user_turn("¿resultados?")
    assert avatar_state(seen) == ("results_ready", "Results ready (simulated)", None)
    assert a.trace.agent_events == [e.to_dict() for e in seen]


def test_results_ready_never_leads_to_confirmed():
    """FE-D6: the avatar never celebrates results; only the user's confirmation shows 'confirmed'."""
    seen = []
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("ok"), seen=seen)
    a.user_turn("motor DC con PI, 0.1 s")
    fake = seen + [ev.JobEvent("j", "completed", 1, 1), ev.ResultsReady("r")]
    sts = states_along(fake)
    assert "confirmed" not in sts and sts[-1] == "awaiting_confirmation"  # the proposal is still pending
    assert not any(e.type == "user_confirmed" for e in seen)
    only = [ev.TurnStarted(1), ev.JobEvent("j", "running", 0, 1), ev.JobEvent("j", "completed", 1, 1), ev.ResultsReady("r"), ev.TurnEnded()]
    assert "confirmed" not in states_along(only) and avatar_state(only)[0] == "results_ready"


def test_question_turn_thinks_then_idles():
    seen = []
    a = agent(reply("", call("set_mode", mode="question")), reply("SPADE simula motores."), seen=seen)
    a.user_turn("¿qué es SPADE?")
    sts = states_along(seen)
    assert sts[0] == "idle" and "thinking" in sts and sts[-1] == "idle"
    assert avatar_state(seen[:2]) == ("thinking", "Thinking…", None)


def test_tool_error_shows_until_the_next_turn():
    seen = []
    a = agent(reply("", call("update_draft", ops=[{"op": "set_plant", "plant": "nope"}])), reply("No existe."), reply("hola"), seen=seen)
    a.user_turn("algo")
    assert avatar_state(seen)[0] == "error" and "not available now" in avatar_state(seen)[1]  # update_draft is hidden in undetermined mode
    a.user_turn("otra cosa")
    assert avatar_state(seen)[0] == "idle"


def test_objective_turn_labels():
    seen = []
    a = agent(reply("", call("set_mode", mode="objective"), call("describe_tunable")), reply("Puedo sintonizar el PI."), tools=CONFIG_TOOLS + O.OBJECTIVE_TOOLS, seen=seen)
    a.user_turn("quiero un PI con sobrepaso < 5 %")
    labels = [avatar_state(seen[: i + 1])[1] for i in range(len(seen))]
    assert "Checking what can be tuned…" in labels


def test_runtime_unavailable():
    class Down:
        def chat(self, *a, **k):
            raise llm.RuntimeUnavailable("the local model runtime does not answer at http://127.0.0.1:11434")

    seen = []
    a = Agent(Down(), CONFIG_TOOLS, on_event=seen.append)
    with pytest.raises(llm.RuntimeUnavailable):
        a.user_turn("hola")
    state, label, _ = avatar_state(seen)
    assert state == "unavailable" and label.startswith("Assistant unavailable") and seen[-1].type == "turn_ended"


def test_priority_when_several_states_are_active():
    events = [ev.JobEvent("j", "running", 1, 4), ev.LlmCall("start"), ev.ToolCallEvent("start", "estimate_cost")]
    assert avatar_state(events) == ("using_tool", "Estimating cost…", None)
    assert avatar_state(events + [ev.ErrorEvent("tool:x", "bad\nsecond line")])[:2] == ("error", "bad")
    assert avatar_state(events + [ev.RuntimeUnavailableEvent("down")])[0] == "unavailable"
    assert avatar_state([ev.JobEvent("j", "running", 1, 4)]) == ("running", "Simulation running: 1/4", 0.25)
