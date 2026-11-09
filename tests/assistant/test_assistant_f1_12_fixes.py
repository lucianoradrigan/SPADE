"""F1.12 fixes (Luciano, 2026-11-08; designed by reading the F1.12 conversations -- declared; F1.13 is the
independent test). Scripted model replies, no Ollama."""

from driveflow.assistant import catalog, guardrails, objective, tools
from driveflow.assistant.cards import card_from_tool
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import DEFAULT_GUARDS, Agent

ALL = tools.CONFIG_TOOLS + objective.OBJECTIVE_TOOLS + catalog.CATALOG_TOOLS
OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"},
       {"op": "set", "field": "omega_ref_rad_s", "value": 105, "unit": "rad/s"}, {"op": "set", "field": "duration_s", "value": 0.25, "unit": "s"}]
USER = "Simulate the stock DC motor with PI at 105 rad/s for 0.25 s."


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


def agent(*replies):
    return Agent(llm.ReplayClient({"format": 1, "exchanges": list(replies)}, strict=False), ALL, guards=DEFAULT_GUARDS)


def test_proposal_cost_only_in_its_card():
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")),
              reply("Proposed experiment:\n- Plant: dc_perm_ex_v2\n- Estimated CPU time: ~0.79 s\nPlease confirm with yes."))
    res = a.user_turn(USER)
    assert "0.79" not in res.text and "CPU" not in res.text and res.text.endswith("Please confirm with yes.")
    assert any(ln.startswith("estimated CPU") for c in res.cards if c["title"].startswith("Proposal") for ln in c["lines"])


def test_single_allowed_controller_is_filled_with_origin_default():
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=[{"op": "set_plant", "plant": "vsc_dpc_v1"}])),
              reply("Which load resistance do you want?"), reply("Which load resistance do you want?"))  # second: after the nudge_docs note
    res = a.user_turn("Simulate the voltage source converter with a 70 V reference for 0.06 s.")
    assert a.state.draft.controller_type == "DPC"
    card = [c for c in res.cards if c["tool"] == "filled"][0]
    assert card["lines"] == ["controller_type = DPC (origin default: the only controller allowed on vsc_dpc_v1; configs/plant_catalog.yaml)"]


def test_no_documentation_card_drops_earlier_search_cards():
    a = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="DC motor bearing")),
              reply("", call("search_docs", query="lubrication interval bearing schedule")), reply("SPADE does not document it."))
    res = a.user_turn("What is the bearing lubrication interval of the DC motor according to SPADE?")
    search = [c for c in res.cards if c["tool"] == "search_docs"]
    assert search and all(c["title"].startswith("No documentation found") for c in search)


def test_job_state_only_in_the_card():
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("Confirm?"),
              reply("", call("run_experiments"), call("get_job_status")), reply("The simulation job has been queued and is running on a SPADE worker. I will summarize when done."))
    a.user_turn(USER)
    res = a.user_turn("/run", confirmed_how="button")
    assert "running" not in res.text and guardrails.JOB_STATE_LEAD["en"] in res.text
    job = [c for c in res.cards if c["tool"] == "get_job_status"][0]
    assert "state: queued" in job["lines"]


def test_leaked_reasoning_and_orphan_lead_in_are_removed():
    leak = ("The user wants to run a simulation immediately without confirmation. However, the system requires explicit confirmation.\n\n"
            "Let me clarify: the rules state never run anything before their yes.\n\nPlease confirm with yes to launch it?")
    text, n = guardrails.strip_reasoning(leak)
    assert n >= 1 and text == "Please confirm with yes to launch it?"
    out, cards, removed = guardrails.bound_capabilities("SPADE cannot do that. Here is the proposal.", [], "en")
    assert removed == 1 and cards == [] and out == "Here is the proposal."  # no orphan "I cannot confirm that…"


def test_launch_card():
    c = card_from_tool("run_experiments", {}, {"job_id": "job-1", "experiment_id": "exp-1", "n_conditions": 1})[0]
    assert c["title"] == "Job job-1 submitted" and "state: queued (waiting for a SPADE worker)" in c["lines"]
