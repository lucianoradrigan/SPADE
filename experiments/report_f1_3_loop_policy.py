"""Phase 1, F1.3: tools-by-state table and confirmation scenarios of the assistant's loop (no LLM).

The table is generated from driveflow.assistant.core.policy (so it cannot drift from the code).
The scenarios run the real Agent loop with a scripted model (ReplayClient) and stand-in tools.

    python experiments/report_f1_3_loop_policy.py
Writes docs/audit/assistant/f1_3_loop_policy.{md,json}.
"""

from __future__ import annotations

import json
from pathlib import Path

from driveflow.assistant.core import llm, policy
from driveflow.assistant.core.loop import Agent, Tool
from driveflow.assistant.draft import Ledger

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f1_3_loop_policy"
READY_OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"},
             {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}]
OBJ = {"type": "object", "properties": {}, "additionalProperties": False}


def _call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def _reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg}}


def _tools():
    def run(state):
        state.jobs.append({"job_id": "job1", "kind": "scenario_sweep"})
        return {"job_id": "job1"}
    return [Tool("update_draft", "patch", {"type": "object", "properties": {"ops": {"type": "array"}}}, lambda s, ops: s.draft.patch(ops, s.ledger)),
            Tool("propose_experiments", "propose", OBJ, lambda s: {"draft_hash": s.propose("configuration").draft_hash}),
            Tool("run_experiments", "run", OBJ, run)]


def scenario(name: str, second_turn: str, confirmed_how, after: list) -> dict:
    script = [_reply("", _call("set_mode", mode="configuration")), _reply("", _call("update_draft", ops=READY_OPS)), _reply("", _call("propose_experiments")),
              _reply("¿Lanzo 1 experimento?"), *after]
    a = Agent(llm.ReplayClient({"format": 1, "exchanges": script}, strict=False), _tools())
    a.user_turn("simula el motor DC con PI a 100 rad/s")
    a.user_turn(second_turn, confirmed_how=confirmed_how)
    refused = [e["reason"] for e in a.trace.events if e["kind"] == "tool_refused"]
    return {"scenario": name, "user_reply": second_turn, "interface_confirmation": confirmed_how, "launched": bool(a.state.jobs), "refusal": refused[0] if refused else None}


def states() -> list[dict]:
    rows = []
    s = policy.SessionState()
    rows.append(("undetermined, empty draft", policy.visible_tools(s)))
    s.mode = "question"
    rows.append(("question", policy.visible_tools(s)))
    s.mode = "configuration"
    rows.append(("configuration, draft not ready", policy.visible_tools(s)))
    led = Ledger()
    led.add_user_message(1, "100 rad/s")
    s.draft.patch(READY_OPS, led)
    rows.append(("configuration, draft ready", policy.visible_tools(s)))
    s.propose("configuration")
    rows.append(("configuration, launch proposed (same draft)", policy.visible_tools(s)))
    s.jobs.append({"job_id": "j", "kind": "scenario_sweep"})
    rows.append(("configuration, a job exists", policy.visible_tools(s)))
    s2 = policy.SessionState(mode="objective")
    rows.append(("objective, no objective yet", policy.visible_tools(s2)))
    return [{"state": k, "tools": v} for k, v in rows]


def main() -> None:
    table = states()
    run = [_reply("", _call("run_experiments")), _reply("fin")]
    scenarios = [
        scenario("user says yes, interface marks it", "sí", "typed: sí", run),
        scenario("user asks something else, model tries to launch", "¿cuánto tarda?", None, run),
        scenario("model claims the user confirmed", "perfecto", None, [_reply("El usuario confirmó: sí", _call("run_experiments")), _reply("fin")]),
        scenario("draft changed in the same turn as the yes", "sí, y 2 s", "typed",
                 [_reply("", _call("update_draft", ops=[{"op": "set", "field": "duration_s", "value": 2, "unit": "s"}]), _call("run_experiments")), _reply("fin")]),
    ]
    result = {"command": "python experiments/report_f1_3_loop_policy.py", "llm": "none (scripted replies)", "tools_by_state": table, "confirmation_scenarios": scenarios}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# F1.3 — bounded tool loop, confirmation policy, tools by state",
        "",
        "**No language model involved:** the model is a scripted `ReplayClient`; tools are stand-ins except `update_draft` (real `Draft.patch`).",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f1_3_loop_policy.json`](f1_3_loop_policy.json). Tests: `tests/assistant/test_assistant_loop.py`.",
        "- **Code:** `src/driveflow/assistant/core/loop.py` (`Agent`, `Tool`, `AgentTrace`, limits), `src/driveflow/assistant/core/policy.py` (`SessionState`, `visible_tools`, `check_call`, `confirm`, `is_affirmative`).",
        "- **Limits** (configs/assistant.yaml): tool calls, model calls and tokens per turn; three failed tool calls in a row end the turn with a message. Errors (invalid JSON, refused or failing tools) always go back to the model.",
        "",
        "## Tools the model sees, by state (C2)",
        "",
        "| State | Visible tools |",
        "|---|---|",
    ] + [f"| {r['state']} | {', '.join('`' + t + '`' for t in r['tools'])} |" for r in table] + [
        "",
        "Tools named here but built in later tasks (`explain_spade`, `search_docs` in F1.7; the configuration and objective tools in F1.5–F1.6) are only sent to the model once they exist in the registry.",
        "",
        "## Confirmation scenarios",
        "",
        "| Scenario | User's reply | Interface mark | Launched | Refusal sent back to the model |",
        "|---|---|---|---|---|",
    ] + [f"| {s['scenario']} | {s['user_reply']} | {s['interface_confirmation'] or '—'} | **{'yes' if s['launched'] else 'no'}** | {s['refusal'] or '—'} |" for s in scenarios] + [
        "",
        "Only the interface layer sets the confirmation mark (`SessionState.confirm`, called by the CLI / page when the user presses Run or types a bare yes right after a proposal); "
        "no tool can set it, and the mark is bound to the user's latest message and to the hash of the proposed draft.",
        "",
        "## Not executed",
        "",
        "- The loop with a real local model: no ejecutado (model chosen in F1.1).",
        "- F1.1's comparison of 'all tools visible' vs 'tools by state': no ejecutado yet (F1.1).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
