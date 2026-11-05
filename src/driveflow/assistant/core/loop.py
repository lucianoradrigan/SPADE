"""The bounded tool loop of the assistant and its trace (task F1.3).

One user turn: the model gets the conversation and ONLY the tools ``policy.visible_tools`` allows
now; every tool call goes through ``policy.check_call`` before it runs; every result (or error) goes
back to the model as a ``tool`` message. The turn ends when the model answers without calling a
tool, or when a limit of configs/assistant.yaml is hit:

- ``max_tool_calls_per_turn``, ``max_model_calls_per_turn``, ``max_tokens_per_turn``;
- three failed tool calls in a row (invalid JSON, refused by the policy, an exception in the tool):
  the turn ends with a message to the user instead of looping.

Errors are never dropped: the model sees them. ``AgentTrace`` keeps every message, call, result
and limit; ``draft_events`` is the record ``draft.audit_trace`` replays (no-invention check).

Loop guards (F1.1 round 2, agreed with Luciano 2026-11-04; designed by reading the F1.1 conversations,
docs/audit/assistant/f1_1_model_choice_round2_rule.md), each switchable in ``guards``:

- ``reject_repeats``: a call identical (tool + arguments) to an earlier one of the same turn is refused;
- ``per_turn_caps``: at most N calls of a tool per turn (search_docs 2, explain_spade 2);
- ``final_answer``: when a limit stops the turn, or the model ends with an empty text, one more model
  call WITHOUT tools asks it to answer with what it has; no turn ends with an empty answer;
- ``objective_needs_plant``: update_objective is hidden until the draft has a plant (policy);
- ``nudge_draft``: in configuration/objective mode, a text-only answer while the draft and objective
  did not change in the turn gets, once, a note to use update_draft / update_objective or to ask;
- ``nudge_docs``: when the user asks about something of SPADE (a finding ID, a page, a tool, "SPADE"),
  a text-only answer before any search_docs / explain_spade gets, once, a note to look it up; an answer
  that then cites no source gets, once, a note to cite it or to say "no lo tengo documentado".

Notes go to the model as a user-role message marked as coming from SPADE, never from the user; they
carry no numbers and are not evidence for the draft (the ledger only records real user messages).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from driveflow.assistant import events as ev
from driveflow.assistant.core.llm import LLMClient, ModelNotAvailable, RuntimeUnavailable, ToolCall
from driveflow.assistant.core.policy import MODES, SessionState, check_call, visible_tools

DEFAULT_TURN_LIMITS = {"max_tool_calls_per_turn": 12, "max_model_calls_per_turn": 16, "max_tokens_per_turn": 60_000}
MAX_CONSECUTIVE_ERRORS = 3
#: F1.1 round 2: the guards configs/assistant.yaml enables (``guards``). Agent(guards=None) keeps them all off.
DEFAULT_GUARDS = {"reject_repeats": True, "per_turn_caps": {"search_docs": 2, "explain_spade": 2}, "final_answer": True,
                  "objective_needs_plant": True, "nudge_draft": True, "nudge_docs": True}
GUARDS_OFF = {"reject_repeats": False, "per_turn_caps": {}, "final_answer": False, "objective_needs_plant": False, "nudge_draft": False, "nudge_docs": False}
NOTE_PREFIX = "[SPADE runtime note, not from the user] "
NOTES = {
    "final": "Stop calling tools. Answer the user now, in their language, with what you already have; say what is missing.",
    "empty": "Your last message to the user was empty. Answer the user now, in their language, in a few sentences.",
    "draft": "You answered without recording anything. If the user gave what you need, record it now with update_draft (configuration) or "
             "update_objective (objective); if something required is missing, ask the user for it.",
    "docs": "The user asked about SPADE. Look it up with search_docs or explain_spade before answering; answer only from what they return.",
    "cite": "Cite the source you used (file and section, page, or finding ID). If the tools returned nothing relevant, say \"no lo tengo documentado\".",
}
EMPTY_FALLBACK = "The assistant could not produce an answer; please rephrase. / El asistente no pudo generar una respuesta; reformula el pedido."
_SPADE_REF = re.compile(r"\b(S-\d+|AI-N\d+|AI-\d+|F\d\.\d+|FE-\d+|SPADE)\b", re.I)
_CITED = re.compile(r"(docs/|\.md\b|\.ya?ml\b|no lo tengo documentado|not documented|fuente|source|según|according to|p[aá]gina|page)", re.I)
STOP_MESSAGES = {
    "errors": "The assistant stopped after 3 failed tool calls in a row; please rephrase. / El asistente se detuvo tras 3 llamadas fallidas seguidas; reformula el pedido.",
    "max_tool_calls_per_turn": "The assistant reached its limit of tool calls for one turn. / El asistente llegó al límite de llamadas a herramientas en un turno.",
    "max_model_calls_per_turn": "The assistant reached its limit of steps for one turn. / El asistente llegó al límite de pasos en un turno.",
    "max_tokens_per_turn": "The assistant reached its token limit for one turn. / El asistente llegó al límite de tokens en un turno.",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments (an object)
    fn: Callable[..., dict]  # fn(state, **arguments) -> JSON-serialisable dict

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolFailure(Exception):
    """Raised by a tool for an expected refusal; its message goes back to the model."""


def _set_mode(state: SessionState, mode: str) -> dict:
    if mode not in MODES[1:]:
        raise ToolFailure(f"mode must be one of {list(MODES[1:])}")
    state.mode = mode
    return {"mode": mode, "tools_now": visible_tools(state)}


SET_MODE = Tool("set_mode", "Set what the user wants: 'configuration' (simulate something concrete), 'objective' (find controller settings that meet "
                "requirements) or 'question' (a question about SPADE). Call it as soon as it is clear; ask the user if it is not.",
                {"type": "object", "properties": {"mode": {"type": "string", "enum": list(MODES[1:])}}, "required": ["mode"], "additionalProperties": False}, _set_mode)


@dataclass
class TurnResult:
    text: str
    stopped: str | None = None  # None, "errors", or the name of the limit that stopped the turn
    model_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0


@dataclass
class AgentTrace:
    events: list = field(default_factory=list)
    draft_events: list = field(default_factory=list)
    agent_events: list = field(default_factory=list)  # FE-1: the AgentEvents of the session, as dicts

    def add(self, kind: str, **data) -> None:
        self.events.append({"kind": kind, **data})


class Agent:
    def __init__(self, client: LLMClient, tools: list, *, system_prompt: str = "", state: SessionState | None = None, limits: dict | None = None, seed: int = 0,
                 on_event: Callable[[ev.AgentEvent], None] | None = None, guards: dict | None = None):
        self.client = client
        self.tools = {t.name: t for t in [SET_MODE, *tools]}
        self.state = state or SessionState()
        self.limits = {**DEFAULT_TURN_LIMITS, **{k: v for k, v in (limits or {}).items() if k in DEFAULT_TURN_LIMITS}}
        self.seed = seed
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.trace = AgentTrace()
        self.on_event = on_event
        # Library default: guards off (the recorded cassettes of the tests predate them); the CLI and the
        # Assistant page pass configs/assistant.yaml ``guards`` (= DEFAULT_GUARDS).
        self.guards = {**GUARDS_OFF, **(guards or {})}
        self.state.objective_needs_plant = bool(self.guards["objective_needs_plant"])

    def _emit(self, event: ev.AgentEvent) -> None:
        self.trace.agent_events.append(event.to_dict())
        if self.on_event is not None:
            self.on_event(event)

    # --- interface layer ---------------------------------------------------------------------
    def user_turn(self, text: str, *, confirmed_how: str | None = None) -> TurnResult:
        """One user message. ``confirmed_how`` is set ONLY by the interface (Run button, or a yes
        typed right after a proposal); the model has no way to set it."""
        seq = self.state.next_seq()
        self.state.last_user_seq = seq
        self._emit(ev.TurnStarted(message_seq=seq))
        self.state.ledger.add_user_message(seq, text)
        self.messages.append({"role": "user", "content": text})
        self.trace.add("user", seq=seq, text=text)
        self.trace.draft_events.append({"role": "user", "seq": seq, "text": text})
        if confirmed_how is not None:
            ok = self.state.confirm(seq, confirmed_how)
            self.trace.add("confirmation", seq=seq, how=confirmed_how, recorded=ok)
            if ok:
                self._emit(ev.UserConfirmed(draft_hash=self.state.confirmation.draft_hash, message_seq=seq))
        try:
            res = self._run()
        except (RuntimeUnavailable, ModelNotAvailable) as exc:
            self._emit(ev.RuntimeUnavailableEvent(reason=str(exc)))
            self._emit(ev.TurnEnded(stopped="runtime_unavailable"))
            raise
        self._emit(ev.TurnEnded(stopped=res.stopped))
        return res

    # --- the loop ------------------------------------------------------------------------------
    def _spade_reference(self, text: str) -> bool:
        if _SPADE_REF.search(text or ""):
            return True
        low = (text or "").lower()
        return any(name in low for name in self.tools if "_" in name)

    def _note(self, key: str, res: TurnResult) -> None:
        self.messages.append({"role": "user", "content": NOTE_PREFIX + NOTES[key]})
        self.trace.add("note", key=key)

    def _run(self) -> TurnResult:
        res = TurnResult(text="")
        errors_in_row = 0
        self._turn_calls: dict = {}
        self._turn_counts: dict = {}
        start_hash, start_obj = self.state.draft.hash(), json.dumps(self.state.draft.objective, sort_keys=True, default=str)
        user_text = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")
        asks_spade = self.guards["nudge_docs"] and self._spade_reference(user_text)
        nudged: set = set()
        while True:
            if res.model_calls >= self.limits["max_model_calls_per_turn"]:
                return self._stop(res, "max_model_calls_per_turn")
            if res.tokens >= self.limits["max_tokens_per_turn"]:
                return self._stop(res, "max_tokens_per_turn")
            names = visible_tools(self.state)
            schemas = [self.tools[n].schema() for n in names if n in self.tools]
            self._emit(ev.LlmCall(phase="start"))
            reply = self.client.chat(self.messages, schemas, seed=self.seed)
            res.model_calls += 1
            res.tokens += (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
            self._emit(ev.LlmCall(phase="end", tokens=(reply.prompt_tokens or 0) + (reply.completion_tokens or 0)))
            self.messages.append(reply.to_message())
            self.trace.add("model", visible_tools=names, text=reply.text, tool_calls=[{"name": c.name, "arguments": c.arguments, "error": c.error} for c in reply.tool_calls],
                           prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens, latency_s=reply.latency_s)
            if not reply.tool_calls:
                looked_up = any(self._turn_counts.get(t) for t in ("search_docs", "explain_spade"))
                changed = self.state.draft.hash() != start_hash or json.dumps(self.state.draft.objective, sort_keys=True, default=str) != start_obj
                if self.guards["nudge_draft"] and "draft" not in nudged and self.state.mode in ("configuration", "objective") and not changed:
                    nudged.add("draft")
                    self._note("draft", res)
                    continue
                if asks_spade and not looked_up and "docs" not in nudged:
                    nudged.add("docs")
                    self._note("docs", res)
                    continue
                if asks_spade and looked_up and "cite" not in nudged and reply.text.strip() and not _CITED.search(reply.text):
                    nudged.add("cite")
                    self._note("cite", res)
                    continue
                if not reply.text.strip() and self.guards["final_answer"]:
                    return self._final_answer(res, "empty", stopped=None)
                res.text = reply.text if reply.text.strip() or not self.guards["final_answer"] else EMPTY_FALLBACK
                return res
            for i, call in enumerate(reply.tool_calls):
                if res.tool_calls >= self.limits["max_tool_calls_per_turn"]:
                    for skipped in reply.tool_calls[i:]:
                        self._tool_message(skipped, {"error": "not executed: tool-call limit of this turn reached"})
                    return self._stop(res, "max_tool_calls_per_turn")
                res.tool_calls += 1
                self._emit(ev.ToolCallEvent(phase="start", tool=call.name))
                result, failed = self._execute(call)
                self._emit(ev.ToolCallEvent(phase="end", tool=call.name, ok=not failed))
                self._emit_outcome(call.name, result, failed)
                self._tool_message(call, result)
                errors_in_row = errors_in_row + 1 if failed else 0
                if errors_in_row >= MAX_CONSECUTIVE_ERRORS:
                    for skipped in reply.tool_calls[i + 1:]:
                        self._tool_message(skipped, {"error": "not executed: the turn stopped after repeated errors"})
                    return self._stop(res, "errors")

    def _execute(self, call: ToolCall) -> tuple[dict, bool]:
        if call.error:
            out = {"error": f"{call.error}; send the arguments as a JSON object matching the tool's schema"}
            self.trace.add("tool_error", name=call.name, error=out["error"], raw_arguments=call.raw_arguments)
            return out, True
        if call.name not in self.tools:
            out = {"error": f"unknown tool {call.name!r}; available: {visible_tools(self.state)}"}
            self.trace.add("tool_error", name=call.name, error=out["error"])
            return out, True
        key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
        if self.guards["reject_repeats"] and key in getattr(self, "_turn_calls", {}):
            out = {"error": f"identical {call.name} call already made in this turn; use its result above instead of repeating it"}
            self.trace.add("tool_refused", name=call.name, arguments=call.arguments, reason="repeat")
            return out, True
        cap = (self.guards.get("per_turn_caps") or {}).get(call.name)
        counts = getattr(self, "_turn_counts", {})
        if cap is not None and counts.get(call.name, 0) >= cap:
            out = {"error": f"limit of {cap} {call.name} calls per turn reached; answer with what you already have"}
            self.trace.add("tool_refused", name=call.name, arguments=call.arguments, reason="cap")
            return out, True
        if hasattr(self, "_turn_calls"):
            self._turn_calls[key] = True
            counts[call.name] = counts.get(call.name, 0) + 1
        refusal = check_call(self.state, call.name)
        if refusal:
            self.trace.add("tool_refused", name=call.name, arguments=call.arguments, reason=refusal)
            return {"error": refusal}, True
        try:
            result = self.tools[call.name].fn(self.state, **call.arguments)
        except ToolFailure as exc:
            self.trace.add("tool_error", name=call.name, arguments=call.arguments, error=str(exc))
            self._record_patch(call, {"ok": False})
            return {"error": str(exc)}, True
        except TypeError as exc:  # wrong or missing arguments
            self.trace.add("tool_error", name=call.name, arguments=call.arguments, error=str(exc))
            return {"error": f"bad arguments: {exc}"}, True
        except Exception as exc:  # noqa: BLE001 -- a tool bug must not end the session; the model sees it
            self.trace.add("tool_error", name=call.name, arguments=call.arguments, error=f"{type(exc).__name__}: {exc}")
            return {"error": f"{type(exc).__name__}: {exc}"}, True
        self.state.ledger.add_tool_result(call.name, call.id, result)
        self.trace.add("tool_result", name=call.name, arguments=call.arguments, result=result)
        self.trace.draft_events.append({"role": "tool", "name": call.name, "call_id": call.id, "result": result})
        self._record_patch(call, result)
        failed = isinstance(result, dict) and result.get("ok") is False
        return result, failed

    def _emit_outcome(self, tool: str, result: dict, failed: bool) -> None:
        """The UI-level events a tool result implies (FE-1)."""
        if failed:
            msg = ""
            if isinstance(result, dict):
                msg = str(result.get("error") or (result.get("errors") or [{}])[0].get("error", ""))
            self._emit(ev.ErrorEvent(where=f"tool:{tool}", message=msg[:300]))
            return
        if not isinstance(result, dict):
            return
        if tool in ev.PROPOSAL_TOOLS:
            cost = result.get("cost") or {}
            n = len(result["experiments"]) if isinstance(result.get("experiments"), list) else None
            self._emit(ev.ProposalReady(draft_hash=str(result.get("draft_hash", "")), n_experiments=n, est_wall_s=cost.get("est_cpu_s")))
        elif tool in ev.LAUNCH_TOOLS:
            self._emit(ev.JobEvent(job_id=str(result.get("job_id", "")), state="queued", done=0, total=result.get("n_conditions") or result.get("evaluations")))
        elif tool == "get_job_status":
            p = result.get("progress") or {}
            self._emit(ev.JobEvent(job_id=str(result.get("job_id", "")), state=str(result.get("state", "")), done=p.get("done"), total=p.get("total")))
        elif tool in ev.RESULT_TOOLS:
            self._emit(ev.ResultsReady(run_id=str(result.get("run_id", ""))))

    def _record_patch(self, call: ToolCall, result: dict) -> None:
        if call.name == "update_draft" and isinstance(call.arguments, dict) and "ops" in call.arguments:
            # Inserted before the tool's own result so the replay sees the evidence in the same order.
            ev = {"role": "patch", "ops": call.arguments["ops"], "accepted": bool(result.get("ok"))}
            if self.trace.draft_events and self.trace.draft_events[-1].get("call_id") == call.id:
                self.trace.draft_events.insert(len(self.trace.draft_events) - 1, ev)
            else:
                self.trace.draft_events.append(ev)

    def _tool_message(self, call: ToolCall, result: dict) -> None:
        self.messages.append({"role": "tool", "tool_name": call.name, "content": json.dumps(result, ensure_ascii=False, default=str)})

    def _stop(self, res: TurnResult, why: str) -> TurnResult:
        if self.guards["final_answer"]:
            return self._final_answer(res, "final", stopped=why)
        res.stopped, res.text = why, STOP_MESSAGES[why]
        self.messages.append({"role": "assistant", "content": res.text})
        self.trace.add("stopped", reason=why)
        return res

    def _final_answer(self, res: TurnResult, note: str, stopped: str | None) -> TurnResult:
        """One model call WITHOUT tools so the user gets an answer with what the turn has (guard
        ``final_answer``); a still-empty reply falls back to a fixed message, never to nothing."""
        res.stopped = stopped
        if stopped:
            self.trace.add("stopped", reason=stopped)
        self._note(note, res)
        self._emit(ev.LlmCall(phase="start"))
        reply = self.client.chat(self.messages, [], seed=self.seed)
        res.model_calls += 1
        res.tokens += (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
        self._emit(ev.LlmCall(phase="end", tokens=(reply.prompt_tokens or 0) + (reply.completion_tokens or 0)))
        self.trace.add("model", visible_tools=[], text=reply.text, tool_calls=[], prompt_tokens=reply.prompt_tokens,
                       completion_tokens=reply.completion_tokens, latency_s=reply.latency_s, final=True)
        text = reply.text.strip() if not reply.tool_calls else ""
        res.text = text or (STOP_MESSAGES[stopped] if stopped else EMPTY_FALLBACK)
        self.messages.append({"role": "assistant", "content": res.text})
        return res
