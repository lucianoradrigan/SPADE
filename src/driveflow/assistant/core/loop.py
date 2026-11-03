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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from driveflow.assistant import events as ev
from driveflow.assistant.core.llm import LLMClient, ModelNotAvailable, RuntimeUnavailable, ToolCall
from driveflow.assistant.core.policy import MODES, SessionState, check_call, visible_tools

DEFAULT_TURN_LIMITS = {"max_tool_calls_per_turn": 12, "max_model_calls_per_turn": 16, "max_tokens_per_turn": 60_000}
MAX_CONSECUTIVE_ERRORS = 3
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
                 on_event: Callable[[ev.AgentEvent], None] | None = None):
        self.client = client
        self.tools = {t.name: t for t in [SET_MODE, *tools]}
        self.state = state or SessionState()
        self.limits = {**DEFAULT_TURN_LIMITS, **{k: v for k, v in (limits or {}).items() if k in DEFAULT_TURN_LIMITS}}
        self.seed = seed
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.trace = AgentTrace()
        self.on_event = on_event

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
    def _run(self) -> TurnResult:
        res = TurnResult(text="")
        errors_in_row = 0
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
                res.text = reply.text
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
        res.stopped, res.text = why, STOP_MESSAGES[why]
        self.messages.append({"role": "assistant", "content": res.text})
        self.trace.add("stopped", reason=why)
        return res
