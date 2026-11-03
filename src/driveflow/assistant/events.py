"""Events of the assistant's loop (task FE-1, ESPEC_FRONTEND_PHD_RADRIGAN §2).

The loop emits these and the UI only translates them (viz/adapters/assistant_adapter.avatar_state),
so the UI does not depend on the model and UI tests run on recorded events. Each event is a frozen
dataclass with ``to_dict()``; ``type`` is its wire name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class AgentEvent:
    type: str = field(init=False, default="event")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TurnStarted(AgentEvent):
    message_seq: int = 0
    type: str = field(init=False, default="turn_started")


@dataclass(frozen=True)
class LlmCall(AgentEvent):
    phase: str = "start"  # start | end
    tokens: int | None = None
    type: str = field(init=False, default="llm_call")


@dataclass(frozen=True)
class ToolCallEvent(AgentEvent):
    phase: str = "start"  # start | end
    tool: str = ""
    ok: bool | None = None  # end only
    type: str = field(init=False, default="tool_call")


@dataclass(frozen=True)
class ProposalReady(AgentEvent):
    draft_hash: str = ""
    n_experiments: int | None = None
    est_wall_s: float | None = None
    type: str = field(init=False, default="proposal_ready")


@dataclass(frozen=True)
class UserConfirmed(AgentEvent):
    draft_hash: str = ""
    message_seq: int = 0
    type: str = field(init=False, default="user_confirmed")


@dataclass(frozen=True)
class JobEvent(AgentEvent):
    job_id: str = ""
    state: str = ""
    done: int | None = None
    total: int | None = None
    type: str = field(init=False, default="job")


@dataclass(frozen=True)
class ResultsReady(AgentEvent):
    run_id: str = ""
    type: str = field(init=False, default="results_ready")


@dataclass(frozen=True)
class ErrorEvent(AgentEvent):
    where: str = ""
    message: str = ""
    type: str = field(init=False, default="error")


@dataclass(frozen=True)
class RuntimeUnavailableEvent(AgentEvent):
    reason: str = ""
    type: str = field(init=False, default="runtime_unavailable")


@dataclass(frozen=True)
class TurnEnded(AgentEvent):
    stopped: str | None = None
    type: str = field(init=False, default="turn_ended")


EVENT_TYPES = ("turn_started", "llm_call", "tool_call", "proposal_ready", "user_confirmed", "job", "results_ready", "error", "runtime_unavailable", "turn_ended")
#: tools whose success means a proposal / a job / results (the loop emits the matching event)
PROPOSAL_TOOLS = ("propose_experiments", "propose_tuning")
LAUNCH_TOOLS = ("run_experiments", "run_tuning")
RESULT_TOOLS = ("summarize_results", "check_requirements")
