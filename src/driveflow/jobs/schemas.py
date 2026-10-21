"""Job entity and its state machine -- expansion task E0 (spec Sec. 6).

    queued -> preparing -> running -> cancelling -> completed | partially_completed | failed | cancelled

- ``resumable`` is an attribute, not a state, and is True only when a real checkpoint exists. No
  E0 handler writes checkpoints, so it is always False for now.
- ``cancelling`` means "cancellation requested": the worker confirms the stop between units of
  work. A solver call already in progress is not interrupted (spec Sec. 6, 19).
- A job whose worker vanished is closed as ``failed`` with ErrorCode.WORKER_LOST.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from driveflow.runs.schemas import ErrorCode, SpadeError


class JobState(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = frozenset({JobState.COMPLETED, JobState.PARTIALLY_COMPLETED, JobState.FAILED, JobState.CANCELLED})
ACTIVE = frozenset({JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING})

_ENDINGS = {JobState.COMPLETED, JobState.PARTIALLY_COMPLETED, JobState.FAILED, JobState.CANCELLED}
TRANSITIONS = {
    JobState.QUEUED: {JobState.PREPARING, JobState.CANCELLED},
    JobState.PREPARING: {JobState.RUNNING, JobState.CANCELLING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.CANCELLING, JobState.COMPLETED, JobState.PARTIALLY_COMPLETED, JobState.FAILED},
    # A cancel request does not change what already happened: if the work finished before the
    # worker reached a cancellation point, the job ends as what it actually is.
    JobState.CANCELLING: set(_ENDINGS),
}


def check_transition(current: JobState, new: JobState) -> None:
    if JobState(new) not in TRANSITIONS.get(JobState(current), set()):
        raise SpadeError(ErrorCode.INVALID_TRANSITION, f"job cannot go from {JobState(current).value} to {JobState(new).value}")


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    experiment_id: str
    kind: str
    engine: str
    state: JobState
    queued_at: str
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    progress_done: int | None = None
    progress_total: int | None = None
    phase: str | None = None
    worker_id: str | None = None
    worker_pid: int | None = None
    cancel_requested: bool = False
    resumable: bool = False
    run_id: str | None = None
    error: str | None = None
    error_code: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    @property
    def progress_fraction(self) -> float | None:
        """None when progress is not measurable (indeterminate) -- never a guessed value."""
        if not self.progress_total:
            return None
        return min(1.0, max(0.0, (self.progress_done or 0) / self.progress_total))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["state"] = self.state.value
        return d
