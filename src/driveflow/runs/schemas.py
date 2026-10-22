"""Data contracts for persistent experiments, runs, plant profiles and approvals -- expansion task E0
(TODO_EXPANSION_CLAUDE_CODE.md; docs/design/SPADE_GUI_Complete_Specification.md Sec. 2 and 3.7).

Every entity is a frozen dataclass with a schema version. ``to_dict()``/``from_dict()`` are the only
way in and out of storage, and ``from_dict()`` refuses a record written by a newer schema instead of
guessing. Validation happens here, in the backend: the GUI only builds proposals of these types.
The ``Job`` entity lives in driveflow.jobs.schemas, next to its state machine.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import uuid
from enum import Enum

SCHEMA_VERSION = 1

#: Provenance labels (spec Sec. 0, principle 2). "LIVE" is deliberately absent: nothing in SPADE
#: reads hardware telemetry, and a reachable PLECS RPC server is a tool, not a signal source.
ORIGINS = ("SIMULATED", "SIMULATED · PLECS", "SIMULATED · SURROGATE", "SIMULATED · OPENMODELICA", "EXTERNAL FILE", "MEASURED · EXTERNAL DATASET")

PROFILE_KINDS = ("native", "plecs_rpc_batch", "plecs_fmu_co_sim", "surrogate")
PROFILE_STATUSES = ("draft", "imported", "mapped", "specified", "verifying", "validated", "published", "failed", "unsupported", "stale", "disconnected")
#: A Run is the record of one execution, successful or not. "legacy_session" marks a record mirrored
#: from an interactive page's session log (viz/adapters/session.py): parameters only, no artifacts.
RUN_STATUSES = ("completed", "partially_completed", "failed", "cancelled", "legacy_session")
CONDITION_STATUSES = ("ok", "failed", "skipped")


class ErrorCode(str, Enum):
    INVALID_SPEC = "invalid_spec"
    NOT_FOUND = "not_found"
    SCHEMA_VERSION = "schema_version"
    INVALID_TRANSITION = "invalid_transition"
    UNKNOWN_JOB_KIND = "unknown_job_kind"
    WORKER_LOST = "worker_lost"
    HANDLER_ERROR = "handler_error"
    APPROVAL_BLOCKED = "approval_blocked"
    UNSUPPORTED = "unsupported"


class SpadeError(Exception):
    """A typed backend error; ``code`` is an ErrorCode, ``str(exc)`` a readable message."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = ErrorCode(code)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """Sortable, collision-safe id: <prefix>-<UTC timestamp>-<8 hex>."""
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj) -> str:
    """Full SHA-256 of the canonical JSON (the spec asks for the complete hash, not a prefix)."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SpadeError(ErrorCode.INVALID_SPEC, message)


def _json_safe(name: str, value) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise SpadeError(ErrorCode.INVALID_SPEC, f"{name} must be JSON-serialisable: {exc}") from None


class _Entity:
    """to_dict/from_dict with a schema-version check, shared by every entity below."""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        d["entity"] = type(self).__name__
        return d

    @classmethod
    def from_dict(cls, data: dict):
        data = dict(data)
        version = data.pop("schema_version", None)
        entity = data.pop("entity", cls.__name__)
        if entity != cls.__name__:
            raise SpadeError(ErrorCode.SCHEMA_VERSION, f"record is a {entity}, not a {cls.__name__}")
        if not isinstance(version, int) or version > SCHEMA_VERSION or version < 1:
            raise SpadeError(ErrorCode.SCHEMA_VERSION, f"{cls.__name__} schema_version {version!r} is not supported by this SPADE (knows 1..{SCHEMA_VERSION})")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise SpadeError(ErrorCode.SCHEMA_VERSION, f"{cls.__name__} record has unknown fields {unknown}")
        return cls._from_fields(data)

    @classmethod
    def _from_fields(cls, data: dict):
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class EnvironmentCapability(_Entity):
    """What a plant backend has been SHOWN to do (spec Sec. 3.7, 3.15). Defaults are all False:
    a capability is switched on by a concrete test, never by the file type."""

    can_simulate_batch: bool = False
    can_step_interactively: bool = False
    can_reset_episode: bool = False
    can_train_rl: bool = False
    can_serialize_config: bool = True
    validated_at: str | None = None
    diagnostic_messages: tuple = ()

    def __post_init__(self):
        for f in dataclasses.fields(self):
            if f.name.startswith("can_"):
                _require(isinstance(getattr(self, f.name), bool), f"EnvironmentCapability.{f.name} must be a bool")
        object.__setattr__(self, "diagnostic_messages", tuple(str(m) for m in self.diagnostic_messages))
        _require(not self.can_train_rl or (self.can_step_interactively and self.can_reset_episode),
                 "can_train_rl requires can_step_interactively and can_reset_episode")


@dataclasses.dataclass(frozen=True)
class PlantProfile(_Entity):
    profile_id: str
    kind: str
    engine: str
    display_name: str
    origin: str = "SIMULATED"
    file_hash: str | None = None
    signals: tuple = ()
    parameters: dict = dataclasses.field(default_factory=dict)
    action_space: dict | None = None
    observation_space: dict | None = None
    capabilities: EnvironmentCapability = dataclasses.field(default_factory=EnvironmentCapability)
    status: str = "draft"
    version: int = 1

    def __post_init__(self):
        _require(bool(self.profile_id), "PlantProfile.profile_id is required")
        _require(self.kind in PROFILE_KINDS, f"PlantProfile.kind must be one of {PROFILE_KINDS}, got {self.kind!r}")
        _require(self.origin in ORIGINS, f"PlantProfile.origin must be one of {ORIGINS}, got {self.origin!r}")
        _require(self.status in PROFILE_STATUSES, f"PlantProfile.status must be one of {PROFILE_STATUSES}, got {self.status!r}")
        _require(isinstance(self.version, int) and self.version >= 1, "PlantProfile.version must be an int >= 1")
        _require(self.kind == "native" or self.file_hash is not None, f"a {self.kind} profile needs the SHA-256 of its circuit/model file")
        object.__setattr__(self, "signals", tuple(self.signals))
        _json_safe("PlantProfile.parameters", self.parameters)

    @classmethod
    def _from_fields(cls, data):
        data["capabilities"] = EnvironmentCapability(**data.get("capabilities") or {})
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class ExperimentSpec(_Entity):
    """An immutable snapshot of what was asked to run (spec Sec. 18.2). ``kind`` names the job
    handler (driveflow.jobs.handlers); ``conditions`` is the frozen condition grid."""

    experiment_id: str
    kind: str
    plant_profile_id: str | None = None
    params: dict = dataclasses.field(default_factory=dict)
    conditions: tuple = ()
    seed: int | None = None
    controller_spec: dict | None = None
    objectives: tuple = ()
    constraints: tuple = ()
    algorithm: str | None = None
    budget: dict | None = None
    created_utc: str = dataclasses.field(default_factory=utc_now)

    def __post_init__(self):
        _require(bool(self.experiment_id), "ExperimentSpec.experiment_id is required")
        _require(bool(self.kind), "ExperimentSpec.kind is required")
        _require(self.seed is None or (isinstance(self.seed, int) and not isinstance(self.seed, bool)), "ExperimentSpec.seed must be an int or None")
        for name in ("conditions", "objectives", "constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("params", "conditions", "controller_spec", "objectives", "constraints", "budget"):
            _json_safe(f"ExperimentSpec.{name}", getattr(self, name))

    def config_hash(self) -> str:
        """Hash of everything that defines the computation -- not its id or creation time."""
        d = self.to_dict()
        for k in ("experiment_id", "created_utc", "schema_version", "entity"):
            d.pop(k)
        return config_hash(d)

    @classmethod
    def create(cls, kind: str, **fields) -> "ExperimentSpec":
        return cls(experiment_id=new_id("exp"), kind=kind, **fields)


@dataclasses.dataclass(frozen=True)
class Run(_Entity):
    """The record of one execution (spec Sec. 2). Failed and cancelled runs are kept with the same
    detail as successful ones; ``conditions`` holds the per-condition outcome."""

    run_id: str
    status: str
    origin: str
    kind: str
    experiment_id: str | None = None
    job_id: str | None = None
    started_utc: str | None = None
    completed_utc: str | None = None
    config_hash: str | None = None
    seed: int | None = None
    metrics: dict = dataclasses.field(default_factory=dict)
    conditions: tuple = ()
    artifacts: tuple = ()
    failure_reason: str | None = None
    command: str | None = None
    environment: dict = dataclasses.field(default_factory=dict)
    params: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        _require(bool(self.run_id), "Run.run_id is required")
        _require(self.status in RUN_STATUSES, f"Run.status must be one of {RUN_STATUSES}, got {self.status!r}")
        _require(self.origin in ORIGINS, f"Run.origin must be one of {ORIGINS}, got {self.origin!r}")
        _require(self.status != "failed" or bool(self.failure_reason), "a failed Run must say why (failure_reason)")
        object.__setattr__(self, "conditions", tuple(dict(c) for c in self.conditions))
        object.__setattr__(self, "artifacts", tuple(dict(a) for a in self.artifacts))
        for c in self.conditions:
            _require(c.get("status") in CONDITION_STATUSES, f"condition status must be one of {CONDITION_STATUSES}, got {c.get('status')!r}")
        for name in ("metrics", "conditions", "environment", "params"):
            _json_safe(f"Run.{name}", getattr(self, name))

    @property
    def failed_conditions(self) -> list:
        return [c for c in self.conditions if c["status"] == "failed"]


@dataclasses.dataclass(frozen=True)
class SignalSeries(_Entity):
    name: str
    unit: str
    run_id: str
    origin: str
    samples_ref: str
    t_ref: str | None = None
    hash: str | None = None
    model_ver: str | None = None
    confidence_metadata: dict | None = None
    source_uri: str | None = None
    license: str | None = None

    def __post_init__(self):
        _require(bool(self.name) and bool(self.run_id), "SignalSeries needs a name and the run_id it came from")
        _require(self.unit is not None, "SignalSeries.unit is required (use '' for dimensionless)")
        _require(self.origin in ORIGINS, f"SignalSeries.origin must be one of {ORIGINS}, got {self.origin!r}")


@dataclasses.dataclass(frozen=True)
class ApprovalRecord(_Entity):
    """A human decision on a candidate (spec Sec. 2 and 9, ApprovalGate). Stored append-only."""

    approval_id: str
    subject: str
    baseline_run: str | None
    candidate_run: str
    gate_report: dict
    approver: str
    user_confirmed_at: str
    decision: str
    config_before_hash: str | None = None
    config_after_hash: str | None = None

    def __post_init__(self):
        _require(bool(self.subject) and bool(self.candidate_run), "ApprovalRecord needs a subject and a candidate run")
        _require(self.decision in ("approved", "rejected"), f"ApprovalRecord.decision must be 'approved' or 'rejected', got {self.decision!r}")
        _require(bool(self.approver and self.approver.strip()), "ApprovalRecord.approver is required: approvals are made by a named person")
        _require(bool(self.user_confirmed_at), "ApprovalRecord.user_confirmed_at is required")
        _require(isinstance(self.gate_report, dict) and "blocking" in self.gate_report, "ApprovalRecord.gate_report must list its blocking items (may be empty)")
        _json_safe("ApprovalRecord.gate_report", self.gate_report)
