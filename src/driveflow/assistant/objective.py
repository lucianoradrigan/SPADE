"""Objective mode (a) of the assistant (task F1.6, decisions D4 / B7).

The user says what they want to ACHIEVE ("overshoot below 5 %, settling below 50 ms"). The assistant
does not build a new objective: it tunes the DC-motor speed PI with SPADE's fixed objective ``J`` or
``J_effort`` (``controller_tuning``, which accepts another base motor since F0.6 / 0b-D2) and then
CHECKS each of the user's thresholds on the separate EVALUATION grid, reporting met / not met with
the margin, for the baseline and every candidate. A constrained objective (``J_spec``) is phase 3.

Tools: ``describe_tunable``, ``update_objective``, ``propose_tuning``, ``run_tuning`` (needs the
user's confirmation, like ``run_experiments``) and ``check_requirements``.

Scope decisions (phase 1): only the DC-motor speed PI (``dc_perm_ex_v1`` with any motor, and
``dc_pmg132_v1`` as that motor's values) is driven from the chat. The buck PI (``buck_pi:*``) is
tunable in Tuning Studio but not from the chat yet; MPC, DPC and the PMSM FOC are not tunable.
"""

from __future__ import annotations

import json
import math

from driveflow.assistant import numbers as nb
from driveflow.assistant.core.loop import Tool, ToolFailure
from driveflow.assistant.core.policy import SessionState
from driveflow.assistant.tools import AI_ORIGIN, _job_entry, _schema

DC_PLANTS = ("dc_perm_ex_v1", "dc_pmg132_v1")
METHODS = ("pso", "bayes", "nsga2")
#: Tuning Studio's defaults for the DC-motor PI (viz/views/tuning_studio.py): first method of
#: tuning_adapter.METHOD_LABELS, 48 evaluations, seed 0. The objective default is J.
DEFAULTS = {"algorithm": "pso", "evaluations": 48, "seed": 0, "objective": "J"}
DEFAULTS_SOURCE = "Tuning Studio defaults for the DC-motor PI (viz/views/tuning_studio.py)"
EVALUATIONS_RANGE = (4, 2000)  # jobs.handlers._validate_tuning
#: metric -> (condition kind it is read on (None = every condition), row key, SI unit, factor row->SI, description)
METRICS = {
    "overshoot_pct": ("step", "overshoot_pct", "%", 1.0, "speed-step overshoot, % of the step"),
    "settling_time_s": ("step", "settling_ms", "s", 1e-3, "speed-step settling time, 2 % band"),
    "rise_time_s": ("step", "rise_ms", "s", 1e-3, "speed-step rise time, 10-90 %"),
    "speed_rmse_rad_s": ("step", "rmse_rad_s", "rad/s", 1.0, "speed RMSE over the step window"),
    "peak_current_a": (None, "peak_i_a", "A", 1.0, "peak armature current"),
    "load_dip_rad_s": ("load", "dip_rad_s", "rad/s", 1.0, "speed dip after a load step"),
    "load_final_error_rad_s": ("load", "final_error_rad_s", "rad/s", 1.0, "speed error at the end of a load step"),
}
OPS = ("<=", "<", ">=", ">")


# --------------------------------------------------------------------------------- the base motor
def motor_block(state: SessionState) -> dict | None:
    """params.motor of the tuning job (keys as Scenario) from the draft's machine; None = stock motor.
    dc_pmg132_v1 is passed as its datasheet values on dc_perm_ex_v1 (the tuning benches use that plant;
    tests/assistant/test_assistant_objective.py checks both give the same simulation)."""
    from driveflow.datagen.runner import PMG132_LIMITS, PMG132_MOTOR_PARAMETER, PMG132_NOMINAL_VALUES

    d = state.draft
    block: dict = {}
    if d.plant == "dc_pmg132_v1":
        block = {"motor_parameter_overrides": dict(PMG132_MOTOR_PARAMETER), "dc_limit_overrides": dict(PMG132_LIMITS), "dc_nominal_overrides": dict(PMG132_NOMINAL_VALUES)}
    for path, v in d.preset.items():
        if not v.passthrough:
            continue
        name, _, key = path.partition(".")
        if name not in ("motor_parameter_overrides", "dc_limit_overrides", "dc_nominal_overrides", "dc_supply_v"):
            continue
        x = d._si(path, v)
        if key:
            block.setdefault(name, {})[key] = x
        else:
            block[name] = x
    return block or None


def _check_tunable(state: SessionState) -> None:
    d = state.draft
    if d.plant not in DC_PLANTS:
        raise ToolFailure(f"only the DC-motor speed PI is tunable from the chat (plants {list(DC_PLANTS)}); for {d.plant}, use configuration mode")
    if d.controller_type == "MPC":
        raise ToolFailure("the MPC has no tunable gains in SPADE; tune the PI, or simulate the MPC in configuration mode")


# --------------------------------------------------------------------------------- describe_tunable
def describe_tunable(state: SessionState, system: str = "dc_motor_pi") -> dict:
    from driveflow.optim import objectives as ob

    if system != "dc_motor_pi":
        return {"system": system, "tunable_from_chat": False,
                "note": "only dc_motor_pi is tuned from the chat in phase 1; the buck PI is tunable in Tuning Studio; MPC, DPC and FOC are not tunable"}
    return {"system": "dc_motor_pi", "tunable_from_chat": True, "plants": list(DC_PLANTS),
            "parameters": [{"name": p, "range": list(b), "baseline": ob.BASELINE[p]} for p, b in zip(ob.PARAMS, ob.BOUNDS)],
            "algorithms": list(METHODS), "objectives": {k: v[3] for k, v in ob.OBJECTIVES.items()},
            "requirement_metrics": {k: {"unit": v[2], "measured_on": v[0] or "every condition", "meaning": v[4]} for k, v in METRICS.items()},
            "defaults": {**DEFAULTS, "source": DEFAULTS_SOURCE}, "evaluations_range": list(EVALUATIONS_RANGE),
            "how": "tunes on the tuning grid, then checks every requirement on the separate evaluation grid; results are candidates (no default changes)",
            "grids": "scaled to the base motor's rated speed and torque when the motor is not the stock one (0b-D2)"}


# --------------------------------------------------------------------------------- update_objective
def _complete(obj: dict) -> bool:
    return bool(obj.get("requirements")) and obj.get("objective") in ("J", "J_effort") and obj.get("algorithm") in METHODS and isinstance(obj.get("evaluations", {}).get("value"), int)


def update_objective(state: SessionState, requirements: list | None = None, objective: str | None = None, algorithm: str | None = None,
                     evaluations: int | None = None, evaluations_origin: str = "user", use_defaults: bool = False) -> dict:
    """Write the objective specification into the draft. Every threshold must be a quantity the user
    wrote (same no-invention rule as the draft); the budget comes from the user or from the defaults."""
    _check_tunable(state)
    obj = dict(state.draft.objective or {"system": "dc_motor_pi"})
    if use_defaults:
        for k in ("objective", "algorithm"):
            obj.setdefault(k, DEFAULTS[k])
        obj.setdefault("evaluations", {"value": DEFAULTS["evaluations"], "origin": "default", "source": DEFAULTS_SOURCE})
        obj.setdefault("seed", DEFAULTS["seed"])
    if objective is not None:
        if objective not in ("J", "J_effort"):
            raise ToolFailure("objective is 'J' or 'J_effort'")
        obj["objective"] = objective
    if algorithm is not None:
        if algorithm not in METHODS:
            raise ToolFailure(f"algorithm is one of {list(METHODS)}")
        obj["algorithm"] = algorithm
    if evaluations is not None:
        if not (isinstance(evaluations, int) and not isinstance(evaluations, bool) and EVALUATIONS_RANGE[0] <= evaluations <= EVALUATIONS_RANGE[1]):
            raise ToolFailure(f"evaluations must be an integer in {list(EVALUATIONS_RANGE)}")
        if evaluations_origin == "default":
            if evaluations != DEFAULTS["evaluations"]:
                raise ToolFailure(f"the default budget is {DEFAULTS['evaluations']} evaluations ({DEFAULTS_SOURCE})")
            obj["evaluations"] = {"value": evaluations, "origin": "default", "source": DEFAULTS_SOURCE}
        else:
            src = state.ledger.user_quantity(evaluations, None, None)
            if src is None:
                raise ToolFailure(f"{evaluations} evaluations: the user did not write this number; ask, or use the default ({DEFAULTS['evaluations']})")
            obj["evaluations"] = {"value": evaluations, "origin": "user", "source": src}
    if requirements is not None:
        reqs = []
        for i, r in enumerate(requirements):
            if not isinstance(r, dict) or r.get("metric") not in METRICS:
                raise ToolFailure(f"requirement {i}: metric must be one of {list(METRICS)}")
            if r.get("op") not in OPS:
                raise ToolFailure(f"requirement {i}: op must be one of {list(OPS)}")
            value, unit = r.get("value"), r.get("unit")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ToolFailure(f"requirement {i}: value must be a finite number")
            si_unit = METRICS[r["metric"]][2]
            try:
                si = value if unit == "%" and si_unit == "%" else nb.to_si(value, unit, si_unit)
            except nb.UnitError as exc:
                raise ToolFailure(f"requirement {i}: {exc}") from None
            src = state.ledger.user_quantity(value, unit, si_unit)
            if src is None:
                raise ToolFailure(f"requirement {i}: {value} {unit or ''}: the user did not write this threshold; ask them".replace("  ", " "))
            reqs.append({"metric": r["metric"], "op": r["op"], "value": value, "unit": unit, "value_si": si, "unit_si": si_unit, "origin": "user", "source": src})
        obj["requirements"] = reqs
    if obj.get("objective") == "J_effort":
        from driveflow.optim import objectives as ob

        try:
            ob.motor_from_spec(motor_block(state)).rated_current_a()
        except (ValueError, TypeError) as exc:
            raise ToolFailure(f"J_effort needs the motor's rated current: {exc}") from None
    obj["complete"] = _complete(obj)
    state.draft.objective = obj
    missing = [k for k in ("requirements", "objective", "algorithm", "evaluations") if not obj.get(k)]
    return {"objective_spec": obj, "complete": obj["complete"], "missing": missing,
            "next": "show the requirements back to the user in plain words with their numbers, then propose_tuning" if obj["complete"] else "fill what is missing"}


# --------------------------------------------------------------------------------- propose / run
def propose_tuning(state: SessionState) -> dict:
    _check_tunable(state)
    obj = state.draft.objective or {}
    if not obj.get("complete"):
        raise ToolFailure("the objective specification is not complete; see update_objective")
    p = state.propose("objective")
    return {"proposal": "show this to the user and ask for confirmation; do not launch before their yes",
            "plant": state.draft.plant, "base_motor": motor_block(state) or "stock motor", "tunes": "speed PI: bandwidth, damping, setpoint weight (full ranges)",
            "objective": obj["objective"], "algorithm": obj["algorithm"], "evaluations": obj["evaluations"], "seed": obj.get("seed", DEFAULTS["seed"]),
            "requirements": [{k: r[k] for k in ("metric", "op", "value", "unit")} for r in obj["requirements"]],
            "verification": "each requirement is checked on the evaluation grid (separate from the tuning grid); results are candidates", "draft_hash": p.draft_hash}


def run_tuning(state: SessionState) -> dict:
    from driveflow.jobs.service import JobService
    from driveflow.optim import objectives as ob
    from driveflow.runs.schemas import ExperimentSpec, SpadeError

    _check_tunable(state)
    obj = state.draft.objective or {}
    if not obj.get("complete"):
        raise ToolFailure("the objective specification is not complete")
    params = {"bounds": [list(b) for b in ob.BOUNDS], "objective": obj["objective"], "origin": AI_ORIGIN}
    block = motor_block(state)
    if block:
        params["motor"] = block
    spec = ExperimentSpec.create("controller_tuning", plant_profile_id="native:dc_perm_ex_v1", algorithm=obj["algorithm"], seed=int(obj.get("seed", 0)),
                                 params=params, budget={"evaluations": int(obj["evaluations"]["value"])})
    try:
        job = JobService().submit(spec)
    except SpadeError as exc:
        raise ToolFailure(f"{exc.code.value}: {exc}") from None
    state.jobs.append({"job_id": job.job_id, "kind": "controller_tuning", "experiment_id": spec.experiment_id, "status": job.state.value,
                       "draft_hash": state.draft.hash(), "requirements": obj["requirements"],
                       "draft_snapshot": state.draft.to_dict(), "confirmed_by_message_seq": state.confirmation.message_seq if state.confirmation else None})
    state.proposal, state.confirmation = None, None
    return {"job_id": job.job_id, "experiment_id": spec.experiment_id, "evaluations": obj["evaluations"]["value"],
            "note": "queued; a SPADE worker runs it. Tuning takes minutes: report progress with get_job_status"}


# --------------------------------------------------------------------------------- check_requirements
def _worst(rows: list, metric: str, op: str) -> tuple[float | None, list]:
    kind, key, _unit, factor, _ = METRICS[metric]
    vals, used = [], []
    for r in rows:
        if kind is not None and r.get("kind") != kind:
            continue
        v = r.get(key)
        x = math.inf if v is None else float(v) * factor  # never settled / no rise: fails an upper bound
        vals.append(x)
        used.append(r.get("condition"))
    if not vals:
        return None, used
    return (max(vals) if op in ("<=", "<") else min(vals)), used


def evaluate_requirements(requirements: list, rows: list) -> list:
    out = []
    for req in requirements:
        worst, used = _worst(rows, req["metric"], req["op"])
        thr = req["value_si"]
        if worst is None:
            out.append({"metric": req["metric"], "op": req["op"], "threshold": thr, "unit": req["unit_si"], "met": None, "note": "no condition of the evaluation grid measures this"})
            continue
        met = {"<=": worst <= thr, "<": worst < thr, ">=": worst >= thr, ">": worst > thr}[req["op"]]
        margin = (thr - worst) if req["op"] in ("<=", "<") else (worst - thr)
        out.append({"metric": req["metric"], "op": req["op"], "threshold": thr, "unit": req["unit_si"], "worst_case": None if math.isinf(worst) else worst,
                    "met": bool(met), "margin": None if math.isinf(margin) else margin, "conditions": used})
    return out


def check_requirements(state: SessionState, job_id: str | None = None) -> dict:
    from driveflow.jobs.service import JobService

    entry = _job_entry(state, job_id)
    if entry.get("kind") != "controller_tuning":
        raise ToolFailure(f"job {entry['job_id']} is not a tuning job")
    svc = JobService()
    job = svc.get(entry["job_id"])
    if not job.terminal:
        raise ToolFailure(f"the tuning job has not finished (state {job.state.value}); check get_job_status later")
    run = svc.run_for(job)
    if run is None:
        raise ToolFailure("the tuning job finished without a run record")
    art = next((a for a in run.artifacts if a.get("name") == "tuning_report.json" or str(a.get("path", "")).endswith("tuning_report.json")), None)
    if art is None:
        raise ToolFailure(f"the run has no tuning report (status {run.status}: {run.failure_reason})")
    report = json.loads(svc.store.artifact_path(art).read_text())
    reqs = entry["requirements"]
    rows = [{"label": "baseline", "params": report["baseline"]["params"], "checks": evaluate_requirements(reqs, report["baseline"]["evaluation_rows"])}]
    for c in report["candidates"]:
        rows.append({"label": c["label"], "params": c["params"], "checks": evaluate_requirements(reqs, c["evaluation_rows"])})
    for r in rows:
        r["n_met"] = sum(1 for c in r["checks"] if c["met"])
    cands = [r for r in rows if r["label"] != "baseline"]
    best = max(cands, key=lambda r: (r["n_met"], sum((c["margin"] or -1e9) / max(abs(c["threshold"]), 1e-9) for c in r["checks"] if c["met"] is not None)), default=None)
    all_met = best is not None and best["n_met"] == len(reqs)
    verdict = (f"candidate '{best['label']}' meets all {len(reqs)} requirements on the evaluation grid" if all_met else
               f"no candidate meets all {len(reqs)} requirements; best compromise '{best['label']}' meets {best['n_met']}" if best else "no candidate")
    return {"origin": run.origin, "run_id": run.run_id, "objective": report.get("objective_name"), "evaluation_grid": report.get("evaluation_grid"),
            "results": rows, "verdict": verdict, "all_met": all_met, "status": "candidates only; no default changes",
            "note": "SIMULATED; report met / not met and the margins as they are, including a negative result"}


# --------------------------------------------------------------------------------- registry
_REQ = {"type": "object", "properties": {"metric": {"type": "string", "enum": list(METRICS)}, "op": {"type": "string", "enum": list(OPS)},
                                         "value": {"type": "number"}, "unit": {"type": "string"}}, "required": ["metric", "op", "value", "unit"]}

OBJECTIVE_TOOLS = [
    Tool("describe_tunable", "What can be tuned from the chat: parameters and ranges, algorithms, objectives, requirement metrics and defaults.",
         _schema({"system": {"type": "string", "enum": ["dc_motor_pi", "buck_pi", "mpc", "dpc", "foc"]}}), describe_tunable),
    Tool("update_objective", "Write the objective specification: the user's requirements (each threshold with the unit the user wrote), objective J or J_effort, "
         "algorithm and budget. use_defaults fills objective, algorithm and budget with Tuning Studio's defaults.",
         _schema({"requirements": {"type": "array", "items": _REQ}, "objective": {"type": "string", "enum": ["J", "J_effort"]},
                  "algorithm": {"type": "string", "enum": list(METHODS)}, "evaluations": {"type": "integer"},
                  "evaluations_origin": {"type": "string", "enum": ["user", "default"]}, "use_defaults": {"type": "boolean"}}), update_objective),
    Tool("propose_tuning", "Freeze the complete objective specification as the proposal to show the user; then ask for confirmation.", _schema({}), propose_tuning),
    Tool("run_tuning", "Launch the proposed tuning job. Only works after the user confirmed the proposal.", _schema({}), run_tuning),
    Tool("check_requirements", "For a finished tuning job: met / not met and margin of every requirement, for the baseline and each candidate, on the evaluation grid.",
         _schema({"job_id": {"type": "string"}}), check_requirements),
]
