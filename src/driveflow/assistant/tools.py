"""The assistant's tools for configuration mode (task F1.5). Each one is a thin wrapper over
existing SPADE code, with a strict JSON schema; the loop (core/loop.py) decides when the model may
see and call them (core/policy.py).

| Tool | Wraps | Effect |
|---|---|---|
| recommend_plant | configs/plant_catalog.yaml plants + keyword table | read |
| describe_plant | datagen.plant_catalog.describe_plant (compacted) | read |
| suggest_value | defaults and ratings read from the code (internal sources only, P5) | read |
| update_draft | assistant.draft.Draft.patch | draft |
| validate_draft / estimate_cost | Draft.status (Scenario validation, configs/sim_cost.yaml) | read |
| propose_experiments | freezes the draft hash the user will confirm (policy) | proposal |
| run_experiments | ExperimentSpec("scenario_sweep") -> JobService.submit; needs the user's confirmation | runs |
| get_job_status | JobService.get / events | read |
| summarize_results | the run's parquet files + validation.step_metrics; every number SIMULATED | read |

Numbers in results are SI, with the unit next to them.
"""

from __future__ import annotations

import math

import numpy as np

from driveflow.assistant.core.loop import Tool, ToolFailure
from driveflow.assistant.core.policy import SessionState
from driveflow.assistant.draft import NATIVE_PLANTS, PatchError, _catalog, field_meta

AI_ORIGIN = "SIMULATED · AI-PLANNED"  # = jobs.handlers.AI_PLANNED
RPM_TO_RAD_S = 2 * math.pi / 60
OBJ_EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}

# --------------------------------------------------------------------------------- recommend_plant
#: (keywords, plant or None, note). Matched on the lower-cased description; first match wins per
#: plant. A None plant means "not simulable in SPADE today", with the closest alternative.
_KEYWORDS = [
    (("pmg 132", "pmg132", "perm-motor", "perm motor"), "dc_pmg132_v1", "the Perm-Motor PMG 132 with its datasheet values at 60 V"),
    (("motor dc", "dc motor", "motor de corriente continua", "corriente continua", "imanes permanentes dc", "permanent magnet dc", "pmdc", "motor cc", "brushed"), "dc_perm_ex_v1",
     "permanently excited DC motor, one-quadrant converter, speed PI or MPC; motor, supply, limits, ratings and load are configurable"),
    (("pmsm", "síncrono", "sincrono", "synchronous", "brushless", "bldc", "imanes permanentes ac", "servo"), "pmsm_v1",
     "salient PMSM with dq current control near standstill (current-step bench), MTPA or i_d = 0; not a speed loop"),
    (("vsc", "inversor", "inverter", "voltage source converter", "convertidor trifásico", "convertidor trifasico", "dpc", "filtro lc", "lc filter"), "vsc_dpc_v1",
     "voltage source converter with L-C filter under the trained DPC neural controller (trained at 50 V, 50 Hz, 8 ohm)"),
    (("buck", "reductor", "step-down", "step down", "dc-dc", "dc/dc", "convertidor cc-cc"), "buck_v1", "averaged synchronous buck converter with a PI cascade"),
]
_NOT_SIMULABLE = [
    (("inducción", "induccion", "induction", "asíncrono", "asincrono", "asynchronous", "jaula de ardilla", "squirrel"),
     "induction motors are not simulated in SPADE today (planned after phase 2, decision D3)", ["pmsm_v1", "dc_perm_ex_v1"]),
    (("boost", "elevador", "step-up", "flyback", "forward converter", "buck-boost", "ćuk", "cuk"),
     "only the buck (step-down) DC-DC converter exists in SPADE", ["buck_v1"]),
    (("reluctancia", "reluctance", "srm", "stepper", "paso a paso", "universal"),
     "this machine type is not simulated in SPADE", ["dc_perm_ex_v1", "pmsm_v1"]),
]


def recommend_plant(state: SessionState, description: str) -> dict:
    text = description.lower()
    blocked = [(why, alts) for kws, why, alts in _NOT_SIMULABLE if any(k in text for k in kws)]
    matches, seen = [], set()
    for kws, plant, note in _KEYWORDS:
        if plant not in seen and any(k in text for k in kws):
            seen.add(plant)
            cat = _catalog(plant)
            matches.append({"plant": plant, "title": cat["title"], "controllers": cat["controllers"], "why": note})
    if "dc_pmg132_v1" in seen and "dc_perm_ex_v1" in seen:
        matches = [m for m in matches if m["plant"] != "dc_perm_ex_v1"]
    out: dict = {"matches": matches}
    if blocked:
        why, alts = blocked[0]
        out["not_simulable"] = {"reason": why, "closest_substitutes": [{"plant": p, "title": _catalog(p)["title"]} for p in alts],
                                "note": "a substitute must be labelled as such to the user"}
    if not matches and not blocked:
        out["note"] = "no plant recognised in the description; ask the user, listing the plants"
        out["plants"] = [{"plant": p, "title": _catalog(p)["title"]} for p in NATIVE_PLANTS]
    return out


# ---------------------------------------------------------------------------------- describe_plant
def describe_plant(state: SessionState, plant: str) -> dict:
    if plant not in NATIVE_PLANTS:
        raise ToolFailure(f"unknown plant {plant!r}; plants: {list(NATIVE_PLANTS)}")
    cat = _catalog(plant)
    fields = []
    for e in cat["fields"]:
        if e["name"] in ("scenario_id", "plant_config_id", "fault_type", "electrical_severity", "mechanical_severity", "fault_order_override", "extra_faults"):
            continue  # generated by the draft, or faults (phase 4)
        f = {k: e[k] for k in ("name", "unit", "importance", "default", "default_source", "range", "description") if e.get(k) is not None}
        if e.get("convention"):
            f["convention"] = e["convention"]
        if "allowed" in e:
            f["allowed"] = e["allowed"]
        if "keys" in e:
            f["keys"] = {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in e["keys"].items()}
        fields.append(f)
    out = {"plant": plant, "title": cat["title"], "controllers": cat["controllers"], "fields": fields}
    if "stock_motor" in cat:
        sm = cat["stock_motor"]
        out["stock_motor"] = {"motor_parameter": sm["motor_parameter"], "nominal_values": sm["nominal_values"], "limits": sm["limits"], "supply_v": sm["supply_v"],
                              "load": sm["load"], "design_check_PI": sm["controller_design_check"]["PI"]["status"], "design_check_MPC": sm["controller_design_check"]["MPC"]["status"]}
    return out


# ----------------------------------------------------------------------------------- suggest_value
def _rated(state: SessionState, key: str) -> tuple[float, str] | None:
    """A rated value of the draft's DC motor (given by the user, or the stock motor's if unchanged)."""
    d = state.draft
    v = d.preset.get(f"dc_nominal_overrides.{key}")
    if v is not None and v.passthrough:
        return d._si(f"dc_nominal_overrides.{key}", v), f"dc_nominal_overrides.{key} of the draft ({v.origin})"
    if any(p.startswith("motor_parameter_overrides.") for p in d.preset):
        return None
    sm = _catalog(d.plant).get("stock_motor")
    if sm is None:
        return None
    return float(sm["nominal_values"][key]), f"rated {key} of the {d.plant} stock motor (describe_plant)"


def suggest_value(state: SessionState, field: str) -> dict:
    """A value with an INTERNAL, verifiable source only (decision P5): the plant's default, or the
    motor's rated value for an operating point. Anything else: no suggestion, ask the user."""
    d = state.draft
    if d.plant is None:
        raise ToolFailure("set the plant first")
    try:
        meta = field_meta(d.plant, field)
    except PatchError as exc:
        raise ToolFailure(str(exc)) from None
    out = {"field": field, "unit": meta["unit"]}
    if d.plant in ("dc_perm_ex_v1", "dc_pmg132_v1"):
        rated = {"omega_ref_rad_s": ("omega", "the motor's rated speed"), "load_torque_nm": ("torque", "the motor's rated torque (100 % load)")}.get(field)
        if rated:
            r = _rated(state, rated[0])
            if r is not None:
                return {**out, "value": r[0], "rule": rated[1], "source": r[1]}
    if "." in field:
        # A parameter, limit or rating of the machine: the stock motor's value is not a suggestion for
        # another motor. Keeping the stock motor is the use_defaults operation, not a suggested number.
        return {**out, "value": None, "why": "machine parameters are not suggested; ask the user, or keep the stock motor with use_defaults"}
    d_ = meta.get("default")
    if isinstance(d_, (int, float)) and not isinstance(d_, bool):
        return {**out, "value": float(d_), "rule": "SPADE's default for this field", "source": f"describe_plant({d.plant}): {meta.get('default_source')}"}
    return {**out, "value": None, "why": "no verifiable internal source for a suggestion; ask the user without suggesting a number (phase 1, P5)"}


# ------------------------------------------------------------------------------------ draft tools
def update_draft(state: SessionState, ops: list) -> dict:
    try:
        r = state.draft.patch(ops, state.ledger)
    except PatchError as exc:
        raise ToolFailure(str(exc)) from None
    return r


def validate_draft(state: SessionState) -> dict:
    return state.draft.status()


def estimate_cost(state: SessionState) -> dict:
    st = state.draft.status()
    return {"experiments": st["experiments"], "cost": st["cost"], "limits": st["limits"]}


def _experiment_table(state: SessionState) -> list:
    rows = []
    for kw in state.draft.conditions():
        rows.append({k: v for k, v in kw.items() if k not in ("plant_config_id",)})
    return rows


def propose_experiments(state: SessionState) -> dict:
    st = state.draft.status()
    if not st["ready"]:
        raise ToolFailure("the draft is not ready; see validate_draft")
    p = state.propose("configuration")
    return {"proposal": "show this to the user and ask for confirmation; do not launch before their yes",
            "plant": state.draft.plant, "controller_type": state.draft.controller_type or "default", "experiments": _experiment_table(state),
            "derived": st["derived"], "assumptions": state.draft.assumptions, "defaults_used": st["defaults_used"], "important_left_at_default": st["missing_important"],
            "cost": st["cost"], "draft_hash": p.draft_hash, "origin": "results will be SIMULATED"}


def run_experiments(state: SessionState) -> dict:
    from driveflow.jobs.service import JobService
    from driveflow.runs.profiles import native_profile_for
    from driveflow.runs.schemas import ExperimentSpec, SpadeError

    st = state.draft.status()
    if not st["ready"]:
        raise ToolFailure("the draft is not ready; see validate_draft")
    profile = native_profile_for(state.draft.plant)
    spec = ExperimentSpec.create("scenario_sweep", plant_profile_id=profile.profile_id if profile else None, conditions=state.draft.conditions(),
                                 params={"origin": AI_ORIGIN})
    try:
        job = JobService().submit(spec)
    except SpadeError as exc:
        raise ToolFailure(f"{exc.code.value}: {exc}") from None
    state.jobs.append({"job_id": job.job_id, "kind": "scenario_sweep", "experiment_id": spec.experiment_id, "status": job.state.value,
                       "draft_hash": state.draft.hash(), "n_conditions": len(spec.conditions),
                       # F1.8: what was launched, exactly, and the user message that confirmed it
                       "draft_snapshot": state.draft.to_dict(), "confirmed_by_message_seq": state.confirmation.message_seq if state.confirmation else None})
    state.proposal, state.confirmation = None, None  # one confirmation = one launch
    return {"job_id": job.job_id, "experiment_id": spec.experiment_id, "n_conditions": len(spec.conditions),
            "note": "queued; a SPADE worker runs it (Experiment Runs page or `python -m driveflow.jobs.worker`)"}


def get_job_status(state: SessionState, job_id: str | None = None) -> dict:
    from driveflow.jobs.service import JobService

    svc = JobService()
    entry = _job_entry(state, job_id)
    job = svc.get(entry["job_id"])
    st = job.state.value
    entry["status"] = st
    out = {"job_id": job.job_id, "state": st, "progress": {"done": job.progress_done, "total": job.progress_total, "phase": job.phase}}
    if job.error:
        out["error"] = job.error
    if job.terminal:
        run = svc.run_for(job)
        if run is not None:
            entry["run_id"] = run.run_id
            out.update({"run_id": run.run_id, "run_status": run.status, "conditions_ok": sum(c["status"] == "ok" for c in run.conditions),
                        "conditions_failed": [{"index": c["index"], "error": c.get("error")} for c in run.failed_conditions]})
    return out


def refresh_jobs(state: SessionState) -> list:
    """Read the state of every job of the session from the job service and store it in its entry
    (the policy shows the result tools only for terminal jobs). Returns the transitions as
    [{"job_id", "kind", "from", "to", "done", "total", "run_id"}]. A job the service cannot read
    keeps its last state."""
    if not state.jobs:
        return []
    from driveflow.jobs.service import JobService

    svc, out = JobService(), []
    for entry in state.jobs:
        try:
            job = svc.get(entry["job_id"])
        except Exception:  # noqa: BLE001 -- a missing record must not break the turn
            continue
        old, new = entry.get("status"), job.state.value
        if job.terminal and "run_id" not in entry:
            run = svc.run_for(job)
            if run is not None:
                entry["run_id"] = run.run_id
        entry["status"] = new
        if old != new:
            out.append({"job_id": job.job_id, "kind": entry.get("kind"), "from": old, "to": new, "done": job.progress_done, "total": job.progress_total,
                        "run_id": entry.get("run_id")})
    return out


def _job_entry(state: SessionState, job_id: str | None) -> dict:
    if not state.jobs:
        raise ToolFailure("no job was launched in this session")
    if job_id is None:
        return state.jobs[-1]
    for j in state.jobs:
        if j["job_id"] == job_id:
            return j
    raise ToolFailure(f"job {job_id!r} was not launched in this session; jobs: {[j['job_id'] for j in state.jobs]}")


# ------------------------------------------------------------------------------- summarize_results
def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)


def _step(t, y, ref, y0) -> dict:
    from driveflow.validation.step_metrics import step_response_metrics

    if ref == y0:
        return {"note": "no step (reference equals the initial value)"}
    m = step_response_metrics(t, y, ref=ref, y0=y0).as_dict()
    return {k: _f(v) if not isinstance(v, bool) else v for k, v in m.items()}


def condition_kpis(df, condition: dict) -> dict:
    """KPIs of one condition from its parquet (SPADE's own definitions, validation.step_metrics)."""
    plant = condition.get("plant_config_id", "dc_perm_ex_v1")
    t = df["timestamp_s"].to_numpy()
    tail = slice(int(len(df) * 0.9), None)
    if plant in ("dc_perm_ex_v1", "dc_pmg132_v1"):
        w = df["rpm"].to_numpy() * RPM_TO_RAD_S
        i = df["current_r"].to_numpy()
        out = {"final_speed_rad_s": _f(w[tail].mean()), "steady_current_a": _f(i[tail].mean()), "peak_current_a": _f(np.max(np.abs(i))),
               "peak_voltage_v": _f(df["voltage_v"].max()), "steady_torque_nm": _f(df["torque_nm"].to_numpy()[tail].mean())}
        if condition.get("torque_ref_nm") is None:
            from driveflow.datagen.scenario import Scenario

            ref = float(condition.get("omega_ref_rad_s", Scenario.omega_ref_rad_s))
            out["speed_reference_rad_s"] = ref
            out["speed_step"] = _step(t, w, ref, float(w[0]))
            out["speed_step"]["units"] = "times s, errors rad/s, overshoot % of the step"
        return out
    if plant == "pmsm_v1":
        iq, iq_ref = df["i_q"].to_numpy(), df["i_q_ref"].to_numpy()
        out = {"i_q_reference_a": _f(iq_ref[-1]), "i_d_reference_a": _f(df["i_d_ref"].to_numpy()[-1]), "final_i_q_a": _f(iq[tail].mean()),
               "final_i_d_a": _f(df["i_d"].to_numpy()[tail].mean()), "steady_torque_nm": _f(df["torque_nm"].to_numpy()[tail].mean())}
        out["i_q_step"] = _step(t, iq, float(iq_ref[-1]), float(iq[0]))
        out["i_q_step"]["units"] = "times s, errors A, overshoot % of the step"
        return out
    if plant == "buck_v1":
        v, ref = df["buck_v_out"].to_numpy(), df["buck_v_ref"].to_numpy()
        out = {"voltage_reference_v": _f(ref[-1]), "final_voltage_v": _f(v[tail].mean()), "peak_inductor_current_a": _f(np.max(np.abs(df["buck_i_l"].to_numpy()))),
               "rms_voltage_error_second_half_v": _f(np.sqrt(np.mean((v[len(v) // 2:] - ref[len(v) // 2:]) ** 2)))}
        out["voltage_step"] = _step(t, v, float(ref[0]), float(v[0]))
        out["voltage_step"]["units"] = "times s, errors V, overshoot % of the step"
        return out
    if plant == "vsc_dpc_v1":
        h = len(df) // 2
        err = np.hypot(df["vc_real"].to_numpy()[h:] - df["v_ref_real"].to_numpy()[h:], df["vc_imag"].to_numpy()[h:] - df["v_ref_imag"].to_numpy()[h:])
        return {"rms_voltage_error_second_half_v": _f(np.sqrt(np.mean(err ** 2))), "max_voltage_error_second_half_v": _f(err.max()),
                "reference_magnitude_v": _f(np.hypot(df["v_ref_real"].to_numpy()[-1], df["v_ref_imag"].to_numpy()[-1]))}
    return {"note": f"no KPI definition for {plant}"}


def summarize_results(state: SessionState, job_id: str | None = None) -> dict:
    import pandas as pd

    from driveflow.jobs.service import JobService

    svc = JobService()
    entry = _job_entry(state, job_id)
    job = svc.get(entry["job_id"])
    if not job.terminal:
        raise ToolFailure(f"job {job.job_id} has not finished (state {job.state.value}); check get_job_status later")
    run = svc.run_for(job)
    if run is None:
        raise ToolFailure(f"job {job.job_id} finished without a run record")
    rows = []
    for c in run.conditions:
        if c["status"] != "ok":
            rows.append({"index": c["index"], "status": c["status"], "error": c.get("error")})
            continue
        df = pd.read_parquet(svc.store.artifact_path({"path": c["artifact"]}))
        rows.append({"index": c["index"], "status": "ok", "condition": c["condition"], "kpis": condition_kpis(df, c["condition"])})
    return {"origin": run.origin, "run_id": run.run_id, "run_status": run.status, "rows": rows,
            "note": "every number here is SIMULATED and computed by SPADE (validation.step_metrics); report them as they are, with units"}


# ------------------------------------------------------------------------------------- registry
def _schema(props: dict, required: list | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


_OP_SCHEMA = {
    "type": "object",
    "description": ("One draft operation. op: set_plant {plant} | set_controller {controller_type} | set {field, value, unit, origin} | "
                    "assume {field, value, unit, reason} | unset {field} | vary {field, values or range {start, stop, step}, unit, origin} | "
                    "combine {mode: product|zip} | use_defaults {fields}. Send the user's number with the unit they wrote (e.g. 1500 rpm, 20 %); "
                    "origin is 'user', 'suggested' (from suggest_value) or 'default'. Never compute or round numbers yourself."),
    "properties": {"op": {"type": "string", "enum": ["set_plant", "set_controller", "set", "assume", "unset", "vary", "combine", "use_defaults"]},
                   "plant": {"type": "string"}, "controller_type": {"type": "string"}, "field": {"type": "string"},
                   "value": {"type": ["number", "string"]}, "unit": {"type": "string"}, "origin": {"type": "string", "enum": ["user", "suggested", "default"]},
                   "reason": {"type": "string"}, "values": {"type": "array", "items": {"type": "number"}},
                   "range": {"type": "object", "properties": {"start": {"type": "number"}, "stop": {"type": "number"}, "step": {"type": "number"}}},
                   "mode": {"type": "string", "enum": ["product", "zip"]}, "fields": {"type": "array", "items": {"type": "string"}}},
    "required": ["op"],
}

CONFIG_TOOLS = [
    Tool("recommend_plant", "Which SPADE plant fits what the user describes, or that it cannot be simulated and the closest substitute.",
         _schema({"description": {"type": "string", "description": "the user's description, in their words"}}, ["description"]), recommend_plant),
    Tool("describe_plant", "Fields a plant lets you configure: units, conventions, ranges, defaults, importance; for DC motors, the stock motor.",
         _schema({"plant": {"type": "string", "enum": list(NATIVE_PLANTS)}}, ["plant"]), describe_plant),
    Tool("suggest_value", "A suggested value for one field of the draft's plant, with its rule and source; value null means: ask the user without suggesting.",
         _schema({"field": {"type": "string", "description": "field name, or group.key like dc_nominal_overrides.torque"}}, ["field"]), suggest_value),
    Tool("update_draft", "Change the draft with a list of operations (applied all or none). Returns the draft status: what is missing, invalid, the cost.",
         _schema({"ops": {"type": "array", "items": _OP_SCHEMA, "minItems": 1}}, ["ops"]), update_draft),
    Tool("validate_draft", "Full status of the draft: missing required and important fields, defaults used, invalid experiments, derived values, cost, limits.",
         OBJ_EMPTY, validate_draft),
    Tool("estimate_cost", "Number of simulations and estimated CPU time of the draft, and whether it is within the limits.", OBJ_EMPTY, estimate_cost),
    Tool("propose_experiments", "Freeze the ready draft as the proposal: returns the experiment table to show the user. Then ask for their confirmation.",
         OBJ_EMPTY, propose_experiments),
    Tool("run_experiments", "Launch the proposed experiments as one job. Only works after the user confirmed the proposal.", OBJ_EMPTY, run_experiments),
    Tool("get_job_status", "Progress of a launched job (default: the last one).", _schema({"job_id": {"type": "string"}}), get_job_status),
    Tool("summarize_results", "KPIs per experiment of a finished job, computed by SPADE (SIMULATED).", _schema({"job_id": {"type": "string"}}), summarize_results),
]
