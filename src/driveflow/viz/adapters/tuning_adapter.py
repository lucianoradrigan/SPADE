"""GUI boundary to controller tuning (expansion task E7; spec Sec. 4). Specs, jobs, results and the
approval record are built here; the view only passes plain values."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.optim import objectives as ob
from driveflow.optim import systems
from driveflow.runs.schemas import ApprovalRecord, ExperimentSpec, Run, new_id, utc_now
from driveflow.runs.store import RunStore

METHOD_LABELS = {"pso": "Particle swarm (PSO)", "bayes": "Bayesian optimization (GP)", "nsga2": "Multi-objective (NSGA-II)"}
PARAM_LABELS = {"speed_bandwidth_rad_s": ("Speed-loop bandwidth", "rad/s"), "speed_damping": ("Speed-loop damping", "–"), "setpoint_weight": ("Setpoint weight b", "–"),
                **systems.BUCK_PARAM_LABELS}
NOT_AUTHORIZED = "Use as default is not authorized in this round (decision D1, 2026-10-21): the result stays a candidate."
SUBJECT = "dc_motor PI speed loop (PICascadeController)"


def available_systems() -> dict:
    """Tunable systems found by capability (B6)."""
    try:
        return systems.available()
    except Exception:  # pragma: no cover -- the page must render
        return {systems.DC_MOTOR: "DC motor PI speed loop (native:dc_perm_ex_v1)"}


def get_system(key: str):
    return systems.get(key)


def run_system_key(rep: dict | None) -> str:
    return (rep or {}).get("system", systems.DC_MOTOR)


def grid_table(grid) -> pd.DataFrame:
    if grid and isinstance(grid[0], systems.BuckCondition):
        return pd.DataFrame([{"Condition": c.name, "Reference v_ref (V)": c.vout_ref_v, "Load R (Ω)": c.load_resistance_ohm, "Episode": "20 ms from rest, no load step"}
                             for c in grid])
    rows = []
    for c in grid:
        what = f"{c.step_rad_s:g} rad/s reference step" if c.kind == "step" else f"{c.load_nm:g} N·m load step"
        rows.append({"Condition": c.name, "Event": what, "Operating speed (rad/s)": c.omega_rad_s, "Plant": "drift (ψ −10 %, J +50 %)" if c.motor_parameter else "nominal"})
    return pd.DataFrame(rows)


def submit(method: str, bounds: list, evaluations: int, seed: int, system_key: str = systems.DC_MOTOR):
    if system_key != systems.DC_MOTOR:
        sysm = systems.get(system_key)
        spec = ExperimentSpec.create("controller_tuning", plant_profile_id=sysm.plant_profile_id, algorithm=method, seed=int(seed),
                                     params={"system": system_key, "bounds": [[float(lo), float(hi)] for lo, hi in bounds], "parameters": list(sysm.params)},
                                     controller_spec={"controller": "BuckVoltageController", "baseline": sysm.baseline},
                                     objectives=("J",), budget={"evaluations": int(evaluations)},
                                     conditions=tuple({"grid": "tuning", "name": c.name} for c in sysm.tuning_grid))
        return JobService().submit(spec)
    spec = ExperimentSpec.create("controller_tuning", plant_profile_id="native:dc_perm_ex_v1", algorithm=method, seed=int(seed),
                                 params={"bounds": [[float(lo), float(hi)] for lo, hi in bounds], "parameters": list(ob.PARAMS)},
                                 controller_spec={"controller": "PICascadeController", "baseline": ob.BASELINE},
                                 objectives=("mean_overshoot_pct", "mean_settling_ms", "mean_dip_rad_s"), budget={"evaluations": int(evaluations)},
                                 conditions=tuple({"grid": "tuning", "name": c.name} for c in ob.TUNING_GRID))
    return JobService().submit(spec)


def active_jobs() -> list[dict]:
    s = JobService()
    return [{"job_id": j.job_id, "state": j.state.value, "done": j.progress_done, "total": j.progress_total}
            for j in s.list([JobState.QUEUED, JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING]) if j.kind == "controller_tuning"]


def runs(limit: int = 50) -> list[Run]:
    return [r for r in RunStore().list_runs(500) if r.kind == "controller_tuning"][:limit]


def _artifact(run: Run, name: str) -> dict | None:
    art = next((a for a in run.artifacts if a["path"].endswith(name)), None)
    if art is None:
        return None
    path = RunStore().artifact_path(art)
    return json.loads(path.read_text()) if path.exists() else None


def report(run: Run) -> dict | None:
    return _artifact(run, "tuning_report.json")


def history(run: Run) -> pd.DataFrame | None:
    h = _artifact(run, "tuning_history.json")
    if h is None:
        return None
    obj = np.array(h["objective"], dtype=float)
    if obj.ndim == 2 and obj.shape[1] == 3:  # NSGA-II: the same scalar J, from the vector objective
        J = obj[:, 0] + 0.1 * obj[:, 1] + 2.0 * obj[:, 2]
    else:
        J = obj.reshape(len(obj), -1)[:, 0]
    df = pd.DataFrame(h["X"], columns=h["params"])
    df.insert(0, "evaluation", np.arange(1, len(df) + 1))
    df["J"] = J
    df["best J so far"] = np.minimum.accumulate(np.where(np.isfinite(J), J, np.inf))
    if obj.ndim == 2 and obj.shape[1] == 3:
        df["overshoot %"], df["settling ms"], df["load dip rad/s"] = obj[:, 0], obj[:, 1], obj[:, 2]
    return df


def comparison_table(rep: dict, candidate: dict) -> pd.DataFrame:
    if rep.get("system", systems.DC_MOTOR) != systems.DC_MOTOR:
        units = {"rmse_pct": "tracking RMSE (% of v_ref)", "overshoot_pct": "overshoot (% of v_ref)", "peak_current_a": "peak |i_L| (A)"}
        rows = [{"Evaluation condition": b["condition"], "Metric": label, "Baseline": b.get(k), "Candidate": c.get(k)}
                for b, c in zip(rep["baseline"]["evaluation_rows"], candidate["evaluation_rows"]) for k, label in units.items()]
        rows.append({"Evaluation condition": "all", "Metric": "J (objective)", "Baseline": rep["baseline"]["evaluation"]["J"], "Candidate": candidate["evaluation"]["J"]})
        return pd.DataFrame(rows)
    units = {"overshoot_pct": "overshoot (%)", "settling_ms": "2 % settling (ms)", "dip_rad_s": "load dip (rad/s)", "final_error_rad_s": "error 0.3 s after (rad/s)"}
    rows = []
    for b, c in zip(rep["baseline"]["evaluation_rows"], candidate["evaluation_rows"]):
        for k in (("overshoot_pct", "settling_ms") if b["kind"] == "step" else ("dip_rad_s", "final_error_rad_s")):
            rows.append({"Evaluation condition": b["condition"], "Metric": units[k], "Baseline": b.get(k), "Candidate": c.get(k)})
    rows.append({"Evaluation condition": "all", "Metric": "J (objective)", "Baseline": rep["baseline"]["evaluation"]["J"], "Candidate": candidate["evaluation"]["J"]})
    return pd.DataFrame(rows)


def gate_report(rep: dict, candidate: dict) -> dict:
    blocking = [NOT_AUTHORIZED]
    if candidate["evaluation"]["J"] >= rep["baseline"]["evaluation"]["J"]:
        blocking.append("The candidate does not improve the objective on the evaluation grid.")
    if rep.get("evaluation_reused_tuning_conditions"):
        blocking.append(ob.REUSED_WARNING)
    return {"blocking": blocking, "baseline_J_evaluation": rep["baseline"]["evaluation"]["J"], "candidate_J_evaluation": candidate["evaluation"]["J"],
            "candidate_params": candidate["params"]}


def _subject(run) -> str:
    """The tuned system named in the run's report (B6); the DC-motor subject for older runs."""
    try:
        return (report(run) or {}).get("subject", SUBJECT)
    except (AttributeError, OSError):
        return SUBJECT


def save_candidate(run: Run, candidate: dict) -> dict:
    """Stores the chosen candidate next to its run. Changes nothing else."""
    data = {"subject": _subject(run), "run_id": run.run_id, "candidate": candidate["label"], "params": candidate["params"], "saved_utc": utc_now(),
            "status": "candidate -- not a default"}
    return RunStore().write_artifact(run.run_id, f"saved_candidate_{candidate['label']}.json", json.dumps(data, indent=2))


def record_decision(run: Run, candidate: dict, gate: dict, approver: str, decision: str) -> dict:
    from driveflow.runs.schemas import SpadeError

    rec = ApprovalRecord(approval_id=new_id("appr"), subject=_subject(run), baseline_run=None, candidate_run=run.run_id, gate_report=gate,
                         approver=approver.strip(), user_confirmed_at=utc_now(), decision=decision)
    try:
        return {"record": RunStore().record_approval(rec)}
    except SpadeError as exc:
        return {"error": str(exc)}

