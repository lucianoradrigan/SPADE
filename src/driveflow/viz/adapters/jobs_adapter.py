"""GUI boundary to the durable job queue and run store (expansion task E0; spec Sec. 6, 12).

Pure functions over driveflow.jobs / driveflow.runs: the view never touches SQLite or the worker.
A fresh JobService per call is cheap (one SQLite open) and keeps every page rerun reading the real
persisted state, never a cached copy.
"""

from __future__ import annotations

from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.runs.compare import compare_runs, numeric_metrics
from driveflow.runs.profiles import native_profile_for
from driveflow.runs.schemas import ApprovalRecord, ErrorCode, ExperimentSpec, Run, SpadeError, new_id, utc_now
from driveflow.runs.store import RunStore

FILTERS = {
    "All": None,
    "Running": [JobState.QUEUED, JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING],
    "Completed": [JobState.COMPLETED, JobState.PARTIALLY_COMPLETED],
    "Failed": [JobState.FAILED],
    "Cancelled": [JobState.CANCELLED],
}

#: The short DC-motor sweep offered on the page (spec Sec. 13 Phase 0 acceptance: "un job de prueba").
DEMO_SWEEP_SPEEDS = (100.0, 150.0, 200.0)


def service() -> JobService:
    return JobService()


def queue_summary() -> dict:
    """{running, queued, workers, error?} for the sidebar footer; never raises."""
    try:
        s = service()
        counts = s.counts()
        return {"running": sum(counts.get(k.value, 0) for k in (JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING)),
                "queued": counts.get(JobState.QUEUED.value, 0), "workers": len(s.live_workers())}
    except Exception as exc:  # the sidebar must never break a page
        return {"running": 0, "queued": 0, "workers": 0, "error": f"{type(exc).__name__}: {exc}"}


def list_jobs(filter_name: str = "All", search: str | None = None):
    return service().list(FILTERS[filter_name], search or None)


def job_details(job_id: str) -> dict:
    s = service()
    job = s.get(job_id)
    try:
        spec = s.experiment(job)
    except SpadeError:
        spec = None
    return {"job": job, "spec": spec, "run": s.run_for(job), "events": s.events(job_id), "eta": s.eta_seconds(job)}


def cancel_job(job_id: str):
    return service().cancel(job_id)


def live_workers() -> list[dict]:
    return service().live_workers()


def start_worker() -> int:
    return service().start_local_worker()


def build_sweep_spec(plant: str, duration_s: float, values: list[float], seed: int) -> ExperimentSpec:
    """A scenario_sweep spec over reference values, validated by the backend on submit."""
    if plant == "dc_perm_ex_v1":
        conditions = [{"plant_config_id": plant, "controller_type": "PI", "omega_ref_rad_s": float(v), "duration_s": float(duration_s), "seed": int(seed)} for v in values]
    elif plant == "vsc_dpc_v1":
        conditions = [{"plant_config_id": plant, "controller_type": "DPC", "reference_magnitude_v": float(v), "duration_s": float(duration_s), "seed": int(seed)} for v in values]
    elif plant == "buck_v1":
        conditions = [{"plant_config_id": plant, "controller_type": "PI", "buck_vout_ref_v": float(v), "duration_s": float(duration_s), "seed": int(seed)} for v in values]
    else:
        raise SpadeError(ErrorCode.UNSUPPORTED, f"no sweep form for plant {plant!r}")
    profile = native_profile_for(plant)
    return ExperimentSpec.create("scenario_sweep", plant_profile_id=profile.profile_id if profile else None, conditions=conditions, seed=int(seed),
                                 params={"swept": {"dc_perm_ex_v1": "omega_ref_rad_s", "vsc_dpc_v1": "reference_magnitude_v", "buck_v1": "buck_vout_ref_v"}[plant]})


def submit(spec: ExperimentSpec):
    return service().submit(spec)


def list_runs(limit: int = 200) -> list[Run]:
    return RunStore().list_runs(limit)


def get_run(run_id: str) -> Run:
    return RunStore().get_run(run_id)


def artifact_bytes(run: Run, artifact: dict) -> bytes | None:
    path = RunStore().artifact_path(artifact)
    return path.read_bytes() if path.exists() else None


def comparison(run_ids: list[str], metric: str | None, lower_is_better: bool = True) -> dict:
    runs = [get_run(r) for r in run_ids]
    metrics = numeric_metrics(runs)
    result = compare_runs(runs, metric or (metrics[0] if metrics else ""), lower_is_better)
    result["available_metrics"] = metrics
    return result


def record_approval(subject: str, baseline_run: str | None, candidate_run: str, gate_report: dict, approver: str, decision: str) -> dict:
    try:
        record = ApprovalRecord(approval_id=new_id("approval"), subject=subject, baseline_run=baseline_run, candidate_run=candidate_run,
                                gate_report=gate_report, approver=approver, user_confirmed_at=utc_now(), decision=decision)
        return {"record": RunStore().record_approval(record)}
    except SpadeError as exc:
        return {"error": str(exc)}


def mirror_session_run(record: dict) -> None:
    """Keeps a durable copy of an interactive page's run record (viz/adapters/session.py) as a
    'legacy_session' Run: parameters and summary only, no artifacts, no reproducible command.
    Never raises -- persistence must not break a page."""
    try:
        RunStore().save_run(Run(run_id=f"session-{new_id(record['kind'])}", status="legacy_session", origin="SIMULATED", kind=f"session:{record['kind']}",
                                completed_utc=record.get("utc"), config_hash=None, params=record.get("params") or {},
                                metrics={"session_run_id": record.get("run_id"),
                                         **{k: v for k, v in (record.get("summary") or {}).items() if isinstance(v, (int, float, str, bool)) or v is None}}))
    except Exception:
        pass
