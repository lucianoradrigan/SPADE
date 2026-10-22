"""GUI boundary to the PLECS batch route (expansion task E2; spec Sec. 3.14). The view passes plain
values; profiles, sweeps, jobs and connection reports are built here from driveflow.integrations.plecs,
driveflow.runs and driveflow.jobs. Nothing here talks to PLECS except ``test_connection`` (on a
button) and the worker running a submitted sweep."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from driveflow.integrations.plecs import batch
from driveflow.integrations.plecs import manifest as mf
from driveflow.integrations.plecs.rpc_client import DEFAULT_URL, EngineCapabilityReport, PlecsRpcClient, PlecsRpcError
from driveflow.jobs.service import JobService
from driveflow.runs.profiles import is_stale
from driveflow.runs.schemas import ExperimentSpec, PlantProfile, Run, SpadeError
from driveflow.runs.store import RunStore

EXAMPLE_MANIFEST = mf.EXAMPLE_BUCK
TEMPLATES = mf.TEMPLATES


def test_connection(url: str) -> dict:
    """The capability report as a dict; an invalid or remote URL becomes a failed report."""
    try:
        return PlecsRpcClient(url).test_connection().to_dict()
    except PlecsRpcError as exc:
        return EngineCapabilityReport(url=url, reachable=False, error_code=exc.code.value, message=str(exc), hint=exc.hint,
                                      checked=("RPC endpoint address",)).to_dict()


def profiles() -> list[PlantProfile]:
    return [p for p in RunStore().list_profiles() if p.kind in ("plecs_rpc_batch", "plecs_fmu_co_sim", "surrogate")]


def readiness(p: PlantProfile) -> str:
    """'Stale' when the circuit file changed or is gone, else the profile's own status."""
    path = p.parameters.get("circuit_path")
    if path and is_stale(p, path):
        return "Stale"
    return {"imported": "Imported", "draft": "Draft", "published": "Published", "validated": "Validated", "unsupported": "Unsupported", "failed": "Failed"}.get(p.status, p.status.title())


def profile_rows(search: str = "") -> pd.DataFrame:
    q = (search or "").strip().lower()
    rows = []
    for p in profiles():
        if q and q not in p.profile_id.lower() and q not in p.display_name.lower():
            continue
        caps = p.capabilities
        rows.append({"Name": p.display_name, "ID": p.profile_id, "Version": p.version,
                     "Source": {"plecs_rpc_batch": "PLECS RPC dataset", "plecs_fmu_co_sim": "PLECS FMU", "surrogate": "Surrogate"}[p.kind],
                     "Readiness": readiness(p), "Batch sweep": "Shown" if caps.can_simulate_batch else "Not shown yet",
                     "Interactive stepping": "Not supported" if p.kind == "plecs_rpc_batch" else ("Verified" if caps.can_step_interactively else "Not verified"),
                     "Last sweep": p.parameters.get("last_sweep_run") or "—", "Circuit": p.parameters.get("circuit_name"), "SHA-256": (p.file_hash or "")[:12]})
    return pd.DataFrame(rows)


def create_profile(profile_id: str, display_name: str, manifest_json: str, rpc_url: str, uploaded: tuple[bytes, str] | None = None,
                   local_path: str | None = None) -> PlantProfile:
    """Raises ValueError (ManifestError is one) with a readable message."""
    man = mf.from_json(manifest_json)
    if uploaded is not None:
        path = batch.store_uploaded_circuit(uploaded[0], uploaded[1])
    elif local_path:
        path = Path(local_path).expanduser()
    else:
        raise ValueError("No circuit has been imported yet.")
    store = RunStore()
    try:
        existing = store.get_profile(profile_id)
    except SpadeError:
        existing = None
    profile = batch.create_profile(profile_id, path, man, display_name or None, rpc_url, version=(existing.version + 1) if existing else 1)
    return store.save_profile(profile)


def get_profile(profile_id: str) -> PlantProfile:
    return RunStore().get_profile(profile_id)


def manifest_of(p: PlantProfile) -> mf.InterfaceManifest:
    return batch.profile_manifest(p)


def design(p: PlantProfile, ranges: dict, method: str, n_samples: int | None, seed: int, fixed: dict) -> list[dict]:
    return batch.design_conditions(manifest_of(p), ranges, method, n_samples, seed, fixed)


def submit_sweep(p: PlantProfile, conditions: list[dict], method: str, seed: int, time_span_s: float, time_limit_s: float, n_output_points: int | None):
    spec = ExperimentSpec.create("plecs_sweep", plant_profile_id=p.profile_id, conditions=conditions, seed=int(seed), algorithm=method,
                                 params={"time_span_s": float(time_span_s), "time_limit_s": float(time_limit_s), "n_output_points": n_output_points,
                                         "profile_version": p.version}, budget={"conditions": len(conditions)})
    return JobService().submit(spec)


def active_sweeps(profile_id: str) -> list[dict]:
    """Queued or running plecs_sweep jobs of this profile: [{job_id, state, done, total}]."""
    from driveflow.jobs.schemas import JobState

    s = JobService()
    out = []
    for job in s.list([JobState.QUEUED, JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING]):
        if job.kind != "plecs_sweep":
            continue
        try:
            if s.experiment(job).plant_profile_id != profile_id:
                continue
        except SpadeError:
            continue
        out.append({"job_id": job.job_id, "state": job.state.value, "done": job.progress_done, "total": job.progress_total})
    return out


def sweep_runs(profile_id: str, limit: int = 50) -> list[Run]:
    store = RunStore()
    out = []
    for run in store.list_runs(500):
        if run.kind != "plecs_sweep" or not run.experiment_id:
            continue
        try:
            if store.get_experiment(run.experiment_id).plant_profile_id == profile_id:
                out.append(run)
        except SpadeError:
            continue
        if len(out) >= limit:
            break
    return out


def conditions_table(run: Run) -> pd.DataFrame:
    rows = [{"#": c["index"], "Status": c["status"], **{k: v for k, v in c["condition"].items()}, "Wall time (s)": c.get("seconds"), "Error": c.get("error") or ""}
            for c in run.conditions]
    return pd.DataFrame(rows)


def condition_frame(run: Run, index: int) -> pd.DataFrame | None:
    """The completed condition's table from the run's artifacts (None for a failed one)."""
    cond = next((c for c in run.conditions if c["index"] == index), None)
    if cond is None or cond["status"] != "ok" or not cond.get("artifact"):
        return None
    path = RunStore().artifact_path({"path": cond["artifact"]})
    return pd.read_parquet(path) if path.exists() else None


def artifact_text(run: Run, name: str) -> str | None:
    art = next((a for a in run.artifacts if a["path"].endswith(name)), None)
    if art is None:
        return None
    path = RunStore().artifact_path(art)
    return path.read_text() if path.exists() else None


def dataset_zip(run: Run) -> bytes:
    """manifest.json, conditions.csv and every completed condition's Parquet, from the run's artifacts."""
    import zipfile

    buf = io.BytesIO()
    store = RunStore()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for art in run.artifacts:
            name = Path(art["path"]).name
            if name.endswith((".parquet", ".csv")) or name == "manifest.json":
                path = store.artifact_path(art)
                if path.exists():
                    z.write(path, name)
    return buf.getvalue()


def diagnostics_json(report: dict) -> str:
    """Shareable diagnostics: the report without anything machine-specific beyond the endpoint."""
    keep = {k: report.get(k) for k in ("url", "reachable", "plecs_version", "methods", "checked", "not_checked", "error_code", "message", "latency_ms")}
    return json.dumps(keep, indent=2)



STATUS_LABELS = {"completed": "Completed", "partially_completed": "Partially completed", "failed": "Failed", "cancelled": "Cancelled"}


def sweeps_of(profile_id: str, limit: int = 20) -> list[dict]:
    """Sweep runs of a profile, newest first, with their counts and per-condition failures (task A1:
    a failed or partial sweep must be visible, never averaged away)."""
    from driveflow.viz.components.failure_panel import failure_rows

    store, out = RunStore(), []
    for r in store.list_runs(500):
        if r.kind != "plecs_sweep":
            continue
        try:
            spec = store.get_experiment(r.experiment_id) if r.experiment_id else None
        except SpadeError:
            spec = None
        if spec is None or spec.plant_profile_id != profile_id:
            continue
        n = len(r.conditions)
        out.append({"run_id": r.run_id, "job_id": r.job_id, "status": STATUS_LABELS.get(r.status, r.status.replace("_", " ").title()),
                    "n_total": n, "n_ok": sum(c["status"] == "ok" for c in r.conditions), "n_failed": sum(c["status"] == "failed" for c in r.conditions),
                    "n_skipped": sum(c["status"] == "skipped" for c in r.conditions), "completed_utc": r.completed_utc,
                    "failures": [{("Outcome" if k == "Status" else k): v for k, v in row.items()} for row in failure_rows(r.conditions)],
                    "failure_reason": r.failure_reason})
        if len(out) >= limit:
            break
    return out
