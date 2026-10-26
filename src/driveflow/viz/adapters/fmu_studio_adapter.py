"""GUI boundary to the OpenModelica FMU route of PLECS Plant Studio ("Use as Gym plant", task B5).

The view passes plain values; profiles, verification jobs and publication go through
driveflow.integrations.fmu.gym_profile and driveflow.jobs. Never raises into the page: domain
errors come back as ValueError messages for the view to show."""

from __future__ import annotations

import json

import pandas as pd

from driveflow.integrations.fmu import gym_profile as gp
from driveflow.runs.schemas import ExperimentSpec, SpadeError

STAGES = ("1. Goal", "2. Engine", "3. Circuit", "4. Signals", "5. Gym behavior", "6. Validate & publish")
READINESS_ITEMS = gp.READINESS_ITEMS
B3_LINK = gp.B3_LINK


def engine_report() -> dict:
    return gp.engine_report()


def candidates() -> list[dict]:
    try:
        return gp.candidates()
    except Exception as exc:  # pragma: no cover -- defensive: the page must render
        return [{"name": "?", "status": "Failed", "reason": f"{type(exc).__name__}: {exc}", "publishable": False, "note": None, "b3": None,
                 "platforms": [], "sha256": None, "path": ""}]


def candidates_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"FMU": r["name"], "Status": r["status"], "Tool": ("PLECS" if r.get("is_plecs") else (r.get("generation_tool") or "—").split(" ")[0]),
                          "FMI": r.get("fmi_version") or "—", "Interfaces": ", ".join({"CoSimulation": "CS", "ModelExchange": "ME"}.get(i, i) for i in r.get("interfaces") or []) or "—",
                          "B3 verdict": (r.get("b3") or {}).get("verdict") or ("— (external)" if r.get("sha256") else "—"),
                          "Platforms": ", ".join(r.get("platforms") or []) or "—", "SHA-256": (r.get("sha256") or "")[:12] or "—",
                          "Why": r["reason"] + (f" · {r['note']}" if r.get("note") else "")} for r in rows])


def profiles() -> list:
    from driveflow.runs.store import RunStore

    return RunStore().list_profiles(gp.KIND)


def get_profile(profile_id: str):
    from driveflow.runs.store import RunStore

    return RunStore().get_profile(profile_id)


def import_fmu(profile_id: str, display_name: str, fmu_name: str):
    return gp.new_profile(profile_id.strip(), display_name.strip(), fmu_name)


def suggest_mapping(p) -> dict:
    return gp.suggest_mapping(p)


def save_mapping(p, mapping: dict):
    return gp.set_mapping(p, mapping)


def default_gym(p) -> dict:
    g = p.parameters.get("gym")
    if g:
        return g
    params = {q["name"]: q["start"] for q in p.parameters["fmu_parameters"] if q.get("start") is not None}
    return {"tau_s": 2e-5 if p.parameters["model"].startswith("Buck") else 1e-4, "duration_s": 0.02, "parameters": params,
            "reward_terms": [{"kind": "tracking", "weight": 1.0}, {"kind": "overcurrent", "weight": 10.0, "threshold": 0.8}, {"kind": "action_rate", "weight": 0.01}],
            "action_out_of_range": "clip", "interface": "auto", "me_substeps": 10}


def upload_fmu(data: bytes, filename: str):
    """Stores an FMU uploaded in Plant Studio (e.g. exported from PLECS). Raises ValueError."""
    return gp.save_upload(data, filename)


PLECS_CS_RUNTIME = gp.PLECS_CS_RUNTIME


def save_gym(p, gym_cfg: dict):
    return gp.set_gym(p, gym_cfg)


def space_text(p) -> str:
    m = p.parameters.get("mapping")
    if not m:
        return "—"
    n_obs = len(m["observations"]) + (2 if m.get("reference") else 0)
    a = m["action"]
    return f"action_space = Box({a['low']:g}, {a['high']:g}, shape=(1,)) · observation_space = Box(-inf, inf, shape=({n_obs},), float32)"


def readiness(p) -> dict:
    return gp.readiness(p)


def state(p) -> str:
    return gp.display_state(p)


def blockers(p) -> list[str]:
    return gp.publish_blockers(p)


def submit_verification(p):
    """Queues the stage-6 checks as a persistent job (kind fmu_verification); the profile shows
    VERIFYING until the worker records the result."""
    from driveflow.jobs.service import JobService

    spec = ExperimentSpec.create("fmu_verification", plant_profile_id=p.profile_id, params={"profile_id": p.profile_id, "version": p.version},
                                 seed=0, budget={"checks": 13})
    job = JobService().submit(spec)
    gp.mark_verifying(p)
    return job


def verification_job(p) -> dict | None:
    """The newest fmu_verification job of this profile version: {job_id, state, run_id}."""
    from driveflow.jobs.service import JobService

    s = JobService()
    for job in s.list(limit=200):
        if job.kind != "fmu_verification":
            continue
        try:
            params = s.experiment(job).params
        except SpadeError:
            continue
        if params.get("profile_id") == p.profile_id and params.get("version") == p.version:
            run = s.run_for(job)
            return {"job_id": job.job_id, "state": job.state.value, "run_id": run.run_id if run else None}
    return None


def verification_report(p) -> dict | None:
    """The stored verification.json of the profile's last verification run."""
    from driveflow.runs.store import RunStore

    v = p.parameters.get("verification")
    if not v or not v.get("run_id"):
        return None
    store = RunStore()
    try:
        run = store.get_run(v["run_id"])
    except SpadeError:
        return None
    art = next((a for a in run.artifacts if a["path"].endswith("verification.json")), None)
    if art is None:
        return None
    path = store.artifact_path(art)
    return json.loads(path.read_text()) if path.exists() else None


def checks_table(report: dict) -> pd.DataFrame:
    icon = {"pass": "✓ Passed", "fail": "✗ Failed", "not_applicable": "— Not applicable"}
    return pd.DataFrame([{"#": str(c["id"]), "Check": c["name"], "Result": icon.get(c["status"], c["status"]), "Detail": c["detail"]} for c in report["checks"]])


def submit_fetch(release: str | None, model: str | None):
    """Queues `python -m driveflow.fmu fetch` as a job (kind fmu_fetch, B8)."""
    from driveflow.jobs.service import JobService

    spec = ExperimentSpec.create("fmu_fetch", params={"release": release, "model": model}, budget={"downloads": 1})
    return JobService().submit(spec)


def publish(p) -> str:
    return gp.publish(p)


def snippet(env_id: str) -> str:
    from driveflow.envs import registry

    return registry.snippet(env_id)
