"""GUI boundary to custom classifier training (AI Diagnostics → Train a model). Builds the request,
submits it as a ``classifier_training`` job and reads finished runs back; no training happens in
the Streamlit process."""

from __future__ import annotations

import json
from pathlib import Path

from driveflow.ai import custom_training as ct
from driveflow.ai.registry import REPO_ROOT
from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.runs.profiles import native_profile_for
from driveflow.runs.schemas import ExperimentSpec, Run
from driveflow.runs.store import RunStore
from driveflow.viz.adapters.simulation_adapter import resolve_fault

KIND = "classifier_training"
PLANT = {"dc_motor": "dc_perm_ex_v1", "vsc_dpc": "vsc_dpc_v1"}


def default_workers() -> int:
    """1: on the development laptop 4 processes gave no measurable speed-up over one
    (docs/audit/ai/custom_training_tl.md), and each extra process re-imports TensorFlow."""
    return 1


def dc_classes(labels: list, custom_faults: dict, custom_mech: dict | None = None) -> list:
    """GUI fault labels ("healthy", built-in, saved custom fault types) -> request classes."""
    out = []
    for label in labels:
        fault_type, override, extras = resolve_fault(label, custom_faults)
        if fault_type is None:
            out.append(ct.dc_class(ct.HEALTHY))
        else:
            out.append(ct.dc_class(label, fault_type, override, extras, (custom_mech or {}).get(label)))
    return out


def submit(req: dict, workers: int):
    req = ct.validate_request(req)
    spec = ExperimentSpec.create(KIND, plant_profile_id=native_profile_for(PLANT[req["domain"]]).profile_id, seed=int(req["model"]["seed"]),
                                 algorithm=req["model"]["preset"], params={"request": req, "workers": int(workers)},
                                 objectives=("test macro-F1",), conditions=tuple(ct.build_conditions(req, validated=True)))
    return JobService().submit(spec)


def active_jobs() -> list[dict]:
    s = JobService()
    return [{"job_id": j.job_id, "state": j.state.value, "done": j.progress_done, "total": j.progress_total, "phase": j.phase}
            for j in s.list([JobState.QUEUED, JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING]) if j.kind == KIND]


def runs(domain: str | None = None, limit: int = 50) -> list[Run]:
    out = []
    for r in RunStore().list_runs(500):
        if r.kind != KIND:
            continue
        if domain is not None and (r.params.get("request") or {}).get("domain") not in (domain, None):
            continue
        out.append(r)
    return out[:limit]


def run_dir_of(run: Run) -> Path | None:
    rel = (run.metrics or {}).get("run_dir")
    if not rel:
        return None
    p = Path(rel)
    p = p if p.is_absolute() else REPO_ROOT / p
    return p if (p / "metrics.json").exists() else None


def model_metrics(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text())
