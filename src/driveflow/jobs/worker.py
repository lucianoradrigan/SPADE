"""Job worker -- a process separate from Streamlit (expansion task E0, spec Sec. 6).

    python -m driveflow.jobs.worker serve [--once] [--poll 1.0]   # process the queue
    python -m driveflow.jobs.worker run runs/<run_id>/spec.json   # re-run a saved spec here

The workspace is $SPADE_WORKSPACE, else <repo>/workspace; a relative spec path is resolved against
it, so the reproducible command stored with each run carries no private absolute path.

On start, ``serve`` closes jobs left active by a worker that died (driveflow.jobs.persistence
recover_orphans): they end as failed with ErrorCode.WORKER_LOST -- never silently re-queued,
because no handler writes checkpoints yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from driveflow.jobs.handlers import JobContext, handler_for
from driveflow.jobs.persistence import JobDB
from driveflow.jobs.schemas import Job, JobState
from driveflow.runs.schemas import ErrorCode, ExperimentSpec, Run, canonical_json, new_id, utc_now
from driveflow.runs.store import RunStore, environment_info, reproducible_command

HEARTBEAT_S = 2.0


def _final_state(conditions: list) -> tuple[JobState, str | None]:
    n = len(conditions)
    ok = sum(c["status"] == "ok" for c in conditions)
    failed = [c for c in conditions if c["status"] == "failed"]
    if any(c["status"] == "skipped" for c in conditions):
        return JobState.CANCELLED, f"Cancelled on request after {n - sum(c['status'] == 'skipped' for c in conditions)} of {n} conditions."
    if n == 0:
        return JobState.FAILED, "The handler returned no conditions."
    if ok == n:
        return JobState.COMPLETED, None
    if ok == 0:
        return JobState.FAILED, f"All {n} conditions failed; first error: {failed[0].get('error')}"
    return JobState.PARTIALLY_COMPLETED, f"{len(failed)} of {n} conditions failed. Run completed with failed conditions. Inspect the failure table."


def execute(db: JobDB, store: RunStore, job: Job) -> Run:
    """Runs one claimed job (state preparing) to a terminal state and records its Run."""
    spec = store.get_experiment(job.experiment_id)
    info = handler_for(spec.kind)
    artifacts = [store.write_artifact(job.run_id, "spec.json", canonical_json(spec.to_dict()))]

    current = db.get(job.job_id)
    if current.state == JobState.CANCELLING:  # cancelled while preparing: nothing runs
        conditions = [{"index": i, "condition": c, "status": "skipped", "error": "Cancelled before it started."} for i, c in enumerate(spec.conditions)] or \
                     [{"index": 0, "condition": {}, "status": "skipped", "error": "Cancelled before it started."}]
        return _finish(db, store, job, spec, conditions, {}, "SIMULATED", artifacts, None)
    db.transition(job.job_id, JobState.RUNNING, phase="running")
    db.log(job.job_id, "Running experiment — results will appear after computation.")

    def write_artifact(name: str, data: bytes) -> dict:
        art = store.write_artifact(job.run_id, name, data)
        artifacts.append(art)
        return art

    ctx = JobContext(spec=spec, run_id=job.run_id, log=lambda m: db.log(job.job_id, m),
                     progress=lambda done, total, phase: db.update_progress(job.job_id, done, total, phase),
                     cancel_requested=lambda: db.cancel_requested(job.job_id), write_artifact=write_artifact)
    try:
        result = info.fn(ctx)
    except Exception as exc:
        tb = traceback.format_exc()
        artifacts.append(store.write_artifact(job.run_id, "traceback.txt", tb))
        db.log(job.job_id, f"Handler error: {type(exc).__name__}: {exc}", "error")
        return _finish(db, store, job, spec, [], {}, "SIMULATED", artifacts, f"{type(exc).__name__}: {exc} (see traceback.txt)", ErrorCode.HANDLER_ERROR)
    return _finish(db, store, job, spec, result.conditions, result.metrics, result.origin, artifacts, None)


def _finish(db, store, job, spec, conditions, metrics, origin, artifacts, error, error_code=None) -> Run:
    if error is None:
        state, reason = _final_state(conditions)
    else:
        state, reason = JobState.FAILED, error
    db.log(job.job_id, f"Finished: {state.value}" + (f" — {reason}" if reason else ""), "error" if state == JobState.FAILED else "info")
    log_lines = [f"{e['utc']} [{e['level']}] {e['message']}" for e in db.events(job.job_id)]
    artifacts.append(store.write_artifact(job.run_id, "log.txt", "\n".join(log_lines) + "\n"))
    current = db.get(job.job_id)
    run = Run(run_id=job.run_id, status=state.value, origin=origin, kind=spec.kind, experiment_id=spec.experiment_id, job_id=job.job_id,
              started_utc=current.started_at, completed_utc=utc_now(), config_hash=spec.config_hash(), seed=spec.seed, metrics=metrics,
              conditions=conditions, artifacts=artifacts, failure_reason=reason, command=reproducible_command(job.run_id),
              environment=environment_info(), params=spec.params)
    store.save_run(run)
    db.transition(job.job_id, state, error=reason, error_code=(error_code.value if error_code else None), phase="finished")
    return run


class Worker:
    def __init__(self, root=None, worker_id: str | None = None):
        self.store = RunStore(root)
        self.db = JobDB(self.store.root)
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.pid = os.getpid()
        self._current: str | None = None
        self._stop = threading.Event()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            try:
                self.db.heartbeat(self.worker_id, self._current, "busy" if self._current else "idle")
            except Exception:  # a locked database for one beat is not fatal
                pass

    def serve(self, once: bool = False, poll_s: float = 1.0, stale_after_s: float = 30.0) -> int:
        """Processes queued jobs; ``once`` returns after the queue is empty. Returns jobs executed."""
        self.db.register_worker(self.worker_id, self.pid)
        recovered = self.db.recover_orphans(stale_after_s)
        if recovered:
            print(f"[{self.worker_id}] closed {len(recovered)} job(s) left by a lost worker: {', '.join(recovered)}", flush=True)
        beat = threading.Thread(target=self._beat, daemon=True)
        beat.start()
        n = 0
        try:
            while not self._stop.is_set():
                job = self.db.claim_next(self.worker_id, self.pid)
                if job is None:
                    if once:
                        break
                    time.sleep(poll_s)
                    continue
                self._current = job.job_id
                self.db.heartbeat(self.worker_id, job.job_id, "busy")
                print(f"[{self.worker_id}] {job.job_id} ({job.kind}) started", flush=True)
                run = execute(self.db, self.store, job)
                print(f"[{self.worker_id}] {job.job_id} -> {run.status}", flush=True)
                self._current = None
                n += 1
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self.db.unregister_worker(self.worker_id)
        return n


def run_spec(spec_path: str | os.PathLike, root=None) -> Run:
    """Re-executes a saved spec in this process as a new experiment with the same config hash."""
    store = RunStore(root)
    path = Path(spec_path)
    if not path.is_absolute():
        path = store.root / path
    data = json.loads(path.read_text())
    data["experiment_id"], data["created_utc"] = new_id("exp"), utc_now()
    spec = ExperimentSpec.from_dict(data)
    from driveflow.jobs.service import JobService

    service = JobService(store.root)
    job = service.submit(spec)
    worker_id = f"cli-{uuid.uuid4().hex[:8]}"
    service.db.register_worker(worker_id, os.getpid())
    try:
        claimed = None
        while claimed is None:  # wait for the engine lock if a serving worker holds it
            claimed = service.db.claim_next(worker_id, os.getpid(), job_id=job.job_id)
            if claimed is None:
                time.sleep(1.0)
        return execute(service.db, service.store, claimed)
    finally:
        service.db.unregister_worker(worker_id)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m driveflow.jobs.worker", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="process the job queue")
    s.add_argument("--once", action="store_true", help="exit when the queue is empty")
    s.add_argument("--poll", type=float, default=1.0)
    r = sub.add_parser("run", help="re-run a saved spec.json in this process")
    r.add_argument("spec")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run = run_spec(args.spec)
        print(f"{run.run_id}: {run.status}" + (f" — {run.failure_reason}" if run.failure_reason else ""))
        return 0 if run.status in ("completed", "partially_completed") else 1
    Worker().serve(once=getattr(args, "once", False), poll_s=getattr(args, "poll", 1.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
