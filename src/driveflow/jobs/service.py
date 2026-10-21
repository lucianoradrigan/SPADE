"""JobService: the facade the GUI adapter and the CLI use -- expansion task E0 (spec Sec. 6, 12).

``submit`` validates the spec against its handler BEFORE anything is queued (a bad spec is an
error now, not a failed job later), saves the immutable ExperimentSpec and queues a Job. Nothing
here executes work: the worker process does (driveflow.jobs.worker).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
from datetime import datetime

from driveflow.jobs.handlers import HANDLERS, handler_for
from driveflow.jobs.persistence import JobDB
from driveflow.jobs.schemas import Job, JobState
from driveflow.runs.schemas import ErrorCode, ExperimentSpec, Run, SpadeError
from driveflow.runs.store import WORKER_MODULE, RunStore

#: Completed jobs of the same kind needed before an ETA is shown (spec Sec. 6: "ETA nulo hasta
#: tener historial suficiente").
ETA_MIN_HISTORY = 3


class JobService:
    def __init__(self, root=None):
        self.store = RunStore(root)
        self.db = JobDB(self.store.root)
        self.root = self.store.root

    # ---------------------------------------------------------------- submit / query
    def submit(self, spec: ExperimentSpec, engine: str | None = None) -> Job:
        info = handler_for(spec.kind)
        info.validate(spec)
        self.store.save_experiment(spec)
        return self.db.insert(spec.experiment_id, spec.kind, engine or info.engine)

    def get(self, job_id: str) -> Job:
        return self.db.get(job_id)

    def list(self, states=None, search: str | None = None, limit: int = 200) -> list[Job]:
        return self.db.list(states, search, limit)

    def cancel(self, job_id: str) -> Job:
        return self.db.request_cancel(job_id)

    def events(self, job_id: str) -> list[dict]:
        return self.db.events(job_id)

    def counts(self) -> dict:
        return self.db.counts()

    def experiment(self, job: Job) -> ExperimentSpec:
        return self.store.get_experiment(job.experiment_id)

    def run_for(self, job: Job) -> Run | None:
        if not job.run_id or not job.terminal:
            return None
        try:
            return self.store.get_run(job.run_id)
        except SpadeError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                return None
            raise

    @staticmethod
    def kinds() -> dict:
        return {k: v.title for k, v in HANDLERS.items()}

    # ---------------------------------------------------------------- ETA
    def eta_seconds(self, job: Job) -> float | None:
        """Median seconds per unit of progress over finished jobs of the same kind, times the
        remaining units. None without measurable progress or without enough history."""
        if job.state not in (JobState.RUNNING, JobState.PREPARING) or not job.progress_total:
            return None
        rates = []
        for past in self.db.list(states=[JobState.COMPLETED], search=None, limit=50):
            if past.kind == job.kind and past.progress_total and past.started_at and past.finished_at:
                seconds = (datetime.fromisoformat(past.finished_at) - datetime.fromisoformat(past.started_at)).total_seconds()
                rates.append(seconds / past.progress_total)
        if len(rates) < ETA_MIN_HISTORY:
            return None
        return statistics.median(rates) * max(0, job.progress_total - (job.progress_done or 0))

    # ---------------------------------------------------------------- worker
    def live_workers(self) -> list[dict]:
        return self.db.live_workers()

    def start_local_worker(self) -> int:
        """Starts ``python -m driveflow.jobs.worker`` as a detached process on this machine and
        returns its pid. Its output goes to <workspace>/worker.log."""
        env = dict(os.environ, SPADE_WORKSPACE=str(self.root))
        log = open(self.root / "worker.log", "ab")  # noqa: SIM115 -- handed to the child process
        kwargs = {"start_new_session": True} if os.name == "posix" else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)}
        proc = subprocess.Popen([sys.executable, "-m", WORKER_MODULE, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
        log.close()
        return proc.pid
