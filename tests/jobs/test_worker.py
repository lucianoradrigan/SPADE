"""The worker as a separate process (driveflow.jobs.worker) -- expansion task E0 acceptance
(spec Sec. 13 Phase 0): a job survives the GUI, is cancelled honestly, and a killed worker leaves
the real state behind instead of a job stuck in "running"."""

import os
import subprocess
import sys
import time

import pytest

from driveflow.jobs.persistence import JobDB
from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.jobs.worker import run_spec
from driveflow.runs.schemas import ErrorCode, ExperimentSpec


def _worker(root, *args):
    env = dict(os.environ, SPADE_WORKSPACE=str(root))
    return subprocess.Popen([sys.executable, "-m", "driveflow.jobs.worker", "serve", "--poll", "0.1", *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait(pred, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = pred()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("condition not reached in time")


def _long(steps=400, step_s=0.05):
    return ExperimentSpec.create("selftest.steps", params={"steps": steps, "step_s": step_s})


def test_cancel_is_requested_then_confirmed_by_the_worker(tmp_path):
    s = JobService(tmp_path)
    job = s.submit(_long())
    proc = _worker(tmp_path)
    try:
        _wait(lambda: (s.get(job.job_id).progress_done or 0) >= 3)
        assert s.cancel(job.job_id).state == JobState.CANCELLING  # requested, not yet confirmed
        done = _wait(lambda: s.get(job.job_id) if s.get(job.job_id).terminal else None)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    run = s.run_for(done)
    assert done.state == JobState.CANCELLED and run.status == "cancelled"
    statuses = [c["status"] for c in run.conditions]
    assert statuses.count("ok") >= 3 and statuses.count("skipped") > 0 and "failed" not in statuses
    assert any("Cancellation requested" in e["message"] for e in s.events(job.job_id))


def test_killed_worker_leaves_the_real_state_and_the_next_worker_carries_on(tmp_path):
    s, db = JobService(tmp_path), JobDB(tmp_path)
    job = s.submit(_long())
    proc = _worker(tmp_path)
    _wait(lambda: s.get(job.job_id).state == JobState.RUNNING)
    proc.kill()
    proc.wait(timeout=30)
    assert s.get(job.job_id).state == JobState.RUNNING  # nobody has noticed yet
    if os.name == "posix":
        assert db.recover_orphans(stale_after_s=3600) == [job.job_id]  # the dead pid is enough
    else:
        assert db.recover_orphans(stale_after_s=0) == [job.job_id]  # heartbeat age
    lost = s.get(job.job_id)
    assert lost.state == JobState.FAILED and lost.error_code == ErrorCode.WORKER_LOST.value and not lost.resumable
    # A new worker does not re-run the lost job, and does run new work.
    nxt = s.submit(ExperimentSpec.create("selftest.steps", params={"steps": 2, "step_s": 0.0}))
    out = _worker(tmp_path, "--once")
    assert out.wait(timeout=120) == 0, out.stdout.read()
    assert s.get(nxt.job_id).state == JobState.COMPLETED and s.get(job.job_id).state == JobState.FAILED


def test_worker_start_closes_jobs_left_by_a_dead_worker(tmp_path):
    s, db = JobService(tmp_path), JobDB(tmp_path)
    job = s.submit(_long(steps=2, step_s=0.0))
    db.claim_next("ghost", 2**22 + 4321)
    db.transition(job.job_id, JobState.RUNNING)
    with db._tx() as conn:
        conn.execute("UPDATE jobs SET heartbeat_ts = 0 WHERE id = ?", (job.job_id,))
    proc = _worker(tmp_path, "--once")
    assert proc.wait(timeout=120) == 0
    assert "closed 1 job(s) left by a lost worker" in proc.stdout.read()
    assert s.get(job.job_id).state == JobState.FAILED


@pytest.mark.simulation
def test_scenario_sweep_runs_in_the_worker_and_its_command_reproduces_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_WORKSPACE", str(tmp_path))
    s = JobService(tmp_path)
    spec = ExperimentSpec.create("scenario_sweep", seed=0, conditions=[{"omega_ref_rad_s": 100.0, "duration_s": 0.05},
                                                                      {"omega_ref_rad_s": 150.0, "duration_s": 0.05}])
    job = s.submit(spec)
    proc = _worker(tmp_path, "--once")
    assert proc.wait(timeout=600) == 0, proc.stdout.read()
    run = s.run_for(s.get(job.job_id))
    assert run.status == "completed" and len([a for a in run.artifacts if a["path"].endswith(".parquet")]) == 2
    assert run.environment["packages"]["numpy"] and run.seed == 0
    # The stored command, executed from the workspace, reproduces the same config and numbers.
    rel = run.command.split()[-1]
    again = run_spec(rel)
    assert again.config_hash == run.config_hash and again.run_id != run.run_id
    assert [c["metrics"] for c in again.conditions] == [c["metrics"] for c in run.conditions]
    for a, b in zip(sorted(x["path"] for x in run.artifacts if x["path"].endswith(".parquet")),
                    sorted(x["path"] for x in again.artifacts if x["path"].endswith(".parquet"))):
        assert s.store.artifact_path({"path": a}).read_bytes() == s.store.artifact_path({"path": b}).read_bytes()
