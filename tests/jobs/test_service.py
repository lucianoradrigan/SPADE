"""Job state machine, queue and service (driveflow.jobs) -- expansion task E0. In-process only;
the separate-process behaviour is in test_worker.py."""

import datetime as dt

import pytest

from driveflow.jobs.persistence import JobDB
from driveflow.jobs.schemas import TRANSITIONS, JobState, check_transition
from driveflow.jobs.service import ETA_MIN_HISTORY, JobService
from driveflow.jobs.worker import Worker
from driveflow.runs.schemas import ErrorCode, ExperimentSpec, SpadeError


def _selftest(steps=3, step_s=0.0, **params):
    return ExperimentSpec.create("selftest.steps", params={"steps": steps, "step_s": step_s, **params})


def test_terminal_states_have_no_way_out():
    for state in (JobState.COMPLETED, JobState.PARTIALLY_COMPLETED, JobState.FAILED, JobState.CANCELLED):
        assert state not in TRANSITIONS
        with pytest.raises(SpadeError):
            check_transition(state, JobState.RUNNING)
    with pytest.raises(SpadeError):
        check_transition(JobState.QUEUED, JobState.COMPLETED)  # never "completed" without running


def test_submit_validates_before_queueing(tmp_path):
    s = JobService(tmp_path)
    with pytest.raises(SpadeError) as exc:
        s.submit(ExperimentSpec.create("no.such.kind"))
    assert exc.value.code == ErrorCode.UNKNOWN_JOB_KIND
    with pytest.raises(SpadeError):
        s.submit(ExperimentSpec.create("scenario_sweep", conditions=[{"duration_s": -1.0}]))
    with pytest.raises(SpadeError):
        s.submit(ExperimentSpec.create("scenario_sweep", conditions=[{"controller_type": "DPC"}]))  # DPC on the DC plant
    assert s.list() == []


def test_cancel_while_queued_ends_at_once_and_runs_nothing(tmp_path):
    s = JobService(tmp_path)
    job = s.submit(_selftest())
    assert s.cancel(job.job_id).state == JobState.CANCELLED
    assert Worker(tmp_path).serve(once=True) == 0
    assert s.run_for(s.get(job.job_id)) is None
    with pytest.raises(SpadeError):
        s.cancel(job.job_id)


def test_worker_completes_and_records_partial_failures(tmp_path):
    s = JobService(tmp_path)
    ok, partial, failed = s.submit(_selftest()), s.submit(_selftest(fail_steps=[1])), s.submit(_selftest(steps=2, fail_steps=[0, 1]))
    assert Worker(tmp_path).serve(once=True) == 3
    assert [s.get(j.job_id).state for j in (ok, partial, failed)] == [JobState.COMPLETED, JobState.PARTIALLY_COMPLETED, JobState.FAILED]
    run = s.run_for(s.get(partial.job_id))
    assert run.status == "partially_completed" and [c["status"] for c in run.conditions] == ["ok", "failed", "ok"]
    assert {a["path"].rsplit("/", 1)[-1] for a in run.artifacts} >= {"spec.json", "log.txt", "selftest.txt"}
    assert run.config_hash == s.experiment(partial).config_hash() and run.command.endswith(f"runs/{run.run_id}/spec.json")
    assert "All 2 conditions failed" in s.run_for(s.get(failed.job_id)).failure_reason
    # The lock was released after every job.
    assert s.db.engine_lock("selftest") is None


def test_handler_exception_is_a_failed_run_with_a_traceback(tmp_path, monkeypatch):
    from driveflow.jobs import handlers

    def boom(ctx):
        raise RuntimeError("solver exploded")

    monkeypatch.setitem(handlers.HANDLERS, "selftest.steps", handlers.HandlerInfo(boom, handlers._validate_selftest, "boom", "selftest"))
    s = JobService(tmp_path)
    job = s.submit(_selftest())
    Worker(tmp_path).serve(once=True)
    job = s.get(job.job_id)
    run = s.run_for(job)
    assert job.state == JobState.FAILED and job.error_code == ErrorCode.HANDLER_ERROR.value
    assert "solver exploded" in run.failure_reason and any(a["path"].endswith("traceback.txt") for a in run.artifacts)


def test_engine_lock_serialises_jobs_of_one_engine(tmp_path):
    db = JobDB(tmp_path)
    s = JobService(tmp_path)
    a, b = s.submit(_selftest(), engine="plecs"), s.submit(_selftest(), engine="plecs")
    c = s.submit(_selftest(), engine="other")
    first = db.claim_next("w1", 1)  # FIFO even within one second (queued_at has 1 s resolution)
    assert first.job_id == a.job_id and db.engine_lock("plecs") == a.job_id
    second = db.claim_next("w2", 2)
    assert second.job_id == c.job_id  # b waits: its engine is busy
    assert db.claim_next("w3", 3) is None
    db.transition(a.job_id, JobState.CANCELLED)
    assert db.claim_next("w3", 3).job_id == b.job_id


def test_eta_is_unavailable_until_there_is_history(tmp_path):
    s = JobService(tmp_path)
    for _ in range(ETA_MIN_HISTORY - 1):
        s.submit(_selftest(steps=4))
    Worker(tmp_path).serve(once=True)
    job = s.submit(_selftest(steps=4))
    claimed = s.db.claim_next("w", 1)
    s.db.transition(claimed.job_id, JobState.RUNNING)
    s.db.update_progress(job.job_id, 1, 4, "step 2/4")
    assert s.eta_seconds(s.get(job.job_id)) is None
    # One more finished job of the same kind -> an ETA from the measured median rate.
    s.db.transition(job.job_id, JobState.COMPLETED)
    s.submit(_selftest(steps=4))
    nxt = s.db.claim_next("w", 1)
    s.db.transition(nxt.job_id, JobState.RUNNING)
    s.db.update_progress(nxt.job_id, 2, 4, "step 3/4")
    eta = s.eta_seconds(s.get(nxt.job_id))
    assert eta is not None and eta >= 0


def test_orphaned_job_is_closed_as_worker_lost_not_requeued(tmp_path):
    db, s = JobDB(tmp_path), JobService(tmp_path)
    job = s.submit(_selftest())
    db.claim_next("ghost", 2**22 + 12345)  # a pid that is not running
    db.transition(job.job_id, JobState.RUNNING)
    with db._tx() as conn:
        conn.execute("UPDATE jobs SET heartbeat_ts = 0 WHERE id = ?", (job.job_id,))
    assert db.recover_orphans(stale_after_s=30) == [job.job_id]
    job = s.get(job.job_id)
    assert job.state == JobState.FAILED and job.error_code == ErrorCode.WORKER_LOST.value and not job.resumable
    assert db.engine_lock("selftest") is None
    assert any("Resume unavailable" in e["message"] for e in s.events(job.job_id))


def test_list_filters_and_search(tmp_path):
    s = JobService(tmp_path)
    a, b = s.submit(_selftest()), s.submit(_selftest())
    s.cancel(b.job_id)
    assert [j.job_id for j in s.list([JobState.CANCELLED])] == [b.job_id]
    assert [j.job_id for j in s.list(search=a.job_id[-8:])] == [a.job_id]
    assert s.counts() == {"queued": 1, "cancelled": 1}


def test_ids_sort_by_time():
    from driveflow.runs.schemas import new_id

    assert new_id("job").startswith(f"job-{dt.datetime.now(dt.timezone.utc):%Y%m%d}")
