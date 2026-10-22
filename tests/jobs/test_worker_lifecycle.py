"""X3 (round PLECS/OM): the worker life cycle on every OS of the CI smoke matrix (marker compat, so
the Windows job runs it): start ``python -m driveflow.jobs.worker serve``, see its heartbeat, finish a
short job, cancel a long one, and stop it cleanly with ``python -m driveflow.jobs.worker stop``
(a timestamp file; Windows has no SIGTERM). Also checks pid_alive on the running platform."""

import os
import subprocess
import sys
import time

import pytest

from driveflow.jobs.persistence import pid_alive
from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.runs.schemas import ExperimentSpec

pytestmark = pytest.mark.compat


def _wait(pred, timeout=90.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("condition not reached in time")


def _cli(root, *args):
    env = dict(os.environ, SPADE_WORKSPACE=str(root))
    return subprocess.Popen([sys.executable, "-m", "driveflow.jobs.worker", *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_pid_alive_on_this_platform():
    assert pid_alive(os.getpid()) is True
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=30)
    assert pid_alive(p.pid) in (False, None)  # a recycled pid can only make this None, never True by accident here


def test_heartbeat_completion_cancellation_and_clean_stop(tmp_path):
    s = JobService(tmp_path)
    proc = _cli(tmp_path, "serve", "--poll", "0.1")
    try:
        workers = _wait(lambda: s.live_workers())
        assert workers[0]["pid"] == proc.pid
        hb0 = workers[0]["heartbeat_ts"]
        short = s.submit(ExperimentSpec.create("selftest.steps", params={"steps": 3, "step_s": 0.05}))
        done = _wait(lambda: s.get(short.job_id) if s.get(short.job_id).terminal else None)
        assert done.state == JobState.COMPLETED
        assert _wait(lambda: (s.live_workers() or [{}])[0].get("heartbeat_ts", 0) > hb0)
        long = s.submit(ExperimentSpec.create("selftest.steps", params={"steps": 400, "step_s": 0.05}))
        _wait(lambda: (s.get(long.job_id).progress_done or 0) >= 2)
        s.cancel(long.job_id)
        cancelled = _wait(lambda: s.get(long.job_id) if s.get(long.job_id).terminal else None)
        assert cancelled.state == JobState.CANCELLED
        stop = _cli(tmp_path, "stop")
        assert stop.wait(timeout=60) == 0
        assert proc.wait(timeout=60) == 0, proc.stdout.read()
        assert s.live_workers() == []  # unregistered itself: a clean exit, not a kill
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
