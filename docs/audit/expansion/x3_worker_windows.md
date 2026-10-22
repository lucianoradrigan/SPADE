# X3 — worker heartbeat on Windows (round PLECS/OM)

## What changed

- **`pid_alive` on Windows** (`jobs/persistence.py`): it used to return `None` outside POSIX, so a dead worker was detected only by heartbeat age (the T9 pending note). It now asks the OS through `ctypes` (standard library, no new dependency): `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` and `GetExitCodeProcess`, where `STILL_ACTIVE` means alive. Error 87 means no such process; error 5 (access denied) means it exists; anything else returns `None`, and the heartbeat-age fallback applies.
- **Clean stop on every OS:** `python -m driveflow.jobs.worker stop` writes `<workspace>/worker.stop` with a timestamp. Every worker that started before it finishes its current job, unregisters itself and exits with code 0. Before, a worker could only be stopped by a signal, and on Windows `terminate()` is `TerminateProcess`: a kill with no unregistration.
- **Test:** `tests/jobs/test_worker_lifecycle.py` (marker `compat`):
  - starts `serve`;
  - checks the heartbeat registration and that it advances;
  - completes a short job;
  - cancels a long one;
  - stops the worker with `stop` and asserts exit code 0 and no live worker left;
  - checks `pid_alive` for the current process and for a finished one.
- **CI** (`.github/workflows/tests.yml`, smoke matrix: Ubuntu, Windows, macOS): new step "Worker life cycle".

## Executed

| Platform | Command | Result |
|---|---|---|
| Linux (Ubuntu 26.04, Python 3.12.13) | `pytest tests/jobs` | 16 passed |
| Windows | CI smoke job `windows-latest` | **no ejecutado: solo CI.** It runs on the next push to a branch with a PR to `main`, or on `workflow_dispatch`. No Windows machine is available here |
| macOS | CI smoke job `macos-latest` | **no ejecutado: solo CI** |
