"""Durable job queue in SQLite -- expansion task E0 (spec Sec. 6, 3.8).

Shares ``<workspace>/spade.db`` with driveflow.runs.store. Every read-modify-write runs inside
``BEGIN IMMEDIATE`` so two workers (or the GUI and a worker) cannot both claim or both finish the
same job. One lock row per engine serialises access to engines that accept a single command at a
time (PLECS RPC is blocking); native jobs use the "native" engine and also run one at a time per
engine, which keeps the E0 worker simple and honest about ETA.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import time

from driveflow.jobs.schemas import TERMINAL, Job, JobState, check_transition
from driveflow.runs.schemas import ErrorCode, SpadeError, new_id, utc_now
from driveflow.runs.store import connect, workspace_root

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, kind TEXT NOT NULL, engine TEXT NOT NULL, state TEXT NOT NULL,
    queued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, heartbeat_at TEXT, heartbeat_ts REAL,
    progress_done INTEGER, progress_total INTEGER, phase TEXT, worker_id TEXT, worker_pid INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0, resumable INTEGER NOT NULL DEFAULT 0, run_id TEXT, error TEXT, error_code TEXT);
CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs (state, queued_at);
CREATE TABLE IF NOT EXISTS job_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, utc TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS job_events_by_job ON job_events (job_id, seq);
CREATE TABLE IF NOT EXISTS workers (id TEXT PRIMARY KEY, pid INTEGER NOT NULL, host TEXT NOT NULL, started_at TEXT NOT NULL,
                                    heartbeat_ts REAL NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS engine_locks (engine TEXT PRIMARY KEY, job_id TEXT NOT NULL, acquired_at TEXT NOT NULL);
"""

_JOB_FIELDS = ("started_at", "finished_at", "heartbeat_at", "progress_done", "progress_total", "phase", "worker_id", "worker_pid",
               "cancel_requested", "resumable", "run_id", "error", "error_code")


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(job_id=row["id"], experiment_id=row["experiment_id"], kind=row["kind"], engine=row["engine"], state=JobState(row["state"]),
               queued_at=row["queued_at"], started_at=row["started_at"], finished_at=row["finished_at"], heartbeat_at=row["heartbeat_at"],
               progress_done=row["progress_done"], progress_total=row["progress_total"], phase=row["phase"], worker_id=row["worker_id"],
               worker_pid=row["worker_pid"], cancel_requested=bool(row["cancel_requested"]), resumable=bool(row["resumable"]),
               run_id=row["run_id"], error=row["error"], error_code=row["error_code"])


def pid_alive(pid: int) -> bool | None:
    """True/False on POSIX; None where it cannot be checked cheaply (Windows) -- callers then fall
    back to the heartbeat age."""
    if os.name != "posix":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobDB:
    def __init__(self, root=None):
        self.root = workspace_root(root)
        conn = connect(self.root)
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _tx(self):
        return _Tx(connect(self.root))

    def _read(self):
        return _Tx(connect(self.root), immediate=False)

    # ---------------------------------------------------------------- jobs
    def insert(self, experiment_id: str, kind: str, engine: str) -> Job:
        job_id = new_id("job")
        with self._tx() as conn:
            conn.execute("INSERT INTO jobs (id, experiment_id, kind, engine, state, queued_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (job_id, experiment_id, kind, engine, JobState.QUEUED.value, utc_now()))
            _event(conn, job_id, "info", "Job queued. You can leave this page while the worker runs.")
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise SpadeError(ErrorCode.NOT_FOUND, f"job {job_id} not found")
        return _row_to_job(row)

    def list(self, states=None, search: str | None = None, limit: int = 200) -> list[Job]:
        sql, args = "SELECT * FROM jobs WHERE 1=1", []
        if states:
            states = [JobState(s).value for s in states]
            sql += f" AND state IN ({','.join('?' * len(states))})"
            args += states
        if search:
            sql += " AND (id LIKE ? OR kind LIKE ? OR experiment_id LIKE ? OR IFNULL(run_id, '') LIKE ?)"
            args += [f"%{search}%"] * 4
        with self._read() as conn:
            rows = conn.execute(sql + " ORDER BY rowid DESC LIMIT ?", (*args, limit)).fetchall()
        return [_row_to_job(r) for r in rows]

    def counts(self) -> dict:
        with self._read() as conn:
            rows = conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state").fetchall()
        return {r["state"]: r["n"] for r in rows}

    def transition(self, job_id: str, new_state: JobState, **fields) -> Job:
        with self._tx() as conn:
            self._transition(conn, job_id, JobState(new_state), fields)
        return self.get(job_id)

    def _transition(self, conn, job_id, new_state, fields):
        row = conn.execute("SELECT state, engine FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise SpadeError(ErrorCode.NOT_FOUND, f"job {job_id} not found")
        check_transition(JobState(row["state"]), new_state)
        unknown = set(fields) - set(_JOB_FIELDS)
        if unknown:
            raise ValueError(f"unknown job fields {sorted(unknown)}")
        if new_state in TERMINAL:
            fields.setdefault("finished_at", utc_now())
            conn.execute("DELETE FROM engine_locks WHERE engine = ? AND job_id = ?", (row["engine"], job_id))
        sets = ", ".join(["state = ?"] + [f"{k} = ?" for k in fields])
        conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (new_state.value, *fields.values(), job_id))

    def update_progress(self, job_id: str, done: int | None, total: int | None, phase: str | None) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE jobs SET progress_done = ?, progress_total = ?, phase = ? WHERE id = ?", (done, total, phase, job_id))

    def request_cancel(self, job_id: str) -> Job:
        """Queued -> cancelled at once. Preparing/running -> cancelling; the worker confirms.
        Terminal jobs are left as they are (cancelling them would rewrite history)."""
        with self._tx() as conn:
            row = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise SpadeError(ErrorCode.NOT_FOUND, f"job {job_id} not found")
            state = JobState(row["state"])
            if state in TERMINAL:
                raise SpadeError(ErrorCode.INVALID_TRANSITION, f"job {job_id} already ended as {state.value}")
            if state == JobState.QUEUED:
                self._transition(conn, job_id, JobState.CANCELLED, {"cancel_requested": 1, "error": "Cancelled before it started."})
                _event(conn, job_id, "info", "Cancelled while queued; nothing was executed.")
            elif state != JobState.CANCELLING:
                self._transition(conn, job_id, JobState.CANCELLING, {"cancel_requested": 1})
                _event(conn, job_id, "info", "Cancellation requested. The current solver call may need to finish.")
        return self.get(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        with self._read() as conn:
            row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def claim_next(self, worker_id: str, pid: int, job_id: str | None = None) -> Job | None:
        """Atomically: oldest queued job (or ``job_id``) whose engine is free -> preparing, engine
        locked, run id assigned. None when nothing is claimable."""
        sql = "SELECT id, engine FROM jobs WHERE state = ? AND engine NOT IN (SELECT engine FROM engine_locks)"
        args = [JobState.QUEUED.value]
        if job_id:
            sql, args = sql + " AND id = ?", args + [job_id]
        with self._tx() as conn:
            row = conn.execute(sql + " ORDER BY rowid LIMIT 1", args).fetchone()
            if row is None:
                return None
            conn.execute("INSERT INTO engine_locks (engine, job_id, acquired_at) VALUES (?, ?, ?)", (row["engine"], row["id"], utc_now()))
            self._transition(conn, row["id"], JobState.PREPARING, {"worker_id": worker_id, "worker_pid": pid, "started_at": utc_now(),
                                                                   "heartbeat_at": utc_now(), "run_id": new_id("run")})
            conn.execute("UPDATE jobs SET heartbeat_ts = ? WHERE id = ?", (time.time(), row["id"]))
            _event(conn, row["id"], "info", f"Claimed by worker {worker_id} (pid {pid}).")
            job_id = row["id"]
        return self.get(job_id)

    def engine_lock(self, engine: str) -> str | None:
        with self._read() as conn:
            row = conn.execute("SELECT job_id FROM engine_locks WHERE engine = ?", (engine,)).fetchone()
        return row["job_id"] if row else None

    # ---------------------------------------------------------------- events
    def log(self, job_id: str, message: str, level: str = "info") -> None:
        with self._tx() as conn:
            _event(conn, job_id, level, message)

    def events(self, job_id: str, after_seq: int = 0) -> list[dict]:
        with self._read() as conn:
            rows = conn.execute("SELECT seq, utc, level, message FROM job_events WHERE job_id = ? AND seq > ? ORDER BY seq", (job_id, after_seq)).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- workers
    def register_worker(self, worker_id: str, pid: int) -> None:
        with self._tx() as conn:
            conn.execute("INSERT OR REPLACE INTO workers (id, pid, host, started_at, heartbeat_ts, state) VALUES (?, ?, ?, ?, ?, 'idle')",
                         (worker_id, pid, socket.gethostname(), utc_now(), time.time()))

    def heartbeat(self, worker_id: str, job_id: str | None = None, state: str = "idle") -> None:
        now = time.time()
        with self._tx() as conn:
            conn.execute("UPDATE workers SET heartbeat_ts = ?, state = ? WHERE id = ?", (now, state, worker_id))
            if job_id:
                conn.execute("UPDATE jobs SET heartbeat_ts = ?, heartbeat_at = ? WHERE id = ?", (now, utc_now(), job_id))

    def unregister_worker(self, worker_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM workers WHERE id = ?", (worker_id,))

    def live_workers(self, stale_after_s: float = 15.0) -> list[dict]:
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM workers WHERE heartbeat_ts >= ?", (time.time() - stale_after_s,)).fetchall()
        return [dict(r) for r in rows if pid_alive(r["pid"]) is not False or r["host"] != socket.gethostname()]

    def recover_orphans(self, stale_after_s: float = 30.0) -> list[str]:
        """Close active jobs whose worker is gone: its pid is dead on this host, or its heartbeat
        is older than ``stale_after_s``. Returns the recovered job ids."""
        host, now, recovered = socket.gethostname(), time.time(), []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT j.id, j.worker_pid, j.heartbeat_ts, j.cancel_requested, j.phase, w.host FROM jobs j LEFT JOIN workers w ON w.id = j.worker_id "
                "WHERE j.state IN (?, ?, ?)", (JobState.PREPARING.value, JobState.RUNNING.value, JobState.CANCELLING.value)).fetchall()
            for r in rows:
                same_host = r["host"] in (None, host)
                dead = same_host and r["worker_pid"] is not None and pid_alive(r["worker_pid"]) is False
                stale = r["heartbeat_ts"] is None or now - r["heartbeat_ts"] > stale_after_s
                if not (dead or stale):
                    continue
                why = f"pid {r['worker_pid']} is not running" if dead else f"no heartbeat for more than {stale_after_s:.0f} s"
                if r["cancel_requested"]:
                    self._transition(conn, r["id"], JobState.CANCELLED, {"error": f"Worker ended while cancelling ({why}).", "error_code": ErrorCode.WORKER_LOST.value})
                else:
                    self._transition(conn, r["id"], JobState.FAILED, {"error": f"Worker process ended unexpectedly ({why}); last phase: {r['phase'] or 'unknown'}.",
                                                                      "error_code": ErrorCode.WORKER_LOST.value})
                _event(conn, r["id"], "error", f"Recovered after worker loss: {why}. Resume unavailable — no recoverable checkpoint was saved.")
                recovered.append(r["id"])
            for w in conn.execute("SELECT id, pid, host, heartbeat_ts FROM workers").fetchall():
                if (w["host"] == host and pid_alive(w["pid"]) is False) or now - w["heartbeat_ts"] > stale_after_s:
                    conn.execute("DELETE FROM workers WHERE id = ?", (w["id"],))
        return recovered


def _event(conn, job_id: str, level: str, message: str) -> None:
    conn.execute("INSERT INTO job_events (job_id, utc, level, message) VALUES (?, ?, ?, ?)", (job_id, utc_now(), level, message))


class _Tx:
    def __init__(self, conn: sqlite3.Connection, immediate: bool = True):
        self.conn, self.immediate = conn, immediate

    def __enter__(self) -> sqlite3.Connection:
        if self.immediate:
            self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.immediate:
                self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.conn.close()
