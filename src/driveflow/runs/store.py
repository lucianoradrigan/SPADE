"""Durable run store -- expansion task E0 (spec Sec. 6, 12; TODO_EXPANSION_CLAUDE_CODE.md E0).

One SQLite database (standard library) at ``<workspace>/spade.db`` and one artifact directory per
run at ``<workspace>/runs/<run_id>/``. The workspace is ``$SPADE_WORKSPACE`` when set, else
``<repo>/workspace`` (git-ignored). The job queue (driveflow.jobs.persistence) shares the same
database file. Streamlit's session state is never the source of truth for any of this.

Reproducible commands are written relative to the workspace (``runs/<run_id>/spec.json``) so they
carry no private absolute paths; ``safe_command()`` rejects anything that looks like one.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
from importlib import metadata
from pathlib import Path, PurePosixPath

from driveflow.runs.schemas import ApprovalRecord, ErrorCode, ExperimentSpec, PlantProfile, Run, SpadeError, canonical_json

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_NAME = "spade.db"
WORKER_MODULE = "driveflow.jobs.worker"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_utc TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, experiment_id TEXT, job_id TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
                                 completed_utc TEXT, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS runs_by_time ON runs (completed_utc);
CREATE TABLE IF NOT EXISTS profiles (id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY (id, version));
CREATE TABLE IF NOT EXISTS approvals (id TEXT PRIMARY KEY, subject TEXT NOT NULL, created_utc TEXT NOT NULL, body TEXT NOT NULL);
"""


def workspace_root(root: str | os.PathLike | None = None) -> Path:
    path = Path(root) if root is not None else Path(os.environ.get("SPADE_WORKSPACE") or REPO_ROOT / "workspace")
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect(root: str | os.PathLike | None = None) -> sqlite3.Connection:
    """Autocommit connection (explicit BEGIN IMMEDIATE where a read-modify-write must be atomic).
    WAL lets the Streamlit process read while the worker writes."""
    conn = sqlite3.connect(workspace_root(root) / DB_NAME, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def environment_info() -> dict:
    """Versions and commit recorded with every run. Packages are read from their metadata, not
    imported (importing TensorFlow here would cost seconds for no information)."""
    versions = {}
    for pkg in ("driveflow", "numpy", "scipy", "pandas", "tensorflow", "gymnasium", "streamlit"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    commit, dirty = None, None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"python": platform.python_version(), "platform": platform.system(), "packages": versions, "commit": commit, "dirty_worktree": dirty}


_ABS_PATH = re.compile(r"(^|\s)(/|[A-Za-z]:\\|\\\\|~)")
_CREDENTIAL_URL = re.compile(r"://[^/\s]*:[^/\s]*@")


def safe_command(command: str) -> str:
    """Refuse commands that would leak a private absolute path or a URL with credentials."""
    if _ABS_PATH.search(command) or _CREDENTIAL_URL.search(command):
        raise SpadeError(ErrorCode.INVALID_SPEC, "reproducible command must not contain absolute paths or credentials")
    return command


def reproducible_command(run_id: str) -> str:
    return safe_command(f"python -m {WORKER_MODULE} run runs/{run_id}/spec.json")


class RunStore:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = workspace_root(root)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return _Closing(connect(self.root))

    # ---------------------------------------------------------------- experiments
    def save_experiment(self, spec: ExperimentSpec) -> ExperimentSpec:
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO experiments (id, kind, created_utc, body) VALUES (?, ?, ?, ?)",
                         (spec.experiment_id, spec.kind, spec.created_utc, _dump(spec)))
        return spec

    def get_experiment(self, experiment_id: str) -> ExperimentSpec:
        return ExperimentSpec.from_dict(self._one("SELECT body FROM experiments WHERE id = ?", experiment_id, "experiment"))

    # ---------------------------------------------------------------- runs
    def run_dir(self, run_id: str) -> Path:
        _check_id(run_id)
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_artifact(self, run_id: str, name: str, data: bytes | str) -> dict:
        """Writes runs/<run_id>/<name> and returns {path, bytes, sha256}; path is workspace-relative."""
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise SpadeError(ErrorCode.INVALID_SPEC, f"artifact name must be a relative path inside the run directory, got {name!r}")
        path = self.run_dir(run_id).joinpath(*rel.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return self.describe_artifact(run_id, name)

    def describe_artifact(self, run_id: str, name: str) -> dict:
        path = self.run_dir(run_id).joinpath(*PurePosixPath(name).parts)
        return {"path": f"runs/{run_id}/{name}", "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    def artifact_path(self, artifact: dict) -> Path:
        return self.root.joinpath(*PurePosixPath(artifact["path"]).parts)

    def save_run(self, run: Run) -> Run:
        """Runs are written once; saving an existing id is an error (a Run is a historical fact)."""
        try:
            with self._conn() as conn:
                conn.execute("INSERT INTO runs (id, experiment_id, job_id, kind, status, completed_utc, body) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (run.run_id, run.experiment_id, run.job_id, run.kind, run.status, run.completed_utc, _dump(run)))
        except sqlite3.IntegrityError:
            raise SpadeError(ErrorCode.INVALID_SPEC, f"run {run.run_id} already exists; runs are immutable") from None
        return run

    def get_run(self, run_id: str) -> Run:
        return Run.from_dict(self._one("SELECT body FROM runs WHERE id = ?", run_id, "run"))

    def list_runs(self, limit: int = 200, status: str | None = None) -> list[Run]:
        sql, args = "SELECT body FROM runs", []
        if status:
            sql, args = sql + " WHERE status = ?", [status]
        with self._conn() as conn:
            rows = conn.execute(sql + " ORDER BY completed_utc DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        return [Run.from_dict(_load(r["body"])) for r in rows]

    # ---------------------------------------------------------------- profiles
    def save_profile(self, profile: PlantProfile) -> PlantProfile:
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO profiles (id, version, body) VALUES (?, ?, ?)", (profile.profile_id, profile.version, _dump(profile)))
        return profile

    def get_profile(self, profile_id: str, version: int | None = None) -> PlantProfile:
        if version is None:
            body = self._one("SELECT body FROM profiles WHERE id = ? ORDER BY version DESC LIMIT 1", profile_id, "profile")
        else:
            with self._conn() as conn:
                row = conn.execute("SELECT body FROM profiles WHERE id = ? AND version = ?", (profile_id, version)).fetchone()
            if row is None:
                raise SpadeError(ErrorCode.NOT_FOUND, f"profile {profile_id} v{version} not found")
            body = _load(row["body"])
        return PlantProfile.from_dict(body)

    # ---------------------------------------------------------------- approvals
    def record_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        """Append-only. Enforced here, not by a disabled button: an 'approved' decision is refused
        while the gate report has blocking items, and the candidate run must exist."""
        if record.decision == "approved" and record.gate_report.get("blocking"):
            raise SpadeError(ErrorCode.APPROVAL_BLOCKED, "Approval blocked. No default has been changed. Blocking: " + "; ".join(map(str, record.gate_report["blocking"])))
        self.get_run(record.candidate_run)
        if record.baseline_run:
            self.get_run(record.baseline_run)
        try:
            with self._conn() as conn:
                conn.execute("INSERT INTO approvals (id, subject, created_utc, body) VALUES (?, ?, ?, ?)",
                             (record.approval_id, record.subject, record.user_confirmed_at, _dump(record)))
        except sqlite3.IntegrityError:
            raise SpadeError(ErrorCode.INVALID_SPEC, f"approval {record.approval_id} already recorded; approval records are immutable") from None
        return record

    def list_approvals(self, subject: str | None = None) -> list[ApprovalRecord]:
        sql, args = "SELECT body FROM approvals", ()
        if subject:
            sql, args = sql + " WHERE subject = ?", (subject,)
        with self._conn() as conn:
            rows = conn.execute(sql + " ORDER BY created_utc", args).fetchall()
        return [ApprovalRecord.from_dict(_load(r["body"])) for r in rows]

    # ---------------------------------------------------------------- helpers
    def _one(self, sql: str, key: str, what: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(sql, (key,)).fetchone()
        if row is None:
            raise SpadeError(ErrorCode.NOT_FOUND, f"{what} {key} not found")
        return _load(row["body"])


class _Closing:
    """sqlite3's own context manager commits but does not close; this one closes."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, *exc) -> None:
        self.conn.close()


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _check_id(value: str) -> None:
    if not _ID.match(value or ""):
        raise SpadeError(ErrorCode.INVALID_SPEC, f"invalid id {value!r}")


def _dump(entity) -> str:
    return canonical_json(entity.to_dict())


def _load(body: str) -> dict:
    return json.loads(body)
