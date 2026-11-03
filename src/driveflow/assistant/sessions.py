"""Persistence of the assistant's chat sessions (task F1.8, plan §8).

Three tables in the run store's SQLite database (runs/store.py ``_SCHEMA``, ``CREATE TABLE IF NOT
EXISTS``, so no migration): ``assistant_sessions`` (model ``tag@digest``, prompt version
``system_v1.md@<sha256[:12]>``, status, final body), ``assistant_messages`` (every user, assistant and
tool message, in order) and ``assistant_launches`` (the exact draft launched, with the origin of every
value, and the user message that confirmed it). The prompt is NOT stored in ExperimentSpec.params,
so it does not change any experiment's config_hash; experiments, runs and their Parquet files are the
existing ones (origin SIMULATED · AI-PLANNED).
"""

from __future__ import annotations

import json

from driveflow.runs.schemas import new_id, utc_now
from driveflow.runs.store import RunStore, connect

STATUSES = ("open", "closed")


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str, sort_keys=True)


class SessionStore:
    def __init__(self, root=None):
        self.store = RunStore(root)  # creates the tables
        self.root = self.store.root

    def _conn(self):
        return connect(self.root)

    def start(self, model_id: str, prompt_version: str, body: dict | None = None) -> str:
        if "@" not in model_id or "@" not in prompt_version:
            raise ValueError("model_id must be 'tag@digest' and prompt_version 'file@sha256[:12]'")
        sid = new_id("ast")
        conn = self._conn()
        try:
            conn.execute("INSERT INTO assistant_sessions (id, created_utc, model_id, prompt_version, status, body) VALUES (?, ?, ?, ?, 'open', ?)",
                         (sid, utc_now(), model_id, prompt_version, _j(body or {})))
        finally:
            conn.close()
        return sid

    def record_messages(self, session_id: str, messages: list, first_seq: int = 0) -> int:
        """Append messages (Ollama chat format) with consecutive seq numbers from first_seq; returns the next seq."""
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for k, m in enumerate(messages):
                conn.execute("INSERT INTO assistant_messages (session_id, seq, role, created_utc, body) VALUES (?, ?, ?, ?, ?)",
                             (session_id, first_seq + k, m.get("role", "?"), utc_now(), _j(m)))
            conn.execute("COMMIT")
        finally:
            conn.close()
        return first_seq + len(messages)

    def record_launch(self, session_id: str, job: dict) -> None:
        if job.get("confirmed_by_message_seq") is None or "draft_snapshot" not in job:
            raise ValueError("a launch is recorded only with its draft snapshot and the confirming user message")
        conn = self._conn()
        try:
            conn.execute("INSERT OR IGNORE INTO assistant_launches (session_id, experiment_id, job_id, confirmed_by_message_seq, draft_snapshot) VALUES (?, ?, ?, ?, ?)",
                         (session_id, job["experiment_id"], job["job_id"], int(job["confirmed_by_message_seq"]), _j(job["draft_snapshot"])))
        finally:
            conn.close()

    def close(self, session_id: str, body: dict) -> None:
        conn = self._conn()
        try:
            conn.execute("UPDATE assistant_sessions SET status = 'closed', body = ? WHERE id = ?", (_j(body), session_id))
        finally:
            conn.close()

    def load(self, session_id: str) -> dict:
        conn = self._conn()
        try:
            s = conn.execute("SELECT * FROM assistant_sessions WHERE id = ?", (session_id,)).fetchone()
            if s is None:
                raise KeyError(session_id)
            msgs = conn.execute("SELECT seq, role, body FROM assistant_messages WHERE session_id = ? ORDER BY seq", (session_id,)).fetchall()
            launches = conn.execute("SELECT * FROM assistant_launches WHERE session_id = ? ORDER BY experiment_id", (session_id,)).fetchall()
        finally:
            conn.close()
        return {"id": s["id"], "created_utc": s["created_utc"], "model_id": s["model_id"], "prompt_version": s["prompt_version"], "status": s["status"],
                "body": json.loads(s["body"]), "messages": [{"seq": m["seq"], "role": m["role"], **json.loads(m["body"])} for m in msgs],
                "launches": [{"experiment_id": x["experiment_id"], "job_id": x["job_id"], "confirmed_by_message_seq": x["confirmed_by_message_seq"],
                              "draft_snapshot": json.loads(x["draft_snapshot"])} for x in launches]}


class SessionRecorder:
    """Keeps one Agent's conversation persisted: call ``sync()`` after every user turn."""

    def __init__(self, agent, store: SessionStore, model_id: str, prompt_version: str, warnings: list | None = None):
        self.agent, self.store = agent, store
        self.session_id = store.start(model_id, prompt_version, {"warnings": list(warnings or [])})
        self._next_seq = 0
        self._launched: set = set()

    def sync(self) -> None:
        new = self.agent.messages[self._next_seq:]
        if new:
            self._next_seq = self.store.record_messages(self.session_id, new, self._next_seq)
        for job in self.agent.state.jobs:
            key = job.get("experiment_id")
            if key and key not in self._launched and job.get("draft_snapshot") is not None:
                self.store.record_launch(self.session_id, job)
                self._launched.add(key)

    def close(self, tokens: int | None = None) -> None:
        self.sync()
        self.store.close(self.session_id, {"final_draft": self.agent.state.draft.to_dict(), "links": [{"experiment_id": j.get("experiment_id"), "job_id": j["job_id"]}
                                                                                                      for j in self.agent.state.jobs], "tokens": tokens})
