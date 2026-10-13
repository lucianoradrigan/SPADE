"""Per-session state (st.session_state, never st.cache_resource for anything mutable).

- One ServerAgent and one GatewayAgent per domain per SESSION (same lesson as the legacy LM tab:
  a cache_resource agent leaked one user's alerts into another's).
- A run log: every simulation/monitoring/TL run gets a run_id with its parameters FROZEN at run
  time, so changing a control afterwards never relabels an earlier result (guide Sec. 5.3)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import streamlit as st

from driveflow.monitoring.agents.agent_server import ServerAgent

MAX_RUN_LOG = 30


def server_agent() -> ServerAgent:
    if "lm_server_agent" not in st.session_state:
        st.session_state["lm_server_agent"] = ServerAgent()
    return st.session_state["lm_server_agent"]


def config_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def record_run(kind: str, params: dict, summary: dict | None = None) -> dict:
    """Appends a frozen run record and returns it ({run_id, kind, utc, params, config_hash, summary})."""
    n = st.session_state.get("_run_counter", 0) + 1
    st.session_state["_run_counter"] = n
    record = {
        "run_id": f"{kind}-{n:03d}",
        "kind": kind,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "params": json.loads(json.dumps(params, default=str)),
        "config_hash": config_hash(params),
        "summary": summary or {},
    }
    log = st.session_state.setdefault("_run_log", [])
    log.append(record)
    del log[:-MAX_RUN_LOG]
    return record


def run_log() -> list:
    return list(st.session_state.get("_run_log", []))
