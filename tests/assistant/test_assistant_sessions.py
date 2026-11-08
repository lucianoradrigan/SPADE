"""F1.8: chat sessions persisted in the run store (tables created with IF NOT EXISTS, no migration)."""

import sqlite3

import pytest

from driveflow.assistant import tools as T
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import Agent
from driveflow.assistant.sessions import SessionRecorder, SessionStore
from driveflow.runs.store import RunStore, workspace_root


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg}}


OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"}, {"op": "set", "field": "duration_s", "value": 0.1, "unit": "s"}]


def test_tables_exist_after_the_store_opens_an_old_database(tmp_path):
    # an older workspace: the database exists without the assistant tables
    conn = sqlite3.connect(tmp_path / "spade.db")
    conn.execute("CREATE TABLE experiments (id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_utc TEXT NOT NULL, body TEXT NOT NULL)")
    conn.close()
    RunStore(tmp_path)
    names = {r[0] for r in sqlite3.connect(tmp_path / "spade.db").execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"assistant_sessions", "assistant_messages", "assistant_launches", "experiments", "runs"} <= names


def test_session_records_messages_launch_and_close():
    agent = Agent(llm.ReplayClient({"format": 1, "exchanges": [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)),
                                                                 reply("", call("propose_experiments")), reply("¿Lanzo?"), reply("", call("run_experiments")), reply("Lanzado.")]},
                                   strict=False), T.CONFIG_TOOLS, system_prompt="sys v1")
    store = SessionStore(workspace_root())
    rec = SessionRecorder(agent, store, "fake:1b@sha256:abc", "system_v1.md@0123456789ab", warnings=["remote runtime"])
    agent.user_turn("motor DC con PI, 0.1 s")
    rec.sync()
    agent.user_turn("sí", confirmed_how="typed: sí")
    rec.close(tokens=42)
    s = store.load(rec.session_id)
    assert s["status"] == "closed" and s["model_id"] == "fake:1b@sha256:abc" and s["prompt_version"] == "system_v1.md@0123456789ab"
    assert [m["seq"] for m in s["messages"]] == list(range(len(agent.messages)))
    assert s["messages"][0]["role"] == "system" and any(m["role"] == "tool" for m in s["messages"])
    [launch] = s["launches"]
    job = agent.state.jobs[0]
    assert launch["job_id"] == job["job_id"] and launch["experiment_id"] == job["experiment_id"]
    assert launch["draft_snapshot"]["plant"] == "dc_perm_ex_v2" and launch["draft_snapshot"]["experiments"]["base"]["duration_s"]["origin"] == "user"
    user_seqs = [m["seq"] for m in s["messages"] if m["role"] == "user"]
    assert launch["confirmed_by_message_seq"] == agent.state.last_user_seq and len(user_seqs) == 2
    assert s["body"]["links"][0]["job_id"] == job["job_id"] and s["body"]["tokens"] == 42
    # the prompt is not part of the experiment (config_hash unchanged by it)
    spec = RunStore().get_experiment(job["experiment_id"])
    assert "sys v1" not in str(spec.to_dict()) and spec.params == {"origin": "SIMULATED · AI-PLANNED"}


def test_ids_are_checked():
    store = SessionStore(workspace_root())
    with pytest.raises(ValueError, match="tag@digest"):
        store.start("fake", "system_v1.md@x")
    sid = store.start("a@b", "p@q")
    with pytest.raises(ValueError, match="draft snapshot"):
        store.record_launch(sid, {"experiment_id": "e", "job_id": "j"})
