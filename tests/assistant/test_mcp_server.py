"""V4 (round B, 2026-11-09): the local MCP server of the assistant's tools. No network, no LLM.

The same policies as the assistant: tools by state, evidence for every number, and nothing launched
without the user's explicit confirmation, asked through MCP elicitation and tied to the proposal hash.
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from driveflow.assistant import mcp_server
from driveflow.assistant.mcp_server import EXPOSED, Server

ROOT = Path(__file__).resolve().parents[2]
USER = "Simula el motor DC de stock con PI a 100 rad/s."
READY_OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"},
             {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s", "origin": "user"}]


class Client:
    """Drives a Server in memory; answers the server's elicitation requests with ``elicit``."""

    def __init__(self, elicitation: bool = True, elicit=None):
        self.elicit = elicit or (lambda params: {"action": "accept", "content": {"confirm": True}})
        self.requests_from_server = []
        self.capabilities = {"elicitation": {}} if elicitation else {}
        self.server = Server(inp=io.StringIO(), out=io.StringIO())
        self._id = 0

    def _exchange(self, msg: dict) -> dict | None:
        """Send one message; answer server requests until the reply to ``msg`` arrives."""
        out = self.server.out
        start = out.tell()
        inp_lines = []

        def feed():
            # responses to the server's own requests are read from its input stream
            self.server.inp = io.StringIO("".join(inp_lines))

        replies = []
        orig_send = self.server._send

        def send(m):
            orig_send(m)
            if "method" in m and "id" in m:  # a request from the server (elicitation)
                self.requests_from_server.append(m)
                inp_lines.append(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": self.elicit(m["params"])}) + "\n")
                feed()
            else:
                replies.append(m)

        self.server._send = send
        try:
            self.server.handle(msg)
        finally:
            self.server._send = orig_send
        return replies[-1] if replies else None

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        return self._exchange({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})

    def initialize(self):
        r = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": self.capabilities, "clientInfo": {"name": "test", "version": "0"}})
        self._exchange({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r

    def call(self, name: str, **arguments) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]


def _ready(c: Client) -> dict:
    c.initialize()
    r = c.call("update_draft", ops=READY_OPS, user_message=USER)
    assert not r["isError"], r
    return r


def _n_jobs() -> int:
    from driveflow.jobs.service import JobService

    return len(JobService().list())


def test_initialize_and_the_exposed_tools():
    c = Client()
    r = c.initialize()["result"]
    assert r["serverInfo"]["name"] == "spade-assistant" and r["protocolVersion"] == "2025-06-18" and "tools" in r["capabilities"]
    names = [t["name"] for t in c.request("tools/list")["result"]["tools"]]
    assert names == list(EXPOSED)
    assert {"describe_plant", "validate_draft", "estimate_cost", "propose_experiments", "run_experiments", "get_job_status",
            "summarize_results", "explain_spade", "search_docs"} <= set(names)
    assert all(t["inputSchema"]["type"] == "object" for t in c.request("tools/list")["result"]["tools"])


def test_a_read_tool_answers_from_spade():
    c = Client()
    c.initialize()
    r = c.call("describe_plant", plant="dc_perm_ex_v2")
    assert not r["isError"] and r["structuredContent"]["plant"] == "dc_perm_ex_v2"
    assert json.loads(r["content"][0]["text"]) == r["structuredContent"]


def test_a_number_without_evidence_is_refused():
    c = Client()
    c.initialize()
    r = c.call("update_draft", ops=READY_OPS)  # no user_message: 100 rad/s has no source
    assert r["isError"] and r["structuredContent"]["applied"] == []
    assert "the user did not write this quantity" in r["structuredContent"]["errors"][0]["error"]


def test_nothing_launches_before_a_proposal(monkeypatch):
    c = Client()
    _ready(c)
    before = _n_jobs()
    r = c.call("run_experiments")
    assert r["isError"] and r["structuredContent"]["launched"] is False
    assert c.requests_from_server == [] and _n_jobs() == before


def test_a_client_without_elicitation_cannot_launch():
    c = Client(elicitation=False)
    _ready(c)
    assert not c.call("propose_experiments")["isError"]
    before = _n_jobs()
    r = c.call("run_experiments")
    assert r["isError"] and "elicitation" in r["structuredContent"]["error"] and _n_jobs() == before


@pytest.mark.parametrize("answer", [{"action": "decline"}, {"action": "cancel"}, {"action": "accept", "content": {"confirm": False}}])
def test_a_user_who_does_not_confirm_launches_nothing(answer):
    c = Client(elicit=lambda params: answer)
    _ready(c)
    c.call("propose_experiments")
    before = _n_jobs()
    r = c.call("run_experiments")
    assert r["isError"] and "did not confirm" in r["structuredContent"]["error"] and _n_jobs() == before


def test_the_users_confirmation_launches_once_and_is_tied_to_the_proposal_hash():
    c = Client()
    _ready(c)
    proposal = c.call("propose_experiments")["structuredContent"]
    before = _n_jobs()
    r = c.call("run_experiments")
    assert not r["isError"], r
    assert _n_jobs() == before + 1 and r["structuredContent"]["job_id"]
    (ask,) = c.requests_from_server
    assert ask["method"] == "elicitation/create" and proposal["draft_hash"] in ask["params"]["message"]
    assert c.server.state.jobs[-1]["draft_hash"] == proposal["draft_hash"]
    again = c.call("run_experiments")  # one confirmation = one launch
    assert again["isError"] and _n_jobs() == before + 1


def test_a_draft_changed_after_the_proposal_needs_a_new_proposal():
    c = Client()
    _ready(c)
    c.call("propose_experiments")
    c.call("update_draft", ops=[{"op": "set", "field": "omega_ref_rad_s", "value": 120, "unit": "rad/s", "origin": "user"}], user_message="mejor 120 rad/s")
    before = _n_jobs()
    r = c.call("run_experiments")
    assert r["isError"] and c.requests_from_server == [] and _n_jobs() == before


def test_unknown_method_and_tool_are_json_rpc_errors():
    c = Client()
    c.initialize()
    assert c.request("resources/list")["error"]["code"] == -32601
    assert c.request("tools/call", {"name": "run_tuning", "arguments": {}})["error"]["code"] == -32602


def test_the_server_is_local_only():
    src = Path(mcp_server.__file__).read_text()
    for forbidden in ("import socket", "http.server", "urllib", "requests", "asyncio.start_server", "ollama"):
        assert forbidden not in src


def test_stdio_round_trip(tmp_path):
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "explain_spade", "arguments": {"topic": "Tuning Studio"}}}]
    env = {"PYTHONPATH": str(ROOT / "src"), "SPADE_WORKSPACE": str(tmp_path / "ws"), "PATH": "/usr/bin:/bin", "TF_CPP_MIN_LOG_LEVEL": "3"}
    proc = subprocess.run([sys.executable, "-m", "driveflow.assistant.mcp_server"], input="".join(json.dumps(m) + "\n" for m in msgs),
                          capture_output=True, text=True, timeout=180, env=env, cwd=ROOT)
    replies = [json.loads(line) for line in proc.stdout.splitlines()]  # stdout carries JSON-RPC only
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert len(replies[1]["result"]["tools"]) == len(EXPOSED)
    assert replies[2]["result"]["isError"] is False


def test_results_become_available_only_when_the_job_has_finished():
    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker

    c = Client()
    _ready(c)
    c.call("propose_experiments")
    job_id = c.call("run_experiments")["structuredContent"]["job_id"]
    early = c.call("summarize_results")
    assert early["isError"] and "has not finished" in early["structuredContent"]["error"]
    Worker().serve(once=True)
    assert JobService().get(job_id).state.value == "completed"
    done = c.call("summarize_results")
    assert not done["isError"], done
