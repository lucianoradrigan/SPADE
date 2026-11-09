"""Decision C7 (2026-11-09): through MCP only configurations, simulated results and SPADE's documentation
leave the machine; no tool reads the user's files.

Every exposed tool is called (also with path-like inputs) under a Python audit hook that records each file
opened. A file outside SPADE (the repository: code, configs, docs), the SPADE workspace and Python's own
libraries fails the test, and so does a tool argument that names a file or a path.
"""

import io
import json
import sys
import sysconfig
from pathlib import Path

import pytest

from driveflow.assistant.mcp_server import EXPOSED, Server

ROOT = Path(__file__).resolve().parents[2]
PATH_WORDS = ("path", "file", "filename", "dir", "directory", "folder", "url", "uri")
_OPENED: list[str] = []
_ACTIVE = [False]


def _hook(event, args):
    if _ACTIVE[0] and event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        _OPENED.append(str(args[0] if not isinstance(args[0], bytes) else args[0].decode(errors="replace")))


sys.addaudithook(_hook)


def _allowed_roots(workspace: Path) -> list[Path]:
    roots = {ROOT.resolve(), workspace.resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    roots |= {Path(p).resolve() for p in sysconfig.get_paths().values() if p}
    return sorted(roots)


def _outside(paths, roots) -> list[str]:
    bad = []
    for p in paths:
        if p.startswith(("/proc/", "/dev/", "/sys/")) or p.isdigit():
            continue
        try:
            rp = Path(p).resolve()
        except (OSError, RuntimeError):
            continue
        if not any(rp == r or r in rp.parents for r in roots):
            bad.append(p)
    return bad


def test_no_tool_argument_names_a_file_or_a_path():
    server = Server(inp=io.StringIO(), out=io.StringIO())

    def props(schema):
        for name, spec in (schema.get("properties") or {}).items():
            yield name, spec
            if spec.get("type") == "array" and isinstance(spec.get("items"), dict):
                yield from props(spec["items"])
            if spec.get("type") == "object":
                yield from props(spec)

    for tool in server.tools.values():
        for name, spec in props(tool.parameters):
            assert not any(w in name.lower() for w in PATH_WORDS), (tool.name, name)
            assert spec.get("format") not in ("uri", "uri-reference", "iri"), (tool.name, name)


# Each exposed tool, with ordinary and with path-like arguments (a client could pass anything).
CALLS = [
    ("describe_plant", {"plant": "dc_perm_ex_v2"}), ("describe_plant", {"plant": "/etc/passwd"}),
    ("update_draft", {"ops": [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"},
                              {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s", "origin": "user"}],
                      "user_message": "PI a 100 rad/s; mis datos están en /etc/passwd y ~/.ssh/id_rsa"}),
    ("update_draft", {"ops": [{"op": "set_plant", "plant": "../../../../etc/passwd"}]}),
    ("validate_draft", {}), ("estimate_cost", {}), ("propose_experiments", {}), ("run_experiments", {}),
    ("get_job_status", {"job_id": "../../../../etc/passwd"}), ("summarize_results", {"job_id": "/etc/passwd"}),
    ("explain_spade", {"topic": "Tuning Studio"}), ("explain_spade", {"topic": "/etc/passwd"}),
    ("search_docs", {"query": "S-21 one-quadrant converter"}), ("search_docs", {"query": "/etc/passwd root:x:0:0"}),
    ("search_docs", {"query": "../../../.ssh/id_rsa"}),
]


def test_every_exposed_tool_is_exercised():
    assert {name for name, _ in CALLS} == set(EXPOSED)


def test_no_tool_opens_a_file_outside_spade_or_returns_one(tmp_path, monkeypatch):
    workspace = tmp_path / "spade_workspace"
    monkeypatch.setenv("SPADE_WORKSPACE", str(workspace))
    secret = tmp_path / "outside" / "user_secret.txt"
    secret.parent.mkdir()
    secret.write_text("USER-SECRET-7f3a")
    calls = CALLS + [("search_docs", {"query": str(secret)}), ("explain_spade", {"topic": str(secret)}), ("describe_plant", {"plant": str(secret)})]
    server = Server(inp=io.StringIO(), out=io.StringIO())
    server.client_capabilities = {}  # no elicitation: run_experiments must refuse, never launch
    _OPENED.clear()
    _ACTIVE[0] = True
    try:
        replies = [server.call_tool(name, dict(args)) for name, args in calls]
    finally:
        _ACTIVE[0] = False
    roots = _allowed_roots(workspace)
    assert _outside(_OPENED, roots) == [], "files opened outside SPADE, its workspace and Python"
    text = json.dumps(replies, default=str)
    assert "USER-SECRET-7f3a" not in text and "root:x:0:0:" not in text.replace("/etc/passwd root:x:0:0", "")
    assert not any(str(secret) == p for p in _OPENED)
