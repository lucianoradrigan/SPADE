"""SPADE's assistant tools as a local MCP server (V4, round B, 2026-11-09). Standard library only.

    python -m driveflow.assistant.mcp_server          # MCP over stdio (newline-delimited JSON-RPC 2.0)

Local only: the server reads stdin and writes stdout, opens no socket and calls no LLM. The MCP client
(its model and its user) plays the role of the assistant's chat; this server keeps one assistant session
(``SessionState``, configuration mode) per process and applies **the same policies** as the assistant:

- every call goes through ``policy.check_call`` (tools by state; ``run_experiments`` only after
  ``propose_experiments`` on the current draft);
- every number in the draft needs its evidence (``Draft.patch`` with the ledger). The user's own words reach
  the ledger through ``update_draft``'s ``user_message`` argument and are recorded as relayed by the MCP
  client, since this server never sees the user directly;
- **nothing is launched without the user's explicit confirmation**: ``run_experiments`` asks the client's
  USER through MCP elicitation (``elicitation/create``) for that proposal and its draft hash. Only an
  ``accept`` with the confirmation ticked sets the confirmation mark (``SessionState.confirm``), exactly
  as the Run button does in the GUI. A client without the elicitation capability cannot launch; there
  is no tool, argument or token that confirms.

Runs launched here are labelled ``SIMULATED · AI-PLANNED`` like the assistant's; a SPADE worker executes them.
"""

from __future__ import annotations

import json
import sys
from typing import IO

from driveflow.assistant import catalog, tools
from driveflow.assistant.core.loop import Tool, ToolFailure
from driveflow.assistant.core.policy import SessionState, check_call
from driveflow.assistant.draft import PatchError

SERVER_INFO = {"name": "spade-assistant", "title": "SPADE assistant tools (local)", "version": "0.2.0"}
#: MCP protocol revisions this server speaks (elicitation exists from 2025-06-18).
PROTOCOL_VERSIONS = ("2025-06-18",)
#: The assistant's tools exposed over MCP, in this order (task V4 list, plus update_draft, needed to build a draft).
EXPOSED = ("describe_plant", "update_draft", "validate_draft", "estimate_cost", "propose_experiments", "run_experiments",
           "get_job_status", "summarize_results", "explain_spade", "search_docs")
INSTRUCTIONS = ("SPADE assistant tools. Build a draft with update_draft (pass the user's own message as user_message so that their "
                "numbers have a source), check it with validate_draft, call propose_experiments and show the table to the user. "
                "run_experiments asks the user for confirmation through the client; it launches nothing without it. All results are SIMULATED.")


def _tools() -> dict[str, Tool]:
    by_name = {t.name: t for t in tools.CONFIG_TOOLS + catalog.CATALOG_TOOLS}
    out = {name: by_name[name] for name in EXPOSED}
    upd = out["update_draft"]
    params = json.loads(json.dumps(upd.parameters))
    params["properties"]["user_message"] = {"type": "string", "description": "the user's latest message, verbatim; numbers set with origin 'user' must appear in it"}
    out["update_draft"] = Tool(upd.name, upd.description + " Pass the user's message as user_message.", params, upd.fn)
    return out


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class Server:
    """One MCP session over a pair of text streams (stdin/stdout by default)."""

    def __init__(self, inp: IO[str] = None, out: IO[str] = None, state: SessionState | None = None):
        self.inp = inp if inp is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self.state = state or SessionState(mode="configuration")
        self.tools = _tools()
        self.client_capabilities: dict = {}
        self.initialized = False
        self._next_id = 0
        self._pending: list[dict] = []  # messages read while waiting for a response to our own request

    # ------------------------------------------------------------------------------------- transport
    def _send(self, msg: dict) -> None:
        self.out.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
        self.out.flush()

    def _read(self) -> dict | None:
        if self._pending:
            return self._pending.pop(0)
        while True:
            line = self.inp.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                self._send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})

    def _request_client(self, method: str, params: dict) -> dict:
        """Send a request to the client and wait for its response (other messages are queued)."""
        self._next_id += 1
        rid = f"spade-{self._next_id}"
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        queued = []
        try:
            while True:
                msg = self._read()
                if msg is None:
                    raise ToolFailure("the client closed the connection before answering")
                if msg.get("id") == rid and "method" not in msg:
                    if "error" in msg:
                        raise ToolFailure(f"the client refused {method}: {msg['error'].get('message')}")
                    return msg.get("result") or {}
                queued.append(msg)
        finally:
            self._pending = queued + self._pending

    def serve(self) -> None:
        while (msg := self._read()) is not None:
            self.handle(msg)

    # ------------------------------------------------------------------------------------- protocol
    def handle(self, msg: dict) -> None:
        if msg.get("jsonrpc") != "2.0" or "method" not in msg:
            if "id" in msg and "method" not in msg:
                return  # a stray response
            self._send({"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32600, "message": "invalid request"}})
            return
        method, mid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        is_notification = "id" not in msg
        try:
            result = self._dispatch(method, params)
        except JsonRpcError as exc:
            if not is_notification:
                self._send({"jsonrpc": "2.0", "id": mid, "error": {"code": exc.code, "message": str(exc)}})
            return
        if not is_notification:
            self._send({"jsonrpc": "2.0", "id": mid, "result": result})

    def _dispatch(self, method: str, params: dict) -> dict | None:
        if method == "initialize":
            self.client_capabilities = params.get("capabilities") or {}
            requested = params.get("protocolVersion")
            return {"protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                    "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.parameters} for t in self.tools.values()]}
        if method == "tools/call":
            return self.call_tool(params.get("name"), params.get("arguments") or {})
        raise JsonRpcError(-32601, f"method not found: {method}")

    # ------------------------------------------------------------------------------------- tools
    @staticmethod
    def _result(payload: dict, error: bool = False) -> dict:
        return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}],
                "structuredContent": payload, "isError": error}

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name not in self.tools:
            raise JsonRpcError(-32602, f"unknown tool {name!r}; available: {list(self.tools)}")
        arguments = dict(arguments)
        if name == "update_draft" and arguments.get("user_message"):
            self._user_message(str(arguments.pop("user_message")))
        arguments.pop("user_message", None)
        if self.state.jobs:
            tools.refresh_jobs(self.state)  # as the assistant's loop does before every model call
        if name == "run_experiments":
            refusal = self._confirm_with_user()
            if refusal:
                return self._result({"error": refusal, "launched": False}, error=True)
        refusal = check_call(self.state, name)
        if refusal:
            return self._result({"error": refusal}, error=True)
        try:
            result = self.tools[name].fn(self.state, **arguments)
        except (ToolFailure, PatchError) as exc:
            return self._result({"error": str(exc)}, error=True)
        except TypeError as exc:
            return self._result({"error": f"bad arguments: {exc}"}, error=True)
        self.state.ledger.add_tool_result(name, f"mcp-{self.state.seq}", result)
        payload = result if isinstance(result, dict) else {"result": result}
        return self._result(payload, error=payload.get("ok") is False)  # update_draft refuses all-or-none with ok: false

    def _user_message(self, text: str) -> int:
        seq = self.state.next_seq()
        self.state.last_user_seq = seq
        self.state.ledger.add_user_message(seq, text)
        return seq

    def _confirm_with_user(self) -> str | None:
        """Ask the client's user to confirm the current proposal (MCP elicitation). None = confirmed."""
        p = self.state.proposal
        if p is None or p.draft_hash != self.state.draft.hash():
            return check_call(self.state, "run_experiments") or "nothing to confirm: call propose_experiments on the current draft"
        if "elicitation" not in self.client_capabilities:
            return ("this MCP client cannot ask its user for confirmation (no 'elicitation' capability), so nothing is launched. "
                    "Launch from SPADE's Assistant page or CLI instead.")
        n = len(self.state.draft.conditions())
        answer = self._request_client("elicitation/create", {
            "message": (f"SPADE: launch {n} SIMULATED experiment(s) of plant {self.state.draft.plant} as one job? "
                        f"Proposal draft hash {p.draft_hash}. Nothing runs unless you confirm."),
            "requestedSchema": {"type": "object", "properties": {
                "confirm": {"type": "boolean", "title": "Launch these experiments", "description": f"draft hash {p.draft_hash}"}},
                "required": ["confirm"]}})
        if answer.get("action") != "accept" or (answer.get("content") or {}).get("confirm") is not True:
            return f"the user did not confirm (action {answer.get('action')!r}); nothing was launched"
        seq = self._user_message("[confirmation through MCP elicitation]")
        if not self.state.confirm(seq, f"mcp elicitation: {p.draft_hash}"):
            return "the confirmation could not be recorded; propose again"
        return None


def main() -> int:
    Server().serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
