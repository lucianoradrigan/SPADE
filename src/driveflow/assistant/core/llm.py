"""Language-model clients of the assistant (task F1.2).

``LLMClient`` is the protocol the tool loop talks to: ``chat(messages, tools, *, seed) -> Reply``.
Messages and tools use Ollama's chat format directly (role/content/tool_calls/tool_name; tools as
{"type": "function", "function": {name, description, parameters}}), so nothing is translated.

- ``OllamaClient``: POST /api/chat on a local Ollama with the standard library only (decision B2),
  ``stream: false``, temperature 0 and a fixed seed (B3). It refuses a non-loopback runtime unless
  allowed (C4), refuses to run without a pinned model, and checks that the installed digest is the
  pinned one (a different digest is a warning recorded on the client, not an error).
- ``ReplayClient``: replays a recorded cassette and fails if a request differs from the recorded
  one (messages + tools + seed), so a prompt or tool-schema change shows up in the tests (B4).
- ``RecordingClient``: wraps a client and records every exchange into a cassette.

No test of the normal suite talks to a real Ollama; those carry ``@pytest.mark.llm``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from driveflow.assistant.config import check_runtime_url, load_config

CASSETTE_FORMAT = 1


class LLMError(RuntimeError):
    pass


class RuntimeUnavailable(LLMError):
    """The runtime does not answer (Ollama not installed, not running, wrong port)."""


class ModelNotAvailable(LLMError):
    """No model pinned, or the pinned model is not downloaded."""


class LLMResponseError(LLMError):
    """The runtime answered with an error or with something that is not a chat reply."""


class CassetteMismatch(LLMError):
    """A replayed request differs from the recorded one, or the cassette ran out."""


@dataclass
class ToolCall:
    name: str
    arguments: dict | None  # None when the model's arguments were not valid JSON / not an object
    id: str
    raw_arguments: str | None = None
    error: str | None = None

    def to_message(self) -> dict:
        """The tool call as it goes back into the conversation (Ollama format)."""
        return {"function": {"name": self.name, "arguments": self.arguments if self.arguments is not None else (self.raw_arguments or "")}}


@dataclass
class Reply:
    text: str
    tool_calls: list = field(default_factory=list)
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    thinking: str | None = None
    raw: dict | None = None

    def to_message(self) -> dict:
        msg = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [c.to_message() for c in self.tool_calls]
        return msg


class LLMClient(Protocol):
    def chat(self, messages: list, tools: list, *, seed: int, think: bool | None = None) -> Reply: ...


def parse_reply(data: dict, latency_s: float | None = None) -> Reply:
    """Turn an Ollama /api/chat response into a Reply. Invalid tool arguments do not raise: the
    call keeps ``error`` so the loop can hand it back to the model (task F1.3)."""
    if not isinstance(data, dict):
        raise LLMResponseError(f"chat reply is not a JSON object: {data!r:.200}")
    if "error" in data:
        raise LLMResponseError(f"runtime error: {data['error']}")
    msg = data.get("message")
    if not isinstance(msg, dict):
        raise LLMResponseError(f"chat reply has no message: {json.dumps(data)[:200]}")
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        args, raw, err = fn.get("arguments"), None, None
        if isinstance(args, str):
            raw = args
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                args, err = None, f"arguments are not valid JSON: {exc.msg}"
        if args is not None and not isinstance(args, dict):
            raw, args, err = json.dumps(args), None, "arguments must be a JSON object"
        if not name:
            err = err or "tool call without a name"
        calls.append(ToolCall(name=name or "", arguments=args, id=tc.get("id") or f"call_{i}", raw_arguments=raw, error=err))
    return Reply(text=msg.get("content") or "", tool_calls=calls, model=data.get("model"), prompt_tokens=data.get("prompt_eval_count"),
                 completion_tokens=data.get("eval_count"), latency_s=latency_s, thinking=msg.get("thinking"), raw=data)


def request_key(messages: list, tools: list, seed: int) -> str:
    """Canonical form of a request: what a cassette must match exactly."""
    return json.dumps({"messages": messages, "tools": tools, "seed": seed}, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------


class OllamaClient:
    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else load_config()
        self.base_url = self.config["base_url"].rstrip("/")
        self.warnings: list[str] = list(check_runtime_url(self.base_url, bool(self.config.get("allow_remote_runtime", False))))
        model = self.config.get("model") or {}
        self.tag, self.digest = model.get("tag"), model.get("digest")
        self.timeout_s = float(self.config.get("timeout_s", 180))
        self.checked = False

    # --- HTTP ------------------------------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                err = json.loads(body).get("error", body.decode(errors="replace"))
            except (json.JSONDecodeError, AttributeError):
                err = body.decode(errors="replace")
            if exc.code == 404 and "not found" in str(err):
                raise ModelNotAvailable(f"{err}. Download it with: ollama pull {self.tag}") from None
            raise LLMResponseError(f"{path}: HTTP {exc.code}: {err}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise RuntimeUnavailable(f"the local model runtime does not answer at {self.base_url} ({exc}). "
                                     "Is Ollama installed and running? (systemctl status ollama)") from None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise LLMResponseError(f"{path}: not JSON: {body[:200]!r}") from None

    def version(self) -> str:
        return self._request("/api/version", timeout=10).get("version", "?")

    def installed_models(self) -> list[dict]:
        return self._request("/api/tags", timeout=10).get("models", [])

    def check(self) -> dict:
        """Runtime version, the pinned model and whether its digest matches. Raises
        RuntimeUnavailable / ModelNotAvailable; a digest mismatch is a warning."""
        version = self.version()
        if not self.tag:
            raise ModelNotAvailable("no model is pinned in configs/assistant.yaml (model.tag); it is chosen in task F1.1")
        found = next((m for m in self.installed_models() if m.get("name") == self.tag or m.get("model") == self.tag), None)
        if found is None:
            raise ModelNotAvailable(f"model {self.tag!r} is not downloaded. Download it with: ollama pull {self.tag}")
        installed = found.get("digest")
        match = bool(self.digest) and installed == self.digest
        if not match:
            self.warnings.append(f"model {self.tag}: installed digest {installed} differs from the pinned {self.digest}; results may not reproduce")
        self.checked = True
        return {"runtime_version": version, "model": self.tag, "pinned_digest": self.digest, "installed_digest": installed, "digest_match": match,
                "warnings": list(self.warnings)}

    def chat(self, messages: list, tools: list, *, seed: int | None = None, think: bool | None = None) -> Reply:
        """``think`` overrides configs/assistant.yaml ``think`` for this call (F1.9: no thinking on the
        final answer without tools)."""
        if not self.tag:
            raise ModelNotAvailable("no model is pinned in configs/assistant.yaml (model.tag); it is chosen in task F1.1")
        if not self.checked:
            self.check()
        options = {"temperature": self.config.get("temperature", 0), "seed": self.config.get("seed", 0) if seed is None else seed}
        if self.config.get("context_tokens"):
            options["num_ctx"] = int(self.config["context_tokens"])
        if self.config.get("num_predict"):
            options["num_predict"] = int(self.config["num_predict"])  # output-token cap per call (F1.1 round 2)
        payload = {"model": self.tag, "messages": messages, "stream": False, "options": options}
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = bool(think)
        elif self.config.get("think") is not None:
            payload["think"] = bool(self.config["think"])
        t0 = time.perf_counter()
        data = self._request("/api/chat", payload)
        return parse_reply(data, time.perf_counter() - t0)

    def model_id(self) -> str:
        return f"{self.tag}@{self.digest}"


# ---------------------------------------------------------------------------------------------


class ReplayClient:
    """Replays a cassette: {"format", "model", "exchanges": [{"request": {...}, "reply": {...}}], ...}."""

    def __init__(self, cassette: dict | str | Path, *, strict: bool = True):
        if not isinstance(cassette, dict):
            cassette = json.loads(Path(cassette).read_text())
        if cassette.get("format") != CASSETTE_FORMAT:
            raise CassetteMismatch(f"cassette format {cassette.get('format')!r}, expected {CASSETTE_FORMAT}")
        self.cassette = cassette
        self.exchanges = list(cassette.get("exchanges", []))
        self.position = 0
        self.strict = strict

    def chat(self, messages: list, tools: list, *, seed: int = 0, think: bool | None = None) -> Reply:
        if self.position >= len(self.exchanges):
            raise CassetteMismatch(f"cassette ran out after {len(self.exchanges)} exchanges")
        ex = self.exchanges[self.position]
        if self.strict:
            got, want = request_key(messages, tools, seed), request_key(ex["request"]["messages"], ex["request"]["tools"], ex["request"]["seed"])
            if got != want:
                raise CassetteMismatch(f"request {self.position} differs from the recorded one (prompt, tools or conversation changed): "
                                       f"{_first_difference(got, want)}")
        self.position += 1
        return parse_reply(ex["reply"], ex.get("latency_s"))

    def model_id(self) -> str:
        return self.cassette.get("model", "replay")


def _first_difference(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at char {i}: got ...{a[max(0, i - 40):i + 40]!r}... recorded ...{b[max(0, i - 40):i + 40]!r}..."


class RecordingClient:
    """Wraps a client and records each exchange; ``cassette()`` returns the recording."""

    def __init__(self, inner, *, model_id: str, runtime_version: str | None = None, prompt_version: str | None = None):
        self.inner = inner
        self.exchanges: list[dict] = []
        self.meta = {"model": model_id, "runtime_version": runtime_version, "prompt_version": prompt_version,
                     "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def chat(self, messages: list, tools: list, *, seed: int = 0, think: bool | None = None) -> Reply:
        reply = self.inner.chat(messages, tools, seed=seed, **({"think": think} if think is not None else {}))
        self.exchanges.append({"request": {"messages": json.loads(json.dumps(messages)), "tools": json.loads(json.dumps(tools)), "seed": seed},
                               "reply": reply.raw, "latency_s": reply.latency_s})
        return reply

    def model_id(self) -> str:
        return self.meta["model"]

    def cassette(self, draft_events: list | None = None) -> dict:
        return {"format": CASSETTE_FORMAT, **self.meta, "exchanges": self.exchanges, "draft_events": draft_events or []}
