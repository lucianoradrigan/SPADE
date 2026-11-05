"""Phase 1, F1.2: configuration, Ollama client (against a fake Ollama on loopback, no real model),
replay and recording of cassettes. The one test that talks to the real runtime is @pytest.mark.llm."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from driveflow.assistant import config as acfg
from driveflow.assistant.core import llm
from driveflow.assistant.draft import DEFAULT_LIMITS

DIGEST = "sha256:" + "ab" * 32
TOOLS = [{"type": "function", "function": {"name": "describe_plant", "description": "d", "parameters": {"type": "object", "properties": {"plant": {"type": "string"}}}}}]


class FakeOllama:
    """A tiny HTTP server that answers like Ollama: /api/version, /api/tags, /api/chat."""

    def __init__(self, chat_replies, models=None):
        self.chat_replies = list(chat_replies)
        self.models = models if models is not None else [{"name": "fake:1b", "model": "fake:1b", "digest": DIGEST}]
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send(200, {"version": "0.0.fake"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": outer.models})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                code, obj = outer.chat_replies.pop(0)
                self._send(code, obj)

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def cfg(base_url, **kw):
    return acfg.load_config(overrides={"base_url": base_url, "model": {"tag": "fake:1b", "digest": DIGEST}, **kw})


def ok_reply(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return 200, {"model": "fake:1b", "message": msg, "done": True, "prompt_eval_count": 42, "eval_count": 7}


# --- configuration ------------------------------------------------------------------------------

def test_config_file_is_local_pinned_for_the_demo_and_matches_the_draft_limits():
    c = acfg.load_config()
    assert c["runtime"] == "ollama" and acfg.is_loopback(c["base_url"]) and c["allow_remote_runtime"] is False and c["warnings"] == []
    # Pinned by Luciano's decision on 2026-11-04 for the internal demo (docs/audit/assistant/f1_1_decision.md).
    assert c["model"]["tag"] == "qwen3.5:9b" and len(c["model"]["digest"]) == 64
    assert c["model"]["status"] == "internal-demo; F1.1 round-3 rule not met"
    assert c["temperature"] == 0 and isinstance(c["seed"], int)
    assert {k: c["limits"][k] for k in DEFAULT_LIMITS} == DEFAULT_LIMITS
    assert "provider" not in c and "privacy" not in c  # no cloud provider (B1)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "http://127.0.0.5:1"])
def test_loopback_urls_are_accepted(url):
    assert acfg.check_runtime_url(url, False) == []


@pytest.mark.parametrize("url", ["http://192.168.1.10:11434", "http://example.com:11434", "https://ollama.internal"])
def test_remote_urls_need_explicit_permission(url):
    with pytest.raises(acfg.RemoteRuntimeRefused, match="leave"):
        acfg.load_config(overrides={"base_url": url})
    c = acfg.load_config(overrides={"base_url": url, "allow_remote_runtime": True})
    assert c["warnings"] and "leave this machine" in c["warnings"][0]
    assert llm.OllamaClient(c).warnings == c["warnings"]  # every session carries the warning


def test_only_the_ollama_runtime_is_accepted():
    with pytest.raises(acfg.ConfigError, match="B1"):
        acfg.load_config(overrides={"runtime": "anthropic"})


# --- OllamaClient against a fake Ollama ----------------------------------------------------------

def test_chat_sends_determinism_options_and_parses_tool_calls():
    tc = [{"function": {"name": "describe_plant", "arguments": {"plant": "dc_perm_ex_v1"}}}]
    fo = FakeOllama([ok_reply("", tc)])
    with fo as url:
        client = llm.OllamaClient(cfg(url, context_tokens=8192, think=False))
        reply = client.chat([{"role": "user", "content": "hola"}], TOOLS, seed=3)
    call = reply.tool_calls[0]
    assert (call.name, call.arguments, call.error) == ("describe_plant", {"plant": "dc_perm_ex_v1"}, None)
    assert (reply.prompt_tokens, reply.completion_tokens) == (42, 7) and reply.latency_s >= 0
    assert client.checked and client.warnings == []  # version, model and digest checked before chatting
    p = fo.requests[0]
    assert p["model"] == "fake:1b" and p["stream"] is False and p["think"] is False and p["tools"] == TOOLS
    assert p["options"] == {"temperature": 0, "seed": 3, "num_ctx": 8192, "num_predict": 2048}  # num_predict: F1.1 round 2 output cap
    assert reply.to_message() == {"role": "assistant", "content": "", "tool_calls": tc}


@pytest.mark.parametrize("arguments, error", [
    ('{"plant": "dc_perm_ex_v1"}', None),  # string JSON is accepted
    ('{"plant": ', "not valid JSON"),
    ('["a"]', "JSON object"),
    ([1, 2], "JSON object"),
])
def test_tool_arguments_that_are_not_an_object_are_kept_as_errors(arguments, error):
    r = llm.parse_reply({"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "describe_plant", "arguments": arguments}}]}})
    call = r.tool_calls[0]
    if error is None:
        assert call.arguments == {"plant": "dc_perm_ex_v1"} and call.error is None
    else:
        assert call.arguments is None and error in call.error and call.raw_arguments is not None


def test_digest_mismatch_is_a_warning_not_an_error():
    other = [{"name": "fake:1b", "model": "fake:1b", "digest": "sha256:" + "cd" * 32}]
    with FakeOllama([ok_reply("x")], models=other) as url:
        client = llm.OllamaClient(cfg(url))
        info = client.check()
        client.chat([{"role": "user", "content": "x"}], [], seed=0)
    assert info["digest_match"] is False and "differs from the pinned" in info["warnings"][0]


def test_missing_model_and_unpinned_model():
    with FakeOllama([], models=[]) as url:
        with pytest.raises(llm.ModelNotAvailable, match="ollama pull fake:1b"):
            llm.OllamaClient(cfg(url)).check()
        unpinned = acfg.load_config(overrides={"base_url": url, "model": {"tag": None, "digest": None}})
        with pytest.raises(llm.ModelNotAvailable, match="F1.1"):
            llm.OllamaClient(unpinned).chat([], [], seed=0)


def test_runtime_errors_are_typed():
    with FakeOllama([(500, {"error": "out of memory"})]) as url:
        client = llm.OllamaClient(cfg(url))
        with pytest.raises(llm.LLMResponseError, match="out of memory"):
            client.chat([{"role": "user", "content": "x"}], [], seed=0)
    # Nothing listens on this port any more: the runtime is "down".
    with pytest.raises(llm.RuntimeUnavailable, match="Is Ollama installed and running"):
        llm.OllamaClient(cfg(url, timeout_s=2)).version()


# --- cassettes -----------------------------------------------------------------------------------

def test_record_then_replay_and_mismatch_detection(tmp_path):
    tc = [{"function": {"name": "describe_plant", "arguments": {"plant": "pmsm_v1"}}}]
    msgs = [{"role": "user", "content": "PMSM"}]
    with FakeOllama([ok_reply("", tc), ok_reply("listo")]) as url:
        rec = llm.RecordingClient(llm.OllamaClient(cfg(url)), model_id="fake:1b@" + DIGEST, runtime_version="0.0.fake")
        r1 = rec.chat(msgs, TOOLS, seed=0)
        follow = msgs + [r1.to_message(), {"role": "tool", "tool_name": "describe_plant", "content": "{}"}]
        rec.chat(follow, TOOLS, seed=0)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(rec.cassette()))
    replay = llm.ReplayClient(path)
    assert replay.chat(msgs, TOOLS, seed=0).tool_calls[0].arguments == {"plant": "pmsm_v1"}
    assert replay.chat(follow, TOOLS, seed=0).text == "listo"
    with pytest.raises(llm.CassetteMismatch, match="ran out"):
        replay.chat(follow, TOOLS, seed=0)
    changed_tools = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "description": "changed"}}]
    for m, t, s in ((msgs, changed_tools, 0), ([{"role": "user", "content": "PMSM!"}], TOOLS, 0), (msgs, TOOLS, 1)):
        with pytest.raises(llm.CassetteMismatch, match="differs from the recorded"):
            llm.ReplayClient(path).chat(m, t, seed=s)
    assert replay.model_id() == "fake:1b@" + DIGEST


def test_cassette_format_is_checked():
    with pytest.raises(llm.CassetteMismatch, match="format"):
        llm.ReplayClient({"format": 99, "exchanges": []})


# --- the real runtime (skipped unless --run-llm) ----------------------------------------------------

@pytest.mark.llm
def test_real_runtime_answers():
    client = llm.OllamaClient(acfg.load_config())
    assert client.version()
    assert isinstance(client.installed_models(), list)
