"""AI-N32 (found 2026-11-09 by the fresh-clone test of round C): the assistant ignored OLLAMA_HOST, Ollama's standard
variable, and always used configs/assistant.yaml's base_url. It now honours it; the loopback rule still applies."""

import pytest

from driveflow.assistant import config as acfg


@pytest.mark.parametrize("value, url", [
    ("127.0.0.1:1", "http://127.0.0.1:1"), ("localhost", "http://localhost:11434"), (":12000", "http://127.0.0.1:12000"),
    ("0.0.0.0:11500", "http://127.0.0.1:11500"), ("0.0.0.0", "http://127.0.0.1:11434"), ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
    ("https://localhost:8443", "https://localhost:8443"), ("[::1]:11434", "http://[::1]:11434"),
])
def test_ollama_host_forms(value, url):
    assert acfg.base_url_from_ollama_host(value) == url


def test_without_ollama_host_the_config_file_decides(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = acfg.load_config()
    assert cfg["base_url"] == "http://127.0.0.1:11434" and cfg["base_url_source"] == "configs/assistant.yaml"


def test_ollama_host_is_honoured(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    cfg = acfg.load_config()
    assert cfg["base_url"] == "http://127.0.0.1:9" and cfg["base_url_source"] == "OLLAMA_HOST"


def test_a_remote_ollama_host_is_still_refused(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    with pytest.raises(acfg.RemoteRuntimeRefused):
        acfg.load_config()


def test_an_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    cfg = acfg.load_config(overrides={"base_url": "http://127.0.0.1:11434"})
    assert cfg["base_url"] == "http://127.0.0.1:11434" and cfg["base_url_source"] == "override"


def test_a_closed_port_gives_a_clear_unavailable_error(monkeypatch):
    from driveflow.assistant.core import llm

    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    client = llm.OllamaClient(acfg.load_config()) if hasattr(llm, "OllamaClient") else None
    if client is None:
        pytest.skip("no OllamaClient class in this version")
    with pytest.raises(llm.RuntimeUnavailable, match="does not answer at http://127.0.0.1:9"):
        client.health() if hasattr(client, "health") else client.chat([{"role": "user", "content": "hi"}], tools=[])
