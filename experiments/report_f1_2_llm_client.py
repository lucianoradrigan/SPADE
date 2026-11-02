"""Phase 1, F1.2: the local model client against the real Ollama of this machine (no model yet).

Checks, live: runtime version and installed models; that an unpinned configuration refuses to chat;
that a non-loopback base_url is refused without allow_remote_runtime (C4); the configuration that
is committed. The client's behaviour with a model (chat payload, tool-call parsing, digest check,
errors, cassettes) is covered offline by tests/assistant/test_assistant_llm.py with a fake Ollama.

    python experiments/report_f1_2_llm_client.py
Writes docs/audit/assistant/f1_2_llm_client.{md,json}.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from driveflow.assistant import config as acfg
from driveflow.assistant.core import llm

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f1_2_llm_client"


def main() -> None:
    cfg = acfg.load_config()
    client = llm.OllamaClient(cfg)
    try:
        version, models = client.version(), client.installed_models()
    except llm.RuntimeUnavailable as exc:
        version, models = f"not reachable: {exc}", []
    try:
        client.chat([{"role": "user", "content": "hi"}], [])
        unpinned = "chatted (unexpected)"
    except llm.ModelNotAvailable as exc:
        unpinned = f"refused: {exc}"
    try:
        acfg.load_config(overrides={"base_url": "http://192.168.1.10:11434"})
        remote = "accepted (unexpected)"
    except acfg.RemoteRuntimeRefused as exc:
        remote = f"refused: {exc}"
    shown = {k: v for k, v in cfg.items() if k != "warnings"}
    result = {"command": "python experiments/report_f1_2_llm_client.py", "python": platform.python_version(), "runtime_version": version,
              "installed_models": models, "config": shown, "unpinned_chat": unpinned, "remote_base_url": remote}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# F1.2 — local model client (Ollama)",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f1_2_llm_client.json`](f1_2_llm_client.json). Tests: `tests/assistant/test_assistant_llm.py` (fake Ollama on loopback; the real-runtime test runs with `pytest --run-llm`).",
        "- **Code:** `src/driveflow/assistant/core/llm.py` (`LLMClient`, `OllamaClient`, `ReplayClient`, `RecordingClient`, `parse_reply`), `src/driveflow/assistant/config.py`, `configs/assistant.yaml`, `scripts/assistant_record_cassette.py`.",
        "- **No Python dependency added** (B2): HTTP with `urllib.request` + `json`.",
        "",
        "## Live checks on this machine",
        "",
        f"- Ollama runtime: **{version}**; installed models: {len(models)} ({', '.join(m.get('name', '?') for m in models) or 'none: candidates are pulled in F1.1'}).",
        f"- Chat with no pinned model: {unpinned}",
        f"- `base_url` outside this machine without `allow_remote_runtime`: {remote}",
        "",
        "## Committed configuration (`configs/assistant.yaml`)",
        "",
        "```json",
        json.dumps(shown, indent=2),
        "```",
        "",
        "Every call sends `temperature: 0` and the seed, `stream: false`, `num_ctx` = `context_tokens`, and `think: false` (models that reason before answering are asked not to; null sends nothing). "
        "The pinned digest is compared with the installed one before the first chat; a mismatch is a warning carried by the client (and, from F1.8, recorded with the session).",
        "",
        "## Not executed",
        "",
        "- A chat with a real model: no ejecutado (no model is downloaded or pinned; that is task F1.1).",
        "- Recording a cassette against the real runtime: no ejecutado for the same reason; `scripts/assistant_record_cassette.py` is in place.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
