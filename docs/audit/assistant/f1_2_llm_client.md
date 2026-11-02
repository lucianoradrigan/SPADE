# F1.2 — local model client (Ollama)

- **Command:** `python experiments/report_f1_2_llm_client.py`. Raw values: [`f1_2_llm_client.json`](f1_2_llm_client.json). Tests: `tests/assistant/test_assistant_llm.py` (fake Ollama on loopback; the real-runtime test runs with `pytest --run-llm`).
- **Code:** `src/driveflow/assistant/core/llm.py` (`LLMClient`, `OllamaClient`, `ReplayClient`, `RecordingClient`, `parse_reply`), `src/driveflow/assistant/config.py`, `configs/assistant.yaml`, `scripts/assistant_record_cassette.py`.
- **No Python dependency added** (B2): HTTP with `urllib.request` + `json`.

## Live checks on this machine

- Ollama runtime: **0.40.13**; installed models: 0 (none: candidates are pulled in F1.1).
- Chat with no pinned model: refused: no model is pinned in configs/assistant.yaml (model.tag); it is chosen in task F1.1
- `base_url` outside this machine without `allow_remote_runtime`: refused: base_url 'http://192.168.1.10:11434' is not on this machine: the conversation and every tool result would leave it. Use 127.0.0.1 / ::1 / localhost, or set allow_remote_runtime: true in configs/assistant.yaml on purpose.

## Committed configuration (`configs/assistant.yaml`)

```json
{
  "runtime": "ollama",
  "base_url": "http://127.0.0.1:11434",
  "allow_remote_runtime": false,
  "timeout_s": 180,
  "model": {
    "tag": null,
    "digest": null
  },
  "temperature": 0,
  "seed": 0,
  "context_tokens": 16384,
  "think": false,
  "limits": {
    "max_tool_calls_per_turn": 12,
    "max_experiments_per_launch": 50,
    "max_total_sim_seconds": 600
  }
}
```

Every call sends `temperature: 0` and the seed, `stream: false`, `num_ctx` = `context_tokens`, and `think: false` (models that reason before answering are asked not to; null sends nothing). The pinned digest is compared with the installed one before the first chat; a mismatch is a warning carried by the client (and, from F1.8, recorded with the session).

## Not executed

- A chat with a real model: no ejecutado (no model is downloaded or pinned; that is task F1.1).
- Recording a cassette against the real runtime: no ejecutado for the same reason; `scripts/assistant_record_cassette.py` is in place.
