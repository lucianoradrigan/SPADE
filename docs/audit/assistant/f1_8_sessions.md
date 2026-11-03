# F1.8 — persistence of the assistant's chat sessions

- **Command:** `PYTHONPATH=src python -m pytest -q tests/assistant/test_assistant_sessions.py`.
- **Code:** three tables in `runs/store.py _SCHEMA` (`CREATE TABLE IF NOT EXISTS`: an existing workspace gains them when the store opens it, no migration — tested on a database without them), `src/driveflow/assistant/sessions.py` (`SessionStore`, `SessionRecorder`).
- **`assistant_sessions`:** id, time, `model_id` = `tag@digest`, `prompt_version` = `system_v1.md@<sha256[:12]>` (both refused without `@`), status open/closed, body (warnings such as a remote runtime, final draft, links to experiments and jobs, tokens).
- **`assistant_messages`:** every message of the conversation (system, user, assistant with tool calls, tool results), in order, with consecutive `seq`.
- **`assistant_launches`:** for each launch, the experiment and job, the **exact draft launched** (with the origin of every value) and the `seq` of the user message that confirmed it. `run_experiments` / `run_tuning` now keep that snapshot and seq with the job before clearing the confirmation; a launch without them is refused.
- **The prompt is not in `ExperimentSpec.params`:** the experiment's `config_hash` does not depend on it (tested). Experiments, runs and Parquet files are the existing ones; the runs carry `SIMULATED · AI-PLANNED` (since F1.5).
- **Not executed:** attachments (`workspace/assistant/<session_id>/`, phase 2); wiring into the CLI and the page (F1.9, F1.10).
