# FE-1 — assistant event contract and avatar state

- **Command:** `PYTHONPATH=src python -m pytest -q tests/assistant/test_assistant_events.py` (9 tests, scripted conversations, no LLM).
- **Code:** `src/driveflow/assistant/events.py` (frozen dataclasses with `to_dict()`: `turn_started`, `llm_call`, `tool_call`, `proposal_ready`, `user_confirmed`, `job`, `results_ready`, `error`, `runtime_unavailable`, `turn_ended`); `core/loop.Agent(on_event=...)`; `src/driveflow/viz/adapters/assistant_adapter.py` (`avatar_state(events) -> (state, label, progress)`, `TOOL_LABELS`).
- **Unchanged with `on_event=None`:** the loop's behaviour and its existing trace (`trace.events`, `trace.draft_events`); the events are also kept in the new `trace.agent_events`. All earlier assistant tests pass without changes.
- **Where events come from:** the loop (turn, LLM call, tool call); tool outcomes (`propose_*` → `proposal_ready`; `run_*` / `get_job_status` → `job`; `summarize_results` / `check_requirements` → `results_ready`; a failed or refused call → `error`); the interface's recorded confirmation → `user_confirmed`; `RuntimeUnavailable` / `ModelNotAvailable` → `runtime_unavailable` (the exception still propagates, as before).
- **Priority** when several are active: unavailable > error > confirmed > extracting > using_tool > thinking > running > awaiting_confirmation > results_ready > idle. `error` lasts until the next `turn_started`; `confirmed` is a one-shot that ends with the turn (the 900 ms and the 400 ms minimum of `thinking` are the UI's job).
- **FE-D6 (tested):** only `user_confirmed` leads to `confirmed`; no sequence ending in `results_ready` shows `confirmed`.
- **Labels:** every registered tool (`set_mode`, `CONFIG_TOOLS`, `OBJECTIVE_TOOLS`, and `explain_spade` / `search_docs` for F1.7) has a readable English label (coverage test).
- **Not executed:** `extraction` events (phase 2, no `read_datasheet` yet; the adapter handles them).
