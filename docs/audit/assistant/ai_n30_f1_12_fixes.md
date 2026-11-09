# AI-N30: fixes after F1.12

- **Decision:** Luciano, 2026-11-08.
- **Declared:** these fixes were designed by reading the F1.12 conversations (`f1_12_benchmark.md`, `f1_12_review.md`). F1.12 can no longer test them. **F1.13**, the last bench of phase 1 (new requests, the same thresholds), is the independent test.
- **Code:** `assistant/guardrails.py` (F1.12 section), `core/loop.py` (`_harness_v2`), `cards.py`.
- **Tests:** `tests/assistant/test_assistant_f1_12_fixes.py` (6 tests). Assistant tests: 305 passed, 1 skipped.

| Fix | F1.12 evidence | What it does |
|---|---|---|
| Proposal cost only in its card | 39 of the 44 "unsourced number" failures were restated costs | In configuration / objective mode, every sentence or line of the model's text that mentions CPU cost or time with a number is removed. The proposal card keeps the estimate. |
| A required field with a single allowed value | m04: the model filled DPC by itself, and the question card never appeared | `fill_single_choice`: `controller_type` missing with one allowed value (DPC on `vsc_dpc_v1`) is set by the interface with origin `default`. A "SPADE filled these values" card says so, with the reason. |
| "No documentation found" drops the earlier search cards | n01: an earlier broad search card was shown next to it | When that card is present, the turn's other `search_docs` / `explain_spade` result cards are removed. |
| Job state only in its card | l05: "is running on a SPADE worker" for a queued job (false) | `get_job_status`, `run_experiments` and `run_tuning` results become cards: state, progress, run, and that a worker must process queued jobs. Every sentence of the model about a job's state is removed, and "The job's state is in the card." is added. |
| Leaked reasoning and orphan lead-ins | l01: the model's reasoning and rules shown as text; "I cannot confirm that from SPADE's documentation" in front of proposals | Paragraphs that start like reasoning ("The user wants…", "I should…", "Let me clarify…", "El usuario quiere…") and the ones that follow them, up to a question, are removed. The guard's lead-in is added only when a limits card backs it, after the kept text. |

**Not changed:** the model, the prompt, the thresholds, the safety rules.
