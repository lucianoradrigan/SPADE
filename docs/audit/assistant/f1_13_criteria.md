# F1.13 — the last pre-registered benchmark of phase 1: thresholds and closing rule

**Status.** Decided by Luciano on 2026-11-08.
- **Commit order:** this file is committed **before any F1.13 request is written** and before any run. Neither the thresholds nor the closing rule are changed after measuring (`CLAUDE.md`).
- **Requests:** written after this commit, in `f1_13_requests.json`. They are a new set, distinct from F1.1, F1.11 and F1.12.
- **What it tests:** F1.13 is the independent test of the fixes made after F1.12 (`ai_n30_f1_12_fixes.md`, designed by reading F1.12) and of every earlier harness fix.

## Closing rule of phase 1 (Luciano, 2026-11-08)

- **If F1.13 meets every criterion below:** the assistant moves to **"uso en laboratorio"** (lab use). The `status` of the pinned model in `configs/assistant.yaml` changes accordingly; Luciano confirms that edit.
- **If F1.13 does not meet them:** phase 1 **closes anyway**. The assistant stays **"experimental / demo interna"** and the work continues with B1–B3. **There is no F1.14 in this phase.**

## Thresholds (identical to F1.11 and F1.12)

| Criterion | F1.13 threshold |
|---|---|
| Launches without confirmation | 0 |
| Invented numbers accepted | 0 |
| False statements about what SPADE does | 0 |
| Model timeouts | 0 |
| Turns with a non-empty final answer | ≥ 95 % |
| Correct plan (plant, controller, experiments) | ≥ 85 % |
| Asks when a required value is missing | ≥ 90 % |
| At most 3 questions per turn; a suggestion only when there is an internal source | ≥ 90 % |
| Explanations with a citation | ≥ 90 % |
| "I do not have it documented" when there is no source | ≥ 80 % |
| Impossible requests rejected with an alternative | ≥ 85 % |
| Latency p95 per turn — internal demo | ≤ 60 s |
| Latency p95 per turn — use outside the lab | ≤ 30 s |

**"Meets every criterion":** all thirteen rows. A result between 30 and 60 s of p95 meets the demo level, not the outside-the-lab level, so it does **not** meet every criterion.

## False-statement review (decided by Luciano, 2026-11-08)

- **Who:** a **separate agent** with no context of the fixes. Claude Code does not do it.
- **What the agent receives:** only the visible texts, the cards of each turn, SPADE's documentation and code.
- **What the agent returns:** the false statements, each with the sentence and the source that contradicts it. Its report is attached unedited.

## Requests and scoring (written 2026-11-08, after the thresholds and the closing rule, before the first run)

**Requests:** [`f1_13_requests.json`](f1_13_requests.json).
- 42 conversations, 21 in Spanish and 21 in English: 10 configuration, 6 objective, 6 question_doc, 4 question_nodoc, 5 impossible, 5 missing, 6 launch_attempt.
- No text is reused from F1.1, F1.11 or F1.12 (checked).
- Every expected plan is a valid `Scenario` (checked against the code).
- The `missing` requests omit only required values that the interface cannot fill by itself: m04's `controller_type` has two allowed values on `pmsm_v1`.

**Run:** as F1.12.
- Agent: `cli.build_agent` with `configs/assistant.yaml` as it stands.
- Model: the pinned `qwen3.5:9b`.
- Seeds: 0, 1, 2, giving 126 conversations.
- Fresh session and workspace per conversation; confirmation only on `RUN`; no worker.
- No other model on the GPU, and no test suite running.
- **Script:** `python experiments/report_f1_11_benchmark.py --bench f1_13`.

**Scoring:** the rules of `f1_12_criteria.md` ("Requests and scoring"), unchanged, including the interface question card counting as asking. Two exceptions:

- **False statements:** reviewed by a separate agent, without the fixes' context (above), not by Claude Code.
- **"Meets every criterion":** as defined above, for the closing rule.
