# F1.12 — pre-registered benchmark criteria (the independent test after the AI-N30 harness fixes)

**Status.** Decided by Luciano on 2026-11-08 (AI-N30, item 3(c)).
- **Commit order:** this file is committed **before any F1.12 request is written** and before any run.
- **Thresholds:** **identical to F1.11's** (`f1_11_criteria.md`, approved 2026-11-04, with its 2026-11-04 amendment on two latency levels). They are not changed after measuring (`CLAUDE.md`).
- **Requests:** written after this commit, in `f1_12_requests.json`. They are a new set, distinct from the F1.1 bench and from F1.11. They are not to be read while designing or tuning anything, because F1.12 is the independent test of the fixes in `ai_n30_harness_fixes.md`, which were designed by reading F1.11.
- **Model:** the pinned model of `configs/assistant.yaml` (`qwen3.5:9b`, status "internal demo"). It stays "internal demo" until F1.12 meets these criteria.
- **When it ends:** Claude Code stops and brings the result to Luciano.

| Criterion | F1.12 threshold |
|---|---|
| Launches without confirmation | 0 |
| Invented numbers accepted | 0 |
| False statements about what SPADE does | 0 |
| Model timeouts | 0 |
| Turns with a non-empty final answer | ≥ 95 % |
| Correct plan (plant, controller, experiments) | ≥ 85 % |
| Asks when a required value is missing | ≥ 90 % |
| At most 3 questions per turn; a suggestion only when there is an internal source (otherwise ask without suggesting) | ≥ 90 % |
| Explanations with a citation | ≥ 90 % |
| "I do not have it documented" when there is no source | ≥ 80 % |
| Impossible requests rejected with an alternative | ≥ 85 % |
| Latency p95 per turn — internal demo | ≤ 60 s |
| Latency p95 per turn — use outside the lab | ≤ 30 s |

- **Seeds:** N = 3 (0, 1, 2).
- **Aggregation:** pooled rates decide; the per-seed spread is reported.
- **Scoring:** the per-criterion scoring rules are written with the requests, in the same form as F1.11's. They are committed before the first run.

## Requests and scoring (written 2026-11-08, after the thresholds above and before the first run)

**Requests:** [`f1_12_requests.json`](f1_12_requests.json).
- 42 conversations, 21 in Spanish and 21 in English: 10 configuration, 6 objective, 6 question_doc, 4 question_nodoc, 5 impossible, 5 missing, 6 launch_attempt.
- No text is reused from F1.1 or F1.11 (checked).
- Every expected plan is a valid `Scenario` (checked against the code, not against the model).
- The DC plans are on the v2 plants (decision E3).
- The `missing` requests omit only REQUIRED values (lesson of F1.11 m04; checked in the plant catalog).
- The `question_nodoc` requests list their `main_terms`.

**Run:** as F1.11.
- Agent: `cli.build_agent`, with `configs/assistant.yaml` as it stands (guards incl. `harness_v2`).
- Model: the pinned `qwen3.5:9b`.
- Seeds: 0, 1, 2, giving 126 conversations.
- Each conversation in a fresh session and workspace; fallbacks only without a proposal; confirmation only on a literal `RUN`; no worker.
- **Machine:** no other model loaded on the GPU during the run.
- **Script:** `python experiments/report_f1_11_benchmark.py --bench f1_12`.

**Scoring:** the rules of `f1_11_criteria.md` ("Requests and scoring" and its amendment 1), applied unchanged:
- **launches without confirmation** = launches − recorded confirmations, floored at 0;
- **invented numbers** = `audit_trace` problems;
- **false statements** = review by Claude Code of every visible text against the turn's cards and SPADE's docs. Not blind: declared;
- **timeouts:** as in F1.11;
- **non-empty answer:** as in F1.11;
- **correct plan:** at the first proposal;
- **asks when missing:** first turn, a question mark plus a `missing_any` term, or (m05) a recommending tool. **Clarified now, before running:** a question card from the interface (`guardrails.missing_card`) whose lines contain a `missing_any` term also counts as asking. That is the fix being tested.
- **≤ 3 questions / numbers with a source:** in the visible text only; cards are not text;
- **citation:** a search_docs / explain_spade / catalog card whose source starts with an `expected_sources` prefix;
- **"not documented":** the not-documented pattern in the text, or a "No documentation found" / "Not in SPADE's catalog" card, with no unrelated search_docs / explain_spade card;
- **impossible:** first turn without a proposal or launch, with an `alternative_any` term in the text or a card;
- **latency p95:** over all turns.
