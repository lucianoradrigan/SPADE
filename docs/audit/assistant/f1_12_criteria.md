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
