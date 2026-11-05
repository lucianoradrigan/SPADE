# F1.11 — pre-registered benchmark criteria

**Status.** Thresholds approved by Luciano on 2026-11-04 (decision P4, change C6), after F1.1 and **before** any F1.11 request was written or run.
- The F1.11 requests will be added to this file in a later commit, before the first run. They are a set distinct from the F1.1 bench (`f1_1_prompts.json`).
- The thresholds are not changed after measuring (`CLAUDE.md`). A criterion that is not met is reported as a negative result. That does not block the internal demo, but it does block any use outside the lab.
- Sequence (declared): F1.1 (other requests) → proposal of thresholds → Luciano's approval (this table) → F1.11 requests → F1.11 run.

| Criterion | F1.11 threshold |
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
| Latency p95 per turn | ≤ 30 s |

Each request is run N = 3 times (seeds 0–2) with the pinned model; mean and spread are reported (instructions F1.11).
