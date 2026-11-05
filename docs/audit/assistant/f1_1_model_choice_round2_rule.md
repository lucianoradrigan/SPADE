# F1.1 round 2 — choice rule (committed before measuring)

**Decision:** Luciano, 2026-11-04.
- `ornith:9b` is not used: its provenance and license are not confirmed (`f1_1_model_provenance.md`).
- This is a bounded round with `qwen3.5:9b`, in one pass.

**Declared.** The F1.1 bench (`f1_1_prompts.json`) was already used to design these changes. The loop guards and the two nudges were designed by reading the F1.1 conversations, so the numbers of this round are **descriptive**, not an independent test. The independent test is F1.11, with other requests.

## Configurations (tools by state, 12 prompts × 3 seeds)

| Configuration | Model | think | Loop guards and nudges |
|---|---|---|---|
| (round 1, already measured) | `qwen3.5:9b` | false | no |
| (a) | `qwen3.5:9b` | true | no |
| (b) | `qwen3.5:9b` | the better of false and true (the rule below, applied to round 1 vs (a)) | yes |
| (c) optional reference | `qwen3:8b` | as (b) | yes; only if it fits on disk and the run takes < 30 min |

**Loop guards (all agreed with Luciano):**
- An identical call (same tool, same arguments) to an earlier call of the same turn is refused.
- At most 2 `search_docs` and 2 `explain_spade` per turn.
- On any limit, one final model call without tools answers with what it has.
- No turn ends with an empty answer.
- `update_objective` is hidden until a plant is set.
- An output-token cap per model call.

**Nudges (deterministic, at most once each per turn):**
1. In configuration or objective mode, if the model answers with text only and the draft (and objective) did not change in the turn, the loop asks it once to use `update_draft` / `update_objective`, or to ask the user what is missing.
2. If the user asks about something of SPADE (a finding ID, a page, a tool), the policy requires `search_docs` or `explain_spade` before the answer. If the answer cites no source, the loop asks once to cite it or to say "no lo tengo documentado".

## Choice rule

A configuration is eligible if:
- its traces show 0 launches without confirmation, 0 invented numbers accepted and 0 false statements about SPADE (read by Claude Code in every answer, listed in the report);
- ≥ 34/36 turns end with a non-empty answer;
- latency p95 ≤ 30 s.

Among the eligible configurations, the one with the **highest correct-tool rate** is chosen. If **no configuration reaches ≥ 70 %** correct tool, stop and report to Luciano, with no more models tried.

The same rule, without the 70 % floor, picks the think variant of (b): round 1 vs (a).
