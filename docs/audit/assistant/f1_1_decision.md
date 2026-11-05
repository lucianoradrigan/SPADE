# F1.1 — closing decision (2026-11-04)

**This is a decision of Luciano, not a result of the bench.** The F1.1 round-3 rule was **not met**, and that verdict stands unchanged (`f1_1_model_choice_round3.md`).

## Decision

- **Pinned in `configs/assistant.yaml`:** `qwen3.5:9b` (Qwen Team, Apache-2.0, Q4_K_M, 9.0 B), digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, with `think: true`.
- **Use:** only the milestone "Demo interna del asistente".
- **Status field:** `status: "internal-demo; F1.1 round-3 rule not met"`.
- **No more selection rounds.**

## Known limits (measured on the F1.1 bench, round 3; `f1_1_model_choice_round3.md`)

1. **3 kinds of false statements** in the model's own text, each repeated in the 3 seeds:
   - q3: says the abstention threshold mitigates errors on unseen fault types "como bolas"; VALIDATION_SUMMARY says the opposite (AI-N12, not solved for ball);
   - c5: attributes 300 rad/s to the user, although it is the nominal value from `describe_plant`;
   - o3: calls the tunable `setpoint_weight` "`pi_setpoint_weight`, el peso proporcional".
2. **Latency p95 90.5 s per turn** (median 23.8 s). c5 takes about 92 s.
3. The bench was used to design the fixes, so its numbers are descriptive.

## What follows (Luciano)

- **The next fixes are F1.9/F1.10 work, with ReplayClient tests:**
  - answers about SPADE as verbatim cards plus at most one checked context sentence;
  - draft values never repeated by the model (a draft panel with origin chips; draft numbers removed from the final sentence);
  - a tunables card in objective mode;
  - think only on tool-choosing calls;
  - no extra "brief" call;
  - at most 2 update_draft retries before asking the user.
- **F1.11 is the independent test:** new requests, with the thresholds committed before writing them (`f1_11_criteria.md`; latency ≤ 60 s for the internal demo, ≤ 30 s outside the lab).
  - If not met, it is a negative result: the assistant stays internal-demo only, and the options (e.g. a larger model) go back to Luciano.
  - The thresholds are not changed after measuring.
