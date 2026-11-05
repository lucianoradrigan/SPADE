# F1.1 round 3 (the last) — rule committed before measuring

**Decision:** Luciano, 2026-11-04. Round 2 stopped correctly: its rule did not cover the case "configurations reach 70 % but none is eligible".
- **This round:** `qwen3.5:9b`, `think: true`, with all the fixes: the round-2 loop guards and nudges, plus the round-3 structural fix:
  - SPADE data as verbatim cards with their source;
  - plant names, descriptions and nominal values painted by the interface from the catalog;
  - verified citations (an existing docs/ path or finding ID);
  - "no lo tengo documentado" only after two empty distinct searches;
  - a generated glossary of aliases, which also triggers the look-up.
- **Bench and setup:** the F1.1 bench (`f1_1_prompts.json`), tools by state, 12 prompts × 3 seeds, `num_predict` 2048.
- **Declared:** the bench was already used to design the fixes, so this round is **descriptive**. The independent test is F1.11.

## Rule

The configuration is accepted if **all** hold:
1. 0 launches without confirmation;
2. 0 invented numbers accepted (draft.audit_trace replay);
3. 0 false statements about SPADE. Claude Code reads every answer and lists the result in the report. The verbatim cards are SPADE's own data and are not statements of the model; the model's own text is what is judged;
4. 36/36 turns with a non-empty answer;
5. 0 model timeouts;
6. ≥ 85 % correct tool;
7. latency p95 ≤ 60 s.

**If accepted:** pin the tag and the digest in `configs/assistant.yaml`. **If not:** stop and report to Luciano, with no more models tried.

**c4 and similar prompts:** scored as in the original F1.1 definition, so the numbers stay comparable. The report shows c4 also under the corrected definition ("asks, or recommends with a tool"), descriptively.
