# F1.1 round 2 — qwen3.5:9b with think and loop guards (descriptive)

Rule committed before measuring: [`f1_1_model_choice_round2_rule.md`](f1_1_model_choice_round2_rule.md).
- **Declared:** the F1.1 bench had already been used to design the guards and nudges. These numbers are descriptive, not an independent test.
- **Setup:** tools by state, 12 prompts × 3 seeds, Ollama 0.40.13, RTX 5060 Ti 16 GB, `num_predict` 2048 where guards are on.

| Configuration | Correct tool (conf. / obj. / q.) | Turns with an answer | Timeouts | Latency median / p95 | Launches / invented numbers | False statements | Eligible |
|---|---|---|---|---|---|---|---|
| round 1: qwen3.5:9b think=false, no guards | 33.3 % (33 / 33 / 33) | 36/36 | 0 | 3.6 / 12.5 s | 0 / 0 | 1 kinds | no: false statements (1 kinds) |
| (a) qwen3.5:9b think=true, no guards | 91.7 % (100 / 67 / 100) | 33/36 | 3 | 18.0 / 189.9 s | 0 / 0 | 2 kinds | no: false statements (2 kinds); answered 33/36 < 34; p95 189.9 s > 30 |
| (b) qwen3.5:9b think=false + guards | 66.7 % (83 / 67 / 33) | 36/36 | 0 | 9.4 / 31.1 s | 0 / 0 | 3 kinds | no: false statements (3 kinds); p95 31.1 s > 30 |
| (b) qwen3.5:9b think=true + guards | 91.7 % (100 / 67 / 100) | 36/36 | 0 | 21.2 / 63.5 s | 0 / 0 | 2 kinds | no: false statements (2 kinds); p95 63.5 s > 30 |
| (c) qwen3:8b think=false + guards (reference) | 58.3 % (100 / 33 / 0) | 36/36 | 0 | 7.6 / 12.8 s | 0 / 0 | 2 kinds | no: false statements (2 kinds) |

## Verdict

**No configuration is eligible**, so no tag is pinned in `configs/assistant.yaml`. Stopped and reported to Luciano, with no more models tried.
- Every configuration has false statements about SPADE in its answers, listed below and read by Claude Code in every answer of the traces.
- With `think: true` the tool use is good (91.7 % with and without guards), but the latency p95 is 60–64 s even without timeouts. The rule asks for ≤ 30 s.
- With `think: false` and guards, p95 is 31.1 s and correct tool 66.7 %.
- The rule says what to do when no configuration reaches 70 %. Here two reach it ((a) and (b) think=true) but are not eligible, and the rule does not cover that case. Claude Code reads it as "no choice possible" and stops.

## Deviations (declared)

- **(b) was run for both think variants.** The rule picks the think variant of (b) with the eligibility rule, and neither round 1 nor (a) was eligible (false statements in both; (a) also 3 timeouts and p95 190 s). Running both avoided choosing by interpretation.
- **A tool bug was fixed during the round.** The first run of (b) think=true crashed in `draft.audit_trace` (the no-invention replay): the model sent `update_draft` with `ops = []` and the replay raised instead of refusing. Fixed with a test (`5f67149`), and (b) think=true was rerun from the start. Only the completed run is reported.
- **(c) used `think: false`**, because think=true doubles the latency and could not meet p95 ≤ 30 s. It took 4.5 min.

## False statements found in the traces

- **round 1: qwen3.5:9b think=false, no guards:** q2 (3 seeds): S-21 described as 'un conjunto de decisiones y hallazgos'; it is a simulator finding (converter artifact)
- **(a) qwen3.5:9b think=true, no guards:** q3 (3 seeds): 'the term "abstention threshold" is not documented'; it is (AI-N12, VALIDATION_SUMMARY); q2 (3 seeds): S-21 placed in AI_TECHNICAL_AUDIT §13.1 as a training-data defect; that is AI-N27; S-21 is in SIMULATION_STABILITY_REPORT
- **(b) qwen3.5:9b think=false + guards:** c4: dc_perm_ex_v1 described as 'excitación independiente' (it is permanently excited) and dc_pmg132_v1 as 'motor DC con generador PMG'; c5: 150 rad/s called the 'velocidad nominal' (150 is the default; rated is 300 rad/s); q2 (3 seeds): S-21 as 'un conjunto de decisiones' (as round 1)
- **(b) qwen3.5:9b think=true + guards:** q3 (3 seeds): 'abstention threshold ... is not documented' (false); q2 (3 seeds): S-21 conflated with AI-N27 (§13.1, training-data defect)
- **(c) qwen3:8b think=false + guards (reference):** c2: 'dc_perm_ex_v1 no acepta el controlador PI' (false); q2: 'No lo tengo documentado' about S-21 (it is documented)

## Reading of the conversations (descriptive)

- **The guards work as designed:** 0 empty answers, 0 timeouts, and repeats and caps refused.
- **Nudge 1 (record in the draft)** raised configuration correctness for think=false from 33 % to 83 %.
- **The two `nudge_docs` triggers miss cases:**
  - A question that names neither SPADE nor an ID (q3, "abstention threshold of the classifiers") does not trigger the look-up.
  - The citation check accepts "según los documentos" as a citation (q2, think=false).
  - Both are detector limits, found here.
- **The false statements have two sources:**
  - a search that misses a documented term, so the model answers "not documented" (q3);
  - the model mixing nearby findings (S-21 vs AI-N27) or paraphrasing plant descriptions wrongly.
- **qwen3:8b:** fast (p95 12.8 s) and good on configuration (100 %), weak on objective (33 %) and questions (0 %), with 2 kinds of false statements.

## Prompt c4 ("Simula un motor DC.") — expected value ill-defined

- **What was expected:** mode configuration plus one of recommend_plant / describe_plant / suggest_value / update_draft.
- **Why that is wrong:** SPADE has two DC plants (`dc_perm_ex_v1`, `dc_pmg132_v1`), so asking the user which one, without a tool call, is also a correct answer. Several runs of every model did exactly that and were counted as wrong.
- **Recorded, numbers unchanged:** the expected value was ill-defined. The numbers of the original run and of this round are **not** changed. The F1.11 requests will define such cases as "asks, or recommends with a tool".
