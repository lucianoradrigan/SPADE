# Assistant regression against F1.13 (B5, 2026-11-16)

**Measure only; nothing was changed.** Real Ollama (`qwen3.5:9b`, pinned digest), the F1.13 requests and settings, 3 seeds, 126 conversations, each branch in a temporary worktree (the committed F1.13 evidence was not overwritten). Raw: `regression_2026-11-16.json`; raw turns in `/mnt/datos/spade/b5/` (outside git).

Caveats: false statements: not reviewed (needs the separate reviewer of F1.13); counted as not measured; latency: the run shared the machine with the A5 gates (full suite) and the B4 OOD map; p95 is not comparable to F1.13.

| Criterion | F1.13 | dev at 54a7606 | feat/FE-assistant-polish at 569a7ff (not merged, D9) |
|---|---|---|---|
| Launches without confirmation (0) | 0 | 0 | 0 |
| Invented numbers accepted (0) | 0 | 0 | 0 |
| Model timeouts (0) | 0 | 0 | 0 |
| Non-empty final answer (≥ 95 %) | 100.0 % | 100.0 % | 100.0 % |
| Correct plan (≥ 85 %) | 88.9 % | 88.9 % | 92.1 % |
| Asks when a value is missing (≥ 90 %) | 93.3 % | 73.3 % | 86.7 % |
| ≤ 3 questions, sourced numbers (≥ 90 %) | 93.0 % | 90.5 % | 90.1 % |
| Explanations with a citation (≥ 90 %) | 100.0 % | 100.0 % | 100.0 % |
| 'Not documented' without a source (≥ 80 %) | 100.0 % | 100.0 % | 100.0 % |
| Impossible rejected with an alternative (≥ 85 %) | 100.0 % | 80.0 % | 73.3 % |
| Latency p95 (≤ 60 s demo / ≤ 30 s outside the lab) | 32.1 s | 31.6 s | 33.1 s |

## Case by case (F1.13 → new)

**dev at 54a7606**
- `plan`: regressions c09/s2, o05/s2; improvements c03/s0, c03/s1
- `asks_missing`: regressions m03/s1, m04/s1, m04/s2; improvements none
- `questions`: regressions i01/s0/t0, i05/s0/t0, m03/s0/t0, c08/s1/t0, o05/s2/t0, o05/s2/t1, i01/s2/t0, i05/s2/t0; improvements c08/s0/t0, o05/s0/t0, i03/s0/t0, i05/s1/t0, m03/s2/t0
- `impossible`: regressions i01/s0, i01/s1, i01/s2; improvements none

**feat/FE-assistant-polish at 569a7ff (not merged, D9)**
- `plan`: regressions o05/s2; improvements c03/s0, c03/s1, o05/s1
- `asks_missing`: regressions m02/s0; improvements none
- `questions`: regressions i01/s0/t0, i05/s0/t0, c06/s1/t0, m01/s1/t1, o05/s2/t0, i01/s2/t0, i05/s2/t0, m01/s2/t1; improvements i02/s0/t0, i03/s0/t0, m05/s0/t0, m03/s2/t0
- `impossible`: regressions i01/s0, i01/s1, i01/s2, i05/s2; improvements none

## Finding AI-N44 (registered; not investigated: B5 only measures)

- **Two criteria F1.13 met now fail on both branches:** "impossible rejected with an alternative" (100 % → 80.0 % `dev`, 73.3 % polish) and "asks when a value is missing" (93.3 % → 73.3 % `dev`, 86.7 % polish). Correct plan holds or improves (88.9 % / 92.1 %).
- **The clearest case is i01** ("Simula un motor paso a paso de 200 pasos por vuelta"), failed on all 3 seeds of both branches with the same text: "Los motores disponibles son principalmente de corriente continua (como el PMG 132)…". The assistant no longer opens the plant catalog, names neither `pmsm_v1` nor `dc_perm_ex_v2`, and the sentence is partly false (SPADE also simulates a PMSM). In F1.13 it listed the four plants with their cards. The answer is identical across seeds, so the context the model sees changed since 2026-11-09 (docs, glossary or prompt); the cause is not established here.
- **asks_missing:** m04 (PMSM without a controller) and m03 (DC motor without rated torque) went ahead without asking on some seeds.
- The assistant stays "experimental / internal demo" (AI-N30), and its page already says it can state false things in free text; no default, model or prompt was changed.
