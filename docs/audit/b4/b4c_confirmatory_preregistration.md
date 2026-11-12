# B4c: confirmatory evaluation of the B4 PC and RPi5 candidates (pre-registration)

**Status: APPROVED (2026-11-12, Luciano, decision M-D2).** Written and committed **before any E4c data exists**. The content below is the one Luciano approved in `INSTRUCCIONES_CLAUDE_CODE_2026-11-12.md` §M3; only the implementation details marked *(implementation)* were added by Claude Code, before any data. Everything is **simulated**.

## 1. Why, declared

In B4 ([`b4_results.md`](b4_results.md)) the PC and RPi5 candidates met every check except the two-sided abstention rule on E4 (|Δ abstention| ≤ 5 pt). They abstain less and answer more correctly: recall 0.97–1.00 at 50/70 rad/s, against 0.44–0.58 for the promoted PC run on inner_race, normal and outer_race.

**The rule is changed after seeing that result.** E4 therefore cannot confirm it. The new rule is evaluated only on **new data (E4c)** that no decision has seen.

## 2. Frozen models

| Tier | Candidate (deployed artifact) | Against (promoted today) |
|---|---|---|
| PC | `pc_server_5class_w2048/2026-11-10_run01` (Keras) | `pc_server_5class_w2048/2026-11-04_run01` |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` (float16 TFLite) | `rpi5_resnet1d_se_5class_w2048/2026-11-04_run01` |

Nothing is retrained or recalibrated. Each model uses its own preprocessing and its own abstention threshold (`metrics.json` → `calibration.abstain_below`), as in B4.

## 3. The E4c set (new)

- **Plant:** `dc_perm_ex_v2`, converter `dcm_exact`, 10 kHz. 2.5 s per run; steady-state windows only (t ≥ 0.5 s), as E4.
- **Speeds:** 45, 55 and 65 rad/s, plus 50 and 70 rad/s **with new seeds**.
- **Per speed, as in E4:**
  - every class of the tier (normal, outer_race, inner_race, ball, cage);
  - severities ×1.0 and ×1.5;
  - 3 runs per (severity, class);
  - the stock load (E4 varied no load).

  That gives 30 runs per speed, **150 runs** in total.
- **Noise and plant drift:** drawn as in E4 and the grid: one third noise-free; ψ_e × U(0.9, 1.1); J × U(0.8, 1.5).
- *(implementation)* **Seeds:** RNG seed 20261212; run seeds 100001–100150.

  This block is disjoint from every earlier block:
  - grids 1001–1675 and 80001–80675;
  - off-grid 85001–85060;
  - E4 90001–90060;
  - E5 95001–95030;
  - AI-N27 held-out from 60001;
  - evaluation seeds 200–203.

  `tests/test_b4c_plan.py` verifies this, as Q3 did.
- **Isolation:** no E4c condition coincides with the 60 off-grid healthy runs (`configs/b4_plan.yaml`), whose speeds are 30, 90, 100, 125, 140, 150, 175, 200 and 210 rad/s. The same test checks it.
- *(implementation)* **Files:**
  - plan `configs/b4c_plan.yaml`;
  - generator `python experiments/generate_b4_sets.py e4c --out data/b4c_e4c_5class.parquet`;
  - evaluation `python experiments/report_b4c.py`;
  - report `docs/audit/b4/b4c_results.{md,json}`.

## 4. Criteria on E4c, per tier (a tier passes only if all hold)

Δ = candidate − promoted, in percentage points, on the same windows; known classes only.

1. **Low speed (D-B4-2):** macro-F1 ≥ 0.80, and recall ≥ 0.50 in every class.
2. **No class drops more than 5 points** of recall against the promoted run.
3. **Abstention, one-sided:**
   - if the candidate's abstention rises, it rises at most **+5 points** against the promoted run;
   - if it falls, the candidate's **selective risk** (error rate among the windows it does not abstain on) may not exceed the promoted run's by more than **1 point**.
4. **E1 and E3 are not re-evaluated:** they already passed in B4.

*(implementation)* Recall and macro-F1 use the argmax over every window, as `test_macro_f1` does in B4. Selective risk is 1 − accuracy over the windows whose top probability is ≥ the model's `abstain_below`. If no window is answered, the selective risk is undefined and the check fails.

**Descriptive only:**
- the bootstrap 95 % CI of Δ macro-F1 and of Δ selective risk (1000 resamples of acquisitions, seed 0);
- results per speed;
- confusion matrices;
- CPU time.

## 5. Result

- **Pass:** recommend promotion to Luciano, **without promoting** (decision M-D5).
- **Fail:** a new finding. Nothing is retried, and no threshold is changed.
- **E4c is used for nothing else.** It does not inform any design, threshold or later model.
