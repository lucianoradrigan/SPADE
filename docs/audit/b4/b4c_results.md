# B4c results: confirmatory evaluation of the B4 PC and RPi5 candidates

**Simulated** (`dc_perm_ex_v2`, `dcm_exact`, 10 kHz). Pre-registration: [`b4c_confirmatory_preregistration.md`](b4c_confirmatory_preregistration.md) (approved 2026-11-12, committed before any E4c data). Command: `python experiments/report_b4c.py`. Raw: [`b4c_results.json`](b4c_results.json).

**Nothing is promoted (M-D5).** A pass is a recommendation to Luciano.

| Tier | Verdict | Macro-F1 cand / prom | Min recall cand | Worst Δ recall (pt) | Δ abstention (pt) | Selective risk cand / prom |
|---|---|---|---|---|---|---|
| PC | **PASSES** | 0.994 / 0.661 | 0.970 | cage -0.7 | -18.4 | 0.004 / 0.275 |
| RPi5 | **PASSES** | 0.994 / 0.892 | 0.970 | cage -0.7 | -15.6 | 0.004 / 0.024 |

## PC: `configs/classifiers/pc_server_5class_w2048/2026-11-10_run01` vs `configs/classifiers/pc_server_5class_w2048/2026-11-04_run01` (1350 windows, 150 acquisitions)

| Criterion | Value | Threshold | Pass |
|---|---|---|---|
| 1. macro-F1 ≥ 0.80 | 0.994 | 0.80 | True |
| 1. recall ≥ 0.50 in every class (min) | 0.970 | 0.50 | True |
| 2. worst Δ recall ≥ −5 pt | cage -0.7 | −5.0 | True |
| 3. abstention fell (-18.4 pt): selective risk excess ≤ 1 pt | 0.004 vs 0.275 → -27.0 pt | +1.0 | True |

Descriptive: Δ macro-F1 +33.3 pt, 95 % CI [+27.3, +40.2]; Δ selective risk 95 % CI [-34.2, -20.5] pt.

| Speed (rad/s) | Macro-F1 cand / prom | Abstention cand / prom | Recall cand (per class) |
|---|---|---|---|
| 45 | 0.978 / 0.313 | 0.007 / 0.252 | ball 1.00, cage 0.89, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 50 | 0.996 / 0.432 | 0.011 / 0.285 | ball 1.00, cage 0.98, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 55 | 0.996 / 0.615 | 0.004 / 0.226 | ball 1.00, cage 0.98, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 65 | 1.000 / 0.829 | 0.000 / 0.089 | ball 1.00, cage 1.00, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 70 | 1.000 / 0.951 | 0.000 / 0.089 | ball 1.00, cage 1.00, inner_race 1.00, normal 1.00, outer_race 1.00 |

Confusion (rows true, columns predicted; candidate (promoted)):

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 270 (268) | 0 (2) | 0 (0) | 0 (0) | 0 (0) |
| cage | 0 (0) | 262 (264) | 0 (0) | 8 (6) | 0 (0) |
| inner_race | 0 (0) | 0 (0) | 270 (82) | 0 (0) | 0 (188) |
| normal | 0 (0) | 0 (90) | 0 (0) | 270 (180) | 0 (0) |
| outer_race | 0 (147) | 0 (0) | 0 (0) | 0 (0) | 270 (123) |

## RPi5: `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` vs `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01` (1350 windows, 150 acquisitions)

| Criterion | Value | Threshold | Pass |
|---|---|---|---|
| 1. macro-F1 ≥ 0.80 | 0.994 | 0.80 | True |
| 1. recall ≥ 0.50 in every class (min) | 0.970 | 0.50 | True |
| 2. worst Δ recall ≥ −5 pt | cage -0.7 | −5.0 | True |
| 3. abstention fell (-15.6 pt): selective risk excess ≤ 1 pt | 0.004 vs 0.024 → -2.0 pt | +1.0 | True |

Descriptive: Δ macro-F1 +10.2 pt, 95 % CI [+6.4, +14.7]; Δ selective risk 95 % CI [-3.4, -0.8] pt.

| Speed (rad/s) | Macro-F1 cand / prom | Abstention cand / prom | Recall cand (per class) |
|---|---|---|---|
| 45 | 0.974 / 0.581 | 0.011 / 0.289 | ball 1.00, cage 0.87, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 50 | 0.996 / 0.906 | 0.041 / 0.315 | ball 1.00, cage 0.98, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 55 | 1.000 / 0.904 | 0.004 / 0.085 | ball 1.00, cage 1.00, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 65 | 1.000 / 0.996 | 0.000 / 0.078 | ball 1.00, cage 1.00, inner_race 1.00, normal 1.00, outer_race 1.00 |
| 70 | 1.000 / 1.000 | 0.000 / 0.067 | ball 1.00, cage 1.00, inner_race 1.00, normal 1.00, outer_race 1.00 |

Confusion (rows true, columns predicted; candidate (promoted)):

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 270 (270) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| cage | 2 (1) | 262 (264) | 0 (0) | 6 (5) | 0 (0) |
| inner_race | 0 (0) | 0 (0) | 270 (270) | 0 (0) | 0 (0) |
| normal | 0 (0) | 0 (54) | 0 (0) | 270 (216) | 0 (0) |
| outer_race | 0 (65) | 0 (0) | 0 (17) | 0 (0) | 270 (188) |

CPU: {'generation': {'wall_s': 269.09, 'user_s': 1480.56, 'sys_s': 8.37, 'cpu_h': 0.41, 'workers': 6, 'source': 'GNU time'}, 'evaluation_wall_s': 12.3, 'evaluation_process_cpu_s': 23.2}

## Reading and recommendation (written by hand after the run; not part of the generated report)

- **Both tiers pass every pre-registered criterion on E4c** (150 new runs, never seen by any decision). Abstention falls (−18.4 pt PC, −15.6 pt RPi5), and so does the selective risk: 0.004 for both candidates, against 0.275 (PC) and 0.024 (RPi5) for the promoted runs. The candidates abstain less **and** err less among what they answer.
- **Where the promoted PC fails at low speed:** inner_race → outer_race (188 of 270), outer_race → ball (147), normal → cage (90). The candidates resolve all three.
- **Weakest point of the candidates:** cage at 45 rad/s (recall 0.89 PC, 0.87 RPi5; 8 cage windows → normal). 45 rad/s is below the 50 rad/s held out in B4 and between the 40 and 60 rad/s grid speeds.
- **Recommendation to Luciano (nothing promoted, M-D5):** promote both, `pc_server_5class_w2048/2026-11-10_run01` and `rpi5_resnet1d_se_5class_w2048/2026-11-10_run01`. The evidence: B4 E1 and E3 met, B4 E4 met except the two-sided abstention rule, and E4c met under the one-sided rule pre-registered before E4c existed.
  - Declared: the rule was changed after seeing B4's E4, which is why it was confirmed on new data.
  - Before promoting, note two things. RPi5's E1 CI of Δ macro-F1 reaches −2.2 pt (`b4_results.md` §5.4). Ball/cage stay "synthetic-only, uncalibrated amplitude".
- **Effect if promoted:** AI-N29 (PC confidently wrong at 50 rad/s) would close for PC and RPi5; AI-N34 and AI-N35 would close as "superseded by B4c".
