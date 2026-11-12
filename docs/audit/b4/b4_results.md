# B4 results (2026-11-10): pre-registered retraining on the v2 plants

**Simulated.** Every dataset comes from `dc_perm_ex_v2` (converter `dcm_exact`, 10 kHz). Pre-registration: [`b4_preregistration.md`](b4_preregistration.md), approved 2026-11-09 (decision N-D3) in commit `306d3ce`, before any data. Plan: `configs/b4_plan.yaml`. Raw numbers: [`b4_results.json`](b4_results.json).

**Verdict: no tier meets its criterion.** Nothing is promoted (decision N-D4); the promoted runs stay as they are.

| Tier | Candidate (deployed artifact) | Verdict | Failing checks | Recommendation |
|---|---|---|---|---|
| PC | `pc_server_5class_w2048/2026-11-10_run01` (Keras) | **Does not meet** (AI-N34) | only E4 \|Δ abstention\| = 16.3 pt (> 5): it abstains **less** | do not promote under the rule; see decision 1 |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` (float16 TFLite) | **Does not meet** (AI-N35) | only E4 \|Δ abstention\| = 20.6 pt: it abstains **less** | same as PC |
| ESP32 | `esp32_dscnn/2026-11-10_run01`, int8 variant V1 (`model.tflite`) | **Does not meet** (AI-N36) | D1 gate, inner_race recall, abstention, int8 vs Keras | do not promote |

## Commands

```bash
python experiments/b4_preconditions.py                       # Q1, Q3 -> b4_preconditions.{md,json}
bash docs/audit/b4/b4_pipeline.sh                             # data, training, exports (resumable), times -> b4_times.tsv
python experiments/report_b4.py --pc configs/classifiers/pc_server_5class_w2048/2026-11-10_run01 \
  --rpi5 configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-10_run01 \
  --esp32 /mnt/datos/spade/b4_work/variants/V1 --esp32-keras configs/classifiers/esp32_dscnn/2026-11-10_run01 \
  --teacher3 configs/classifiers/pc_server/2026-11-10_run01
```

The variant V1 directory was then copied into the 4a run: `esp32_dscnn/2026-11-10_run01/model.tflite` is V1, byte for byte; its weights are the run's own.

## Preconditions (§2)

- **Q1:** pass. The first, middle and last acquisitions of the stored 5-class dcm grid, regenerated with `dc_perm_ex_v2`: NRMS ≤ 7.4e-5 (limit 1e-4); the noise-free run has min `current_r` = 33.8 A ([`b4_preconditions.md`](b4_preconditions.md)).
- **Q2:** pass for every candidate (no acquisition in two splits; `Q2_split_by_acquisition` in the JSON).
- **Q3:** pass. The E4 (90001–) and E5 (95001–) seeds overlap no grid, off-grid or evaluation seed.

## Data and candidates

| Set | Runs | Seeds | File (`/mnt/datos/spade/data/`) |
|---|---|---|---|
| G3-v2 | 405 (9 speeds × 5 severities × 3 seeds × 3 classes) | 80001– | `b4_grid_3class.parquet` |
| Off-grid healthy (§3.5) | 60 (20 conditions × 3) | 85001– | `b4_offgrid_healthy.parquet`; G3-v2 + these = `b4_train_3class.parquet` |
| G5-v2 | 675 | 80001– | `b4_grid_5class.parquet` |
| E3 | 13 conditions, seeds 200–203, 1.5 s | as `eval_cache_5class_dcm` | `b4_eval_cache_5class.parquet` |
| E4 | 36 (3-class) and 60 (5-class): 50 and 70 rad/s × 1.0 and 1.5 × 3 seeds | 90001– | `b4_e4_3class.parquet`, `b4_e4_5class.parquet` |

Recipe of every candidate: the recipe of the promoted runs (their `run_metadata.json`): 30 epochs, batch 32, seed 42, `cap_per_class` 3000, `--keep-rpm-level`. Distillation α 0.5.

| # | Run | Data | Teacher | Test macro-F1 (its own split) |
|---|---|---|---|---|
| 1 | `pc_server/2026-11-10_run01` (3-class teacher) | G3-v2 + off-grid | — | 0.897 |
| 2 | `pc_server_5class_w2048/2026-11-10_run01` (PC) | G5-v2 | — | 0.939 |
| 3 | `rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` (RPi5) | G5-v2 | #2 | 0.913 |
| 4a | `esp32_dscnn/2026-11-10_run01` (ESP32) | G3-v2 + off-grid | #1 | 0.842 (Keras) |
| 4b | `esp32_dscnn/2026-11-10_run02` (ESP32) | G3-v2 + off-grid | — | 0.861 (Keras) |

### ESP32 variant selection on validation (§4; [`b4_esp32_selection.json`](b4_esp32_selection.json), 1796 windows)

| Variant | Keras | Quantization | Calibration | Min recall | Macro-F1 |
|---|---|---|---|---|---|
| V0 | 4a | per channel | current recipe | 0.162 | 0.589 |
| **V1 (selected)** | 4a | per tensor | noise-free stratified, train split (334/333/333) | **0.798** | **0.870** |
| V2 | 4b | per channel | current recipe | 0.773 | 0.862 |
| V3 | 4b | per tensor | noise-free stratified | 0.716 | 0.844 |

## Results against the pre-registered criteria (Δ = candidate − promoted, points)

### PC: `pc_server_5class_w2048/2026-11-10_run01` vs `…/2026-11-04_run01`

| Set (windows) | Macro-F1 cand / prom | Δ macro-F1 (≥ −1) | Worst Δ recall (≥ −5) | Δ abstention (\|·\| ≤ 5) |
|---|---|---|---|---|
| E1 (840) | 0.939 / 0.885 | +5.4 ✓ | ball −0.6 ✓ | −4.3 ✓ |
| E3 (1000) | 0.984 / 0.982 | +0.2 ✓ | ball −2.0 ✓ | 0.0 ✓ |
| E4 (540) | 0.994 / 0.684 | +31.1 ✓ | cage −0.9 ✓ | **−16.3 ✗** |

D1 gate on E1: macro-F1 0.939 ≥ 0.88 ✓, min recall 0.893 ≥ 0.70 ✓. Low speed on E4: macro-F1 0.994 ≥ 0.80 ✓, min recall 0.972 ≥ 0.50 ✓. At 50/70 rad/s the promoted run gets inner_race 0.444, normal 0.583, outer_race 0.472; the candidate 1.000 in each.

### RPi5: `rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` (float16) vs `…/2026-11-04_run01`

| Set (windows) | Macro-F1 cand / prom | Δ macro-F1 | Worst Δ recall | Δ abstention |
|---|---|---|---|---|
| E1 (840) | 0.913 / 0.917 | −0.4 ✓ | outer_race −3.0 ✓ | −0.2 ✓ |
| E3 (1000) | 0.978 / 0.980 | −0.2 ✓ | ball −2.0 ✓ | +1.1 ✓ |
| E4 (540) | 0.993 / 0.922 | +7.0 ✓ | ball −0.9 ✓ | **−20.6 ✗** |

D1 gate: 0.913 ✓, min recall 0.869 ✓. Low speed: 0.993 ✓, min recall 0.972 ✓.

### ESP32: `esp32_dscnn/2026-11-10_run01` int8 V1 vs `esp32_dscnn/2026-10-18_run01` (int8)

| Set (windows) | Macro-F1 cand / prom | Δ macro-F1 | Worst Δ recall | Δ abstention |
|---|---|---|---|---|
| E1 (898) | 0.838 / 0.808 | +3.1 ✓ | inner_race **−8.7 ✗** | **−31.2 ✗** |
| E3 (2844) | 0.987 / 0.990 | −0.2 ✓ | inner_race −0.4 ✓ | −3.9 ✓ |
| E4 (1404) | 0.755 / 0.726 | +2.9 ✓ | inner_race **−65.4 ✗** | **−24.1 ✗** |

D1 gate on E1: macro-F1 **0.838 < 0.88 ✗**; min recall 0.746 ✓.

**int8 against its own Keras model** (limits: recall drop ≤ 5 pt per class, macro-F1 drop ≤ 3 pt):

| Set | Macro-F1 Keras → int8 | Worst recall drop |
|---|---|---|
| E1 | 0.842 → 0.838 (0.4 ✓) | inner_race 0.870 → 0.746, **12.4 ✗** |
| E3 | 0.987 → 0.987 (0.0 ✓) | outer_race 0.4 ✓ |
| E4 | 0.886 → 0.755 (**13.1 ✗**) | inner_race 1.000 → 0.346, **65.4 ✗** |

The Keras model 4a learns low speed (E4 inner_race 1.000); the int8 file loses it. That is AI-N28 again, now at the held-out speeds.

## Reading (not a verdict)

- **PC and RPi5** fail only on abstention at low speed, and in the direction of answering more and answering right: E4 macro-F1 rises from 0.684 to 0.994 (PC) and from 0.922 to 0.993 (RPi5), and their recall at 50/70 rad/s is 0.97–1.00 in every class. The pre-registered rule is two-sided (\|Δ abstention\| ≤ 5), so the verdict is "does not meet". Changing how a decrease in abstention counts would be a decision after seeing the results; it is listed for Luciano (decision 1) and does not change this verdict.
- **ESP32** fails on its own merits: the gate, and the int8 loss of inner_race at low speed.

## CPU time (this machine: 12 logical CPUs; `b4_times.tsv`, GNU time user + sys)

| Phase | CPU-h real | Wall | Estimate (pre-registration §8) |
|---|---|---|---|
| Simulation (preconditions, G3, G5, off-grid, E4, E3) | **5.66** | 35 min | ≈ 3.2 CPU-h and 35–45 min (it also included E5, not run) |
| Training (5 models × 30 epochs) | **2.10** | 23 min | ≤ 5 CPU-h (assumption) |
| Exports (float16, 4 int8) | 0.02 | 46 s | — |
| Evaluation (`report_b4.py`) | 0.04 | 90 s | < 1 h |

The simulation took 1.8 times the estimated CPU, at the estimated wall time; more processes ran (8–11, not 6).

## Deviations from the pre-registration (declared)

| # | Deviation | Weight | Why |
|---|---|---|---|
| 1 | The ESP32 per-tensor variant is labelled "(C1, Qt)" in §4 but described as noise-free stratified calibration; that is the repair round's (C2, Qt), selected there. **The description was followed** (C2) | minor | the label and the text disagree; the text matches the repair selection |
| 2 | The 60 off-grid healthy runs follow the acquisition split (train/val/test) like any other run, instead of going all to training | minor | `train_model.py` has no option to force runs into training; training code was not changed |
| 3 | **E5 not executed** | minor (descriptive only, does not decide) | §3.5 gives its seeds and size but not its conditions; inventing them would be improvising |
| 4 | The grid's noise/drift RNG seed is the generator default 7 (not stated in §3.1) | minor | "same generator as today" |
| 5 | §5.4 descriptive items: the bootstrap CI of Δ macro-F1 and the confusion matrices **were computed** by `report_b4.py` and stored in the JSON, but this report first omitted them and wrongly called them not executed (**corrected 2026-11-12**, section §5.4 below). The start-up/steady split is only in E3 | none (reporting error) | — |
| 6 | The report is `b4_results.{md,json}` (night instructions), not `b4_report.{md,json}` (§9) | none | file name only |

No threshold, hyperparameter, set or seed was changed after a result was seen. No candidate was retrained.

<!-- B4-5.4-BEGIN -->

## §5.4 descriptive (pre-registered; no verdict changes)

Computed by `report_b4.py` at evaluation time (stored in `b4_results.json`) and laid out by `python experiments/report_b4_descriptive.py`. CI: 95 % percentile bootstrap of Δ macro-F1 (candidate − promoted, points), 1000 resamples of whole acquisitions, seed 0.

| Tier | Set | Acquisitions | Δ macro-F1 (pt) | 95 % CI (pt) | CI below −1 pt? |
|---|---|---|---|---|---|
| PC | E1 | 70 | +5.4 | [+1.1, +10.9] | no |
| PC | E3 | 260 | +0.2 | [-0.5, +0.8] | no |
| PC | E4 | 60 | +31.1 | [+21.7, +43.2] | no |
| RPi5 | E1 | 70 | -0.4 | [-2.2, +1.3] | yes |
| RPi5 | E3 | 260 | -0.2 | [-1.0, +0.5] | no |
| RPi5 | E4 | 60 | +7.0 | [+1.9, +13.0] | no |
| ESP32 | E1 | 41 | +3.1 | [-2.6, +10.4] | yes |
| ESP32 | E3 | 156 | -0.2 | [-0.4, -0.1] | no |
| ESP32 | E4 | 36 | +2.9 | [-12.3, +16.8] | yes |

**Reading.** No CI changes a verdict (the rule is on the point estimates). Four notes:
- **RPi5 E1:** the point estimate (−0.4 pt) passes, but the CI reaches −2.2 pt: non-inferiority on E1 is not established with confidence.
- **ESP32 E1:** +3.1 pt, CI [−2.6, +10.4]: the gain is not established.
- **ESP32 E3:** the whole CI is below 0 ([−0.4, −0.1] pt): a small but consistent loss, within the −1 pt margin.
- **ESP32 E4:** the CI spans −12.3 to +16.8 pt (36 acquisitions): the low-speed comparison is not informative on its own; the int8-vs-Keras check is.

### Confusion matrices (rows: true class, columns: predicted; argmax, abstention not applied)

**PC · E1**

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 153 (154) | 7 (6) | 2 (1) | 6 (5) | 0 (2) |
| cage | 0 (6) | 160 (154) | 0 (1) | 8 (7) | 0 (0) |
| inner_race | 1 (4) | 6 (2) | 150 (122) | 9 (2) | 2 (38) |
| normal | 0 (6) | 2 (2) | 0 (0) | 166 (160) | 0 (0) |
| outer_race | 0 (9) | 2 (3) | 1 (1) | 6 (1) | 159 (154) |

Candidate (promoted).

**PC · E3**

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 196 (200) | 0 (0) | 0 (0) | 4 (0) | 0 (0) |
| cage | 0 (4) | 200 (196) | 0 (0) | 0 (0) | 0 (0) |
| inner_race | 0 (4) | 0 (0) | 196 (196) | 4 (0) | 0 (0) |
| normal | 0 (6) | 4 (0) | 0 (0) | 196 (194) | 0 (0) |
| outer_race | 0 (4) | 0 (0) | 0 (0) | 4 (0) | 196 (196) |

Candidate (promoted).

**PC · E4**

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 108 (108) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| cage | 0 (0) | 105 (106) | 0 (0) | 3 (2) | 0 (0) |
| inner_race | 0 (0) | 0 (0) | 108 (48) | 0 (0) | 0 (60) |
| normal | 0 (0) | 0 (45) | 0 (0) | 108 (63) | 0 (0) |
| outer_race | 0 (57) | 0 (0) | 0 (0) | 0 (0) | 108 (51) |

Candidate (promoted).

**RPi5 · E1**

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 146 (149) | 10 (6) | 0 (7) | 12 (4) | 0 (2) |
| cage | 0 (4) | 158 (155) | 0 (4) | 10 (5) | 0 (0) |
| inner_race | 4 (2) | 5 (5) | 148 (152) | 10 (8) | 1 (1) |
| normal | 0 (4) | 3 (3) | 0 (1) | 165 (160) | 0 (0) |
| outer_race | 6 (6) | 7 (3) | 0 (2) | 6 (3) | 149 (154) |

Candidate (promoted).

**RPi5 · E3**

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 196 (200) | 0 (0) | 0 (0) | 4 (0) | 0 (0) |
| cage | 0 (4) | 196 (196) | 0 (0) | 4 (0) | 0 (0) |
| inner_race | 0 (4) | 0 (0) | 196 (196) | 4 (0) | 0 (0) |
| normal | 0 (4) | 4 (4) | 0 (0) | 196 (192) | 0 (0) |
| outer_race | 2 (4) | 0 (0) | 0 (0) | 4 (0) | 194 (196) |

Candidate (promoted).

**RPi5 · E4**

| true \ pred | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| ball | 107 (108) | 0 (0) | 0 (0) | 0 (0) | 1 (0) |
| cage | 1 (0) | 105 (106) | 0 (1) | 1 (1) | 1 (0) |
| inner_race | 0 (0) | 0 (0) | 108 (108) | 0 (0) | 0 (0) |
| normal | 0 (0) | 0 (0) | 0 (0) | 108 (108) | 0 (0) |
| outer_race | 0 (30) | 0 (0) | 0 (8) | 0 (0) | 108 (70) |

Candidate (promoted).

**ESP32 · E1**

| true \ pred | inner_race | normal | outer_race |
|---|---|---|---|
| inner_race | 223 (249) | 35 (16) | 41 (34) |
| normal | 0 (17) | 285 (246) | 15 (37) |
| outer_race | 4 (44) | 50 (25) | 245 (230) |

Candidate (promoted).

**ESP32 · E3**

| true \ pred | inner_race | normal | outer_race |
|---|---|---|---|
| inner_race | 932 (936) | 12 (8) | 4 (4) |
| normal | 0 (1) | 940 (942) | 8 (5) |
| outer_race | 0 (3) | 12 (8) | 936 (937) |

Candidate (promoted).

**ESP32 · E4**

| true \ pred | inner_race | normal | outer_race |
|---|---|---|---|
| inner_race | 162 (468) | 0 (0) | 306 (0) |
| normal | 0 (0) | 468 (468) | 0 (0) |
| outer_race | 2 (331) | 0 (4) | 466 (133) |

Candidate (promoted).

- **ESP32 E4:** 306 of 468 inner_race windows go to outer_race in the int8 candidate (the promoted run's error goes the other way). That is where AI-N36 loses inner_race.

### Teachers of the distilled B4 runs (AI-N37)

| Run | Teacher (from `b4_pipeline.sh`) |
|---|---|
| `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-10_run01` | `configs/classifiers/pc_server_5class_w2048/2026-11-10_run01` |
| `configs/classifiers/esp32_dscnn/2026-11-10_run01` | `configs/classifiers/pc_server/2026-11-10_run01` |

<!-- B4-5.4-END -->

## New findings

AI-N34 (PC), AI-N35 (RPi5), AI-N36 (ESP32), AI-N37 (`run_metadata.json` did not name the teacher run; corrected 2026-11-12 in `train_model`, the B4 teachers are listed in §5.4 above). See `docs/AI_TECHNICAL_AUDIT.md` §13.8. Each candidate run has its `audit_findings.json`.
