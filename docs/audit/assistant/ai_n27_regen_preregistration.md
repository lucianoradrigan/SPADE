# AI-N27 option 2 — pre-registration: grids regenerated with `dcm` ×1000, four candidates retrained, evaluation against the promoted models

**Status of this file.**
- Committed **before** any regenerated dataset, candidate or evaluation number exists. Its criteria are not changed after results are seen (`CLAUDE.md`). Any deviation is declared in the report as a deviation.
- Decision: Luciano, 2026-11-03, option 2 of AI-N27 (after `ai_n27_impact.md`: verdict AFFECTS). His message fixes:
  - the full regeneration of the 3- and 5-class grids, not only the affected runs, because a partial one would add a discretization signature correlated with the class;
  - the same assignment of acquisitions to train / validation / test;
  - the unchanged training recipe;
  - the three evaluation sets;
  - the per-tier promotion criterion;
  - nothing is promoted; the report goes to him.
- Choices left to Claude Code are marked **[CC]** and fixed here.
- **Measured before this file (declared):**
  - From `ai_n27_impact.json` / `s21_dataset_scan.json`: the affected acquisitions per split of the promoted runs. 3-class grid: 21 affected, 17 in train, 4 in validation, **0 in test**. 5-class grid: 103 affected, 69 in train, 21 in validation, **13 in test** (8 cage, 3 outer_race, 1 inner_race, 1 ball). This is why §4.2 needs new held-out runs.
  - On the impact study's ×1000 runs (154 acquisitions per version):
    - The S21-1 branch signature (u ≥ 64.9 V with ω > ω_ref) is calibrated on the 1-step production plant. It fires on **none** of the runs, GEM1000 included, so it is not usable at ×1000.
    - Minimum armature current over the noise-free runs (59): GEM1000 −156 A, with 48 runs below −1e-6 A; DCM1000 0.0 A, with 0 runs below. This is precondition P2.
  - Window assignment: the R4 candidates reproduced the promoted runs' split exactly (equal `split.group_hashes` and `n_windows` for all four pairs). That works because `build_classification_windows` and `grouped_train_val_test_split` depend only on the acquisition names, their order, the labels, the row count and the seed. This is precondition P3.
  - No regenerated sample, model output or metric of this study was computed.

## 1. Generation

### 1.1 Generator change

`experiments/generate_diagnosis_grid_dataset.py` gains `--dc-converter-model {gem,dcm}` (default `gem`, i.e. the current behaviour) and `--dc-plant-substeps N` (default none). Both are passed unchanged to `Scenario.dc_converter_model` / `Scenario.dc_plant_substeps`. The columns `grid_dc_converter_model` and `grid_dc_plant_substeps` are written only when the options are given. `experiments/report_classifier_evaluation.py` gains the same two options for its cache generation. **No default changes** (S21-4): the plant default and `dc_perm_ex_v1` stay GEM.

### 1.2 Datasets (new files; the existing ones are kept unchanged)

| Set | Command (all with `--workers 10`) | Output | Runs |
|---|---|---|---|
| 3-class grid | `generate_diagnosis_grid_dataset.py --dc-converter-model dcm --dc-plant-substeps 1000 --seeds 3 --duration-s 2.5 --seed 7 --out data/diagnosis_grid_dataset_dcm.parquet` | | 315 |
| 5-class grid | same + `--faults outer_race,inner_race,ball,cage --out data/diagnosis_grid_dataset_5class_dcm.parquet` | | 525 |
| Generalization (§4.3) | `report_classifier_evaluation.py --dc-converter-model dcm --dc-plant-substeps 1000 --extra-faults ball,cage --seeds 4 --first-seed 200 --duration-s 1.5 --cache data/eval_cache_5class_dcm.parquet` (cache generation only) | | 248 |
| AI-N27 held-out (§4.2) | `experiments/generate_ai_n27_heldout.py` (§4.2) | `data/ai_n27_heldout_{3class,5class}_dcm.parquet` | 45 + 90 |

The grids use the same `build_jobs(3, 2.5, 7, faults)` as the originals: same acquisitions, names, speeds, severities, noise and plant-drift draws, and seeds. Only the converter model and the substeps change.

**Estimated cost** (from the impact study: 49 s of wall time per 2.5 s run with 10 processes): grids ≈ 11.5 h, held-out ≈ 1.8 h, generalization ≈ 2 h, so **≈ 15 h** of generation. Order: 3-class, 5-class, held-out, generalization. Disk: ≈ 2.1 GB of new parquet (22 GB free).

### 1.3 Preconditions (checked in this order; if one fails, stop and report — no training and no evaluation)

- Checked by `experiments/report_ai_n27_regen_preconditions.py p1|p2|p3` (outputs `ai_n27_regen_p{1,2,3}.json`).
- **P1, before generating: the generator still reproduces the stored data in default mode.** Three acquisitions per grid [CC: the first, middle and last job of `build_jobs`] are regenerated without the new options. The criterion is that of amendment 2:
  - `current_r`, `rpm`, `voltage_v`, `torque_nm` bit-identical;
  - `acc_x/y/z` max relative difference ≤ 1e-12.
- **P2, after generating: the regenerated grids have the corrected physics.**
  - Every noise-free acquisition (`grid_electrical_noise_pct` = 0) has min `current_r` ≥ −1e-6 A.
  - `grid_dc_converter_model` = "dcm" and `grid_dc_plant_substeps` = 1000 in every row.
  - Same acquisitions (set of `source_file`), in the same order, with the same row count and timestamps as the stored grid.
- **P3, after training: same split.** Each candidate's `metrics.json` has `split.group_hashes` (train, validation, test) and `split.n_windows` equal to the promoted run of its tier. For the PC 3-class teacher, the comparison is with `pc_server/2026-10-25_run01`.

## 2. Training (recipe unchanged; one run per candidate, seed 42)

Common flags (the promoted and R4 recipe): `python experiments/train_model.py --epochs 30 --batch-size 32 --seed 42 --keep-rpm-level --cap-per-class 3000`.

| # | Candidate | Config | Dataset | Distillation |
|---|---|---|---|---|
| 1 | PC 3-class (teacher only; no tier, no criterion) | `configs/classifiers/pc_server.yaml` | `diagnosis_grid_dataset_dcm` | — |
| 2 | ESP32 | `configs/classifiers/esp32_dscnn.yaml` | `diagnosis_grid_dataset_dcm` | `--distill --distill-alpha 0.5 --teacher-run <#1>` |
| 3 | PC (5-class, w2048) | `configs/classifiers/pc_server_5class_w2048.yaml` | `diagnosis_grid_dataset_5class_dcm` | — |
| 4 | RPi5 | `configs/classifiers/rpi5_resnet1d_se_5class_w2048.yaml` | `diagnosis_grid_dataset_5class_dcm` | `--distill --distill-alpha 0.5 --teacher-run <#3>` |

- **TFLite export:**
  - RPi5 float16: `python -m experiments.export_tflite --domain dc_motor --tier rpi5 --block classifier --run-dir <#4>`.
  - ESP32 int8: `--tier esp32 --run-dir <#2> --dataset data/diagnosis_grid_dataset_dcm.parquet --n-calibration-samples 1000`, without `--calibration-noise-free`, like the promoted ESP32 run (exported 2026-10-18, before that flag existed).
- **Not done:**
  - no C2000 export;
  - no registry change and no `promote_run.py`;
  - no hyperparameter or seed search, and no retraining "to rescue" a candidate.
- **Reruns:** if a run crashes (exception, not a bad metric), the same command is rerun once and declared.

## 3. Models compared, per tier

| Tier | Promoted (reference) | Candidate | Artifact evaluated (both) |
|---|---|---|---|
| PC | `pc_server_5class_w2048/2026-10-20_run01` | #3 | Keras |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` | #4 | float16 TFLite |
| ESP32 | `esp32_dscnn/2026-10-18_run01` | #2 | int8 TFLite |

- **Deployed artifacts.** The decision uses each tier's deployed artifact. Keras numbers for RPi5 and ESP32 are reported as descriptive only.
- **Preprocessing and calibration.** Each model uses its own preprocessing (window, channels, rpm level scaling) and its own calibration (`temperature`, `abstain_below`) from its `metrics.json`.
- **Windows.** All windows come from `build_classification_windows`, the training builder.

## 4. Evaluation sets

The metrics are computed over the windows of the tier's known classes. ESP32 uses its 3 classes; windows of other classes are excluded.

### 4.1 E1 — test split with `dcm`

- Windows of the test-split acquisitions (identical for the promoted run and the candidate, P3) of the regenerated grid of the tier: 3-class for ESP32, 5-class for PC and RPi5.
- Built with the same cap and windowing as training, so the candidate's E1 is its own recorded test set.

### 4.2 E2 — AI-N27 subset

- **E2a, decides: new held-out simulations in the affected regime** [CC], `dcm` ×1000, never used in training.
  - Cells: every (speed, severity) cell of the grid where the S21-1 scan found affected acquisitions.
    - 3-class: (80, ×1.0), (80, ×1.5), (80, ×2.0), (100, ×1.5), (100, ×2.0).
    - 5-class: the 18 cells of `s21_dataset_scan`: 80 and 100 rad/s at ×0.75, ×1.0, ×1.5, ×2.0; 125 rad/s at ×1.0, ×1.5, ×2.0; 150, 175 and 200 rad/s at ×1.5, ×2.0; 220 rad/s at ×2.0.
  - Every class of the tier per cell, including healthy.
    - 3-class: 3 seeds per class and cell, so 45 runs.
    - 5-class: 1 seed per class and cell, so 90 runs.
  - 2.5 s per run.
  - Noise and plant drift are drawn as in `build_jobs`: one third noise-free, the rest U(0, 100) % each; ψ_e × U(0.9, 1.1); J × U(0.8, 1.5). The RNG seed is 20261103 and the run seeds start at 60001 (no overlap with the grid's 1001–1525 or the evaluation's 200–203).
  - All windows of each run, without a cap, start-up included, as in the impact study's "all" regime.
- **E2b, descriptive only:** the 13 affected test-split acquisitions of the 5-class grid, in their `dcm` version (a subset of E1). Too few to decide on.

### 4.3 E3 — regenerated generalization

- `data/eval_cache_5class_dcm.parquet` (§1.2): the 13 conditions of `report_classifier_evaluation.CONDITIONS`, with the same seeds, durations and condition time windows (start-up = t < 0.5 s; the others t ≥ 0.5 s).
- The criterion uses the **pooled windows of the 13 conditions** (known classes). Per-condition results are descriptive.
- The `ood_unseen_fault_type` runs (ball and cage for ESP32) are descriptive only.

## 5. Metrics

- **Macro-F1:** argmax over the model's classes, on all windows of known classes. This is the definition of `test_macro_f1` in `models/training.py`.
- **Per-class recall** (recall of class k = the fraction of class-k windows predicted as k).
- **Abstention:** the fraction of windows whose calibrated max probability < `abstain_below`.
- **Also reported (descriptive):**
  - accuracy;
  - the confusion matrix;
  - a 95 % bootstrap CI of Δ macro-F1, resampling acquisitions (1000 resamples, seed 0);
  - start-up versus steady-state split in E2a.

## 6. Promotion criterion, per tier (decision rule)

A tier's candidate **MEETS THE CRITERION** if and only if all of the following hold (Δ = candidate − promoted, in percentage points):

1. **Current gate (D1, 2026-10-22), on E1:** candidate macro-F1 ≥ 0.88, and recall ≥ 0.70 in every class.
2. **Non-inferiority in macro-F1, on each of E1, E2a and E3:** Δ macro-F1 ≥ −1.0 point.
3. **No class drops more than 5 points, on each of E1, E2a and E3:** Δ recall ≥ −5.0 points for every known class.
4. **Abstention, on each of E1, E2a and E3:** |Δ abstention| ≤ 5.0 points.

- **How the rule is applied.**
  - Point estimates decide. The CIs do not change the verdict.
  - A tier that does not meet every check is reported as **DOES NOT MEET THE CRITERION**, with the failing checks listed.
  - The verdict is per tier. The three tiers are independent.
- **Not in the rule.** Meeting the criterion is **not** a promotion. Nothing is promoted in this task; Luciano decides from the report.

## 7. If the ESP32 gets worse, or a candidate comes out worse than the promoted model

- **A candidate that does not meet the criterion (any tier, ESP32 included):**
  - Its promoted model stays in the registry with the AI-N27 warning already measured.
  - The report says which checks failed and by how much, with E2b and the per-condition E3 results as context.
  - **Nothing is retried in this round:** no other seed, no recipe change, no other teacher, no other int8 calibration, no threshold change.
  - Possible next steps are listed only as proposals. Each would need a new pre-registration and Luciano's decision.
- **ESP32 in particular:**
  - If the ESP32 fails and PC/RPi5 meet the criterion, the report says so, and the tiers can be decided separately.
  - The promoted ESP32 and the C2000 export that derives from it are not touched.
  - The board re-measurement (ESP32 and C2000) applies only if Luciano promotes a candidate. It is out of scope here.
- **A candidate better than the promoted model on E2a but worse on E1 or E3:** it is reported as not meeting the criterion. No check is weighted differently after the fact.
- **A precondition fails (P1–P3):** stop and report before training or evaluating.

## 8. Report

`docs/audit/assistant/ai_n27_regen.{md,json}`:
- the generation commands, times, files and SHA-256;
- P1–P3;
- the training commands and runs;
- per tier: E1, E2a, E2b and E3 for the promoted model and the candidate, the four checks and the verdict.

The candidate run directories are committed as candidates (as R4), with `audit_findings.json` saying they are not promoted.
