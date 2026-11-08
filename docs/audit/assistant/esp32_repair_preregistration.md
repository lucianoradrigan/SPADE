# AI-N28 — pre-registration: repair round of the ESP32 int8 classifier

**Status: APPROVED by Luciano (2026-11-08, decision E1 of `PLAN_CONSOLIDADO_2026-11-08.md`).** Committed before any export, calibration, generation or evaluation of this round. Its criteria are not changed after results are seen (`CLAUDE.md`). Any deviation is declared in the report.

- **Decision being executed:**
  - Luciano, 2026-11-04 (`docs/AI_TECHNICAL_AUDIT.md` §13.2, "Next step"): a pre-registered repair round on the same Keras candidate (calibration set, int8 per tensor or per channel), then, only if needed, another distillation alpha or no distillation; the variant is chosen on validation by a fixed rule and tested once.
  - Refined by E1, 2026-11-08:
    - **option (b)**: both stages run, and the selection is made over all their variants;
    - **a new E2a with unused seeds**, because the current E2a already informed the design of this round (§0).
- Choices left to Claude Code are marked **[CC]**.

## 0. What is already known (measured before this file, declared)

From `ai_n27_regen.{md,json}` (2026-11-04), candidate `esp32_dscnn/2026-11-04_run01` against promoted `esp32_dscnn/2026-10-18_run01`:

| | E1 macro-F1 | E2a macro-F1 | E3 macro-F1 | outer_race recall E1 / E2a / E3 |
|---|---|---|---|---|
| Promoted, int8 (deployed) | 0.899 | 0.843 | 0.990 | 0.893 / 0.699 / 0.988 |
| Candidate, int8 | 0.718 | 0.563 | 0.679 | 0.383 / 0.047 / 0.206 |
| Candidate, **Keras float** | 0.875 | 0.709 | 0.961 | 0.762 / 0.314 / 0.899 |

**Consequence.** The int8 model approximates the float one. A repair that only re-quantizes the same Keras candidate is bounded by the float numbers. Even if its int8 matched Keras exactly, on E2a:
- Δ macro-F1 would be 0.709 − 0.843 = **−13.4 points** (the §6 criterion allows −1.0);
- Δ outer_race recall would be 0.314 − 0.699 = **−38.5 points** (allowed −5.0).

So **stage 1 alone cannot meet the §6 criterion on E2a.** It can still show how much of the collapse is the quantization: on E1 and E3 the float model is close to the gate. The criterion can only be met by a new Keras model, i.e. stage 2.

**Q1, decided (E1): option (b).** Stages 1 and 2 both run, and the validation rule ranks all their variants together (§4.2).

**The current E2a is no longer an independent test.** Its numbers (the table above) were used to write this design. The decision set of §5 therefore uses a **new E2a (E2a′)** with seeds never used before (§5.1). The old E2a is reported for the selected variant as descriptive only.

## 1. Preconditions (checked in order; if one fails, stop and report)

- **R1, reproduction:** re-exporting the candidate with the current recipe (variant **V0** below) with the code of this round gives the same int8 outputs as the committed `configs/classifiers/esp32_dscnn/2026-11-04_run01/model.tflite`. Measured on every validation window: identical int8 output tensors. The file SHA-256 is reported; a byte difference alone does not fail R1.
- **R2, per-tensor switch:** the installed TensorFlow converter accepts `_experimental_disable_per_channel = True`. A model exported with it has one quantization scale per weight tensor; a model exported without it has one per output channel in the conv / depthwise layers. Both are checked in the exported file. If the switch does not exist, stop and report: the per-tensor arm cannot be built without a new dependency.
- **R3, split:** the validation windows rebuilt by the evaluation code have the candidate's `split.group_hashes["val"]` and `n_windows["val"]` = 1811.
- **R4, E2a′ is new data:**
  - no run seed of E2a′ (§5.1) appears in any existing dataset: grids 1001–1525, evaluation cache 200–203, AI-N27 held-out 60001–60135;
  - E2a′ is generated after this file is committed;
  - its plant columns are not bit-identical to any acquisition of `ai_n27_heldout_3class_dcm.parquet`.

## 2. Stage 1: re-quantize the same Keras candidate (no training)

**Model:** `configs/classifiers/esp32_dscnn/2026-11-04_run01/model.weights.h5`, unchanged.

**Calibration sets** (all 1000 windows; windows built with `build_classification_windows` and the run's `level_scaling`):

| Id | Calibration windows |
|---|---|
| **C0** | Current recipe (`export_tflite._representative_windows`): 1000 windows drawn with seed 0 from all windows of `data/diagnosis_grid_dataset_dcm.parquet`. This includes validation and test windows. Kept only as the reference, declared. |
| **C1** [CC] | Training split only, stratified by class: 334 / 333 / 333 windows (inner_race / normal / outer_race), seed 0, all noise levels. |
| **C2** [CC] | Training split only, stratified by class, noise-free acquisitions only (`grid_electrical_noise_pct` = 0 and `grid_mechanical_noise_pct` = 0): 334 / 333 / 333, seed 0. If a class has fewer noise-free windows, all are used and the count is reported. |

**Weight quantization:**

| Id | Weights |
|---|---|
| **Qc** | Per channel: the converter's default, i.e. what the current export does. |
| **Qt** | Per tensor: `_experimental_disable_per_channel = True`. |

Activations and the int8 input/output are as today (`export_int8`).

**Variants:** V0 = (C0, Qc), the existing candidate; V1 = (C0, Qt); V2 = (C1, Qc); V3 = (C1, Qt); V4 = (C2, Qc); V5 = (C2, Qt).

- **Code:** `export_int8` and `experiments/export_tflite.py` gain an option for the calibration set and one for per-tensor weights. The defaults stay what they are now, so existing exports are unchanged (test). Variants are written to a scratch directory. Nothing in `configs/classifiers/` is overwritten.

## 3. Stage 2: a new Keras model (another distillation alpha or none)

- **When:** see §4.2 (Q1).
- **Recipe:** the candidate's recipe unchanged except the alpha (`ai_n27_regen_preregistration.md` §2, row 2):
  - `experiments/train_model.py --config configs/classifiers/esp32_dscnn.yaml --dataset data/diagnosis_grid_dataset_dcm.parquet --epochs 30 --batch-size 32 --seed 42 --keep-rpm-level --cap-per-class 3000`;
  - teacher `configs/classifiers/pc_server/2026-11-04_run01`.

| Model | Alpha | Meaning |
|---|---|---|
| K0 | 0.5 | The existing candidate (stage 1) |
| **K1** [CC] | 0.75 | Closer to the true labels |
| **K2** | 1.0, no `--distill` | No distillation |

- **Exports:** each of K1, K2 is exported in the six ways of stage 1 (C0 / C1 / C2 × Qc / Qt), giving V6–V11 (K1) and V12–V17 (K2).
- **Split:** P3 of the AI-N27 round must hold, i.e. the same `split.group_hashes` as the candidate. If it does not, stop.
- **Limits:** one run per alpha, seed 42. No other seed, no other epoch count, no other architecture. A crash is rerun once and declared.
- The new run directories are kept as candidates (`audit_findings.json`: not promoted).

## 4. Selection on validation (fixed rule)

### 4.1 The rule

- **Data:** each variant's deployed artifact (int8 TFLite) is evaluated on the **validation split** of the dcm 3-class grid (1811 windows, R3). Each Keras model uses its own `metrics.json` calibration (temperature, `abstain_below`).
- **Ranking**, in order:
  1. highest **minimum per-class recall**;
  2. then highest macro-F1;
  3. then the lowest variant number. This puts stage 1 before stage 2 and the current recipe first.
- **Reported for every variant:** the validation numbers: macro-F1, per-class recall, abstention, confusion matrix.
- **Not used for the choice:** E1, E2a, E3.

### 4.2 Which variants compete

Decision E1, option (b): stages 1 and 2 both run, and the rule ranks **V0–V17** together. There is no validation gate that skips stage 2.

**Caveat:** the validation split was also used in training (temperature fit) and is the source of the abstention threshold. It is a selection set, not an unbiased estimate. That is why the test below is run once.

## 5. One test of the selected variant

### 5.1 E2a′: new held-out runs in the affected regime

- **Generator:** `experiments/generate_ai_n27_heldout.py --grid 3class`, with two new options whose defaults keep today's set, so `ai_n27_heldout_3class_dcm` stays reproducible: `--rng-seed` and `--first-run-seed`.
- **Draws:** E2a′ uses `--rng-seed 20261108 --first-run-seed 70001`. These are the same cells, the same 3 seeds per class and cell (45 runs, 2.5 s, dcm ×1000) and the same noise and plant-drift distributions. Only the draws and the seeds are new.
- **Output:** `data/esp32_repair_heldout_3class_dcm.parquet`. Not committed, like every dataset.
- **Cost:** ≈ 45 runs × 612 s CPU ≈ 7.7 h CPU, ≈ 50 min with 10 processes.

### 5.2 The test

The selected variant only, once, with the evaluation of `ai_n27_regen_preregistration.md`:
- **sets:** E1 (§4.1), **E2a′** (§5.1) and E3 (§4.3): the files `diagnosis_grid_dataset_dcm`, `esp32_repair_heldout_3class_dcm`, `eval_cache_5class_dcm`. The promoted model is also evaluated on E2a′, for the Δ;
- **metrics:** §5;
- **criterion:** §6, against the promoted `esp32_dscnn/2026-10-18_run01` int8:
  1. D1 gate on E1;
  2. Δ macro-F1 ≥ −1.0 on E1, E2a′ and E3;
  3. Δ recall ≥ −5.0 per class on E1, E2a′ and E3;
  4. |Δ abstention| ≤ 5.0 on E1, E2a′ and E3.

The verdict is **MEETS / DOES NOT MEET THE CRITERION**, with the failing checks.
- The other variants are never evaluated on E1, E2a′ or E3.
- The old E2a (`ai_n27_heldout_3class_dcm`) is reported for the selected variant as **descriptive only**.

## 6. If it does not meet the criterion

- The promoted ESP32 stays, with its AI-N27 warning; AI-N28 stays open with the new evidence.
- Nothing more is retried in this round.

## 7. Not in this round

- No promotion: a variant that meets the criterion is reported, and Luciano decides.
- No C2000 export.
- No board measurement: it applies only if a variant is promoted.
- No change to the PC or RPi5 models, to AI-N29 or to any default.

## 8. Cost (estimate)

- **Stage 1:** 6 exports, each a few minutes of CPU, plus validation inference of 6 int8 models on 1811 windows. Under 1 h.
- **Stage 2:** 2 trainings. The 4 trainings of the AI-N27 round took 16 min together, so 2 should take about 10 min. Then 12 exports and validation, about 1–2 h.
- **E2a′:** ≈ 50 min with 10 processes (§5.1).
- **Test:** about 15 min.
- **Disk:** below 100 MB.

## 9. Report

`docs/audit/assistant/esp32_repair.{md,json}`, with:
- the commands;
- R1–R3;
- the validation table of every variant run;
- the selected variant and why (the rule);
- the test of that variant against the promoted model;
- the four checks and the verdict.

AI-N28 is updated in `docs/AI_TECHNICAL_AUDIT.md`.
