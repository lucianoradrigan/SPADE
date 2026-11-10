# B4: consolidated model round — PRE-REGISTRATION (draft for approval, P6)

**Status: DRAFT, NOT APPROVED AS A WHOLE.** Written 2026-11-09 (round C, T9, decision C11). **Decisions D-B4-1 and D-B4-3 were approved by Luciano on 2026-11-09, and D-B4-2 was approved with a change (§0). D-B4-4 is pending (options in §7). B4 is not executed until Luciano approves this file.**

**Thresholds fixed before any data.** Every threshold in §5 (the AI-N27 rule, the absolute low-speed criterion and the int8-vs-Keras limit) was fixed on 2026-11-09, before any B4 dataset, candidate or evaluation existed. They are not changed after results are seen.

The machine-readable plan is `configs/b4_plan.yaml`. `tests/test_b4_plan.py` checks that the healthy off-grid training conditions do not coincide with the evaluation conditions. Nothing has been generated, trained, exported or measured for B4. Once Luciano approves it, this file is committed **before** any B4 data exists, and its criteria do not change after results are seen (`CLAUDE.md`). Any deviation is declared as a deviation. Every result is **simulated**; the single C2000 visit replays simulated windows from flash.

Findings addressed: **AI-N14** and **AI-N29** (low speed), **AI-N28** (ESP32 int8), **AI-N26** (C2000 timing at 10 kHz) and **AI-N27** (still open for the ESP32). Not in B4: AI-N24 and AI-N25 (C2000 size and speed of the RPi5 models), phase 2 of the assistant and the induction motor.

## 0. Decisions needed before approval (marked [D])

| ID | Question | Decision (Luciano, 2026-11-09) |
|---|---|---|
| D-B4-1 | Speeds added to the training grid below 80 rad/s | **Approved:** 40 and 60 rad/s in training; held-out low speeds **50 and 70 rad/s** (§3.4) |
| D-B4-2 | Thresholds | **Approved with a change:** within the grid, the AI-N27 rule; at low speed (E4), macro-F1 ≥ 0.80 and recall ≥ 0.50 in every class; **int8 against Keras: at most 5 points of recall per class and 3 of macro-F1** (§5) |
| D-B4-3 | Healthy off-grid conditions in the ESP32 training (AI-N28) | **Approved**, with a test that they do not coincide with the evaluation conditions (§3.5) |
| D-B4-4 | Which runs go to the C2000 visit | **Pending:** options and costs in §7 |

## 1. Data: the v2 plants

- **Plant:** every new dataset uses `dc_perm_ex_v2` (decision E3), with the converter `dcm_exact` and the stock motor and load. The DC dataset scripts take `--plant dc_perm_ex_v2`. Their default is already v2.
- **Equivalence with the current data:** v2 data have already shown classifier agreement 1.000 with the dcm ×1000 data the promoted PC and RPi5 models were trained on (`dc_perm_ex_v2_equivalence.md`). B4 nevertheless regenerates every set, so that training and evaluation use one plant.
- **Sampling rate:** 10 kHz, written to `fs_hz` in every run (AI-N26).

## 2. Preconditions (checked in order; if one fails, stop and report)

- **Q1, the generator still reproduces the stored data:** the first, middle and last acquisition of the current 5-class dcm grid are regenerated with `--plant dc_perm_ex_v2`. The expected values are recorded in the report:
  - equivalence within the tolerance of `dc_perm_ex_v2_equivalence.md` (NRMS ≤ 1e-4 against dcm ×1000);
  - every noise-free run has min `current_r` ≥ −1e-6 A.
- **Q2, the split is by acquisition:** no acquisition is in two splits. This is checked on the windows, as in AI-N27 P3.
- **Q3, the held-out seeds:** the E4 and E5 seeds do not overlap with any grid seed or with any earlier evaluation seed (§3).

## 3. Datasets (all new; none used before for design)

### 3.1 Training grids (extended)

| Grid | Classes | Speeds (rad/s) | Severities (× calibrated) | Seeds per cell | Runs | Duration |
|---|---|---|---|---|---|---|
| G3-v2 (ESP32) | normal, inner_race, outer_race | **40, 60**, 80, 100, 125, 150, 175, 200, 220 | 0.5, 0.75, 1.0, 1.5, 2.0 | 3 | 405 (+ §3.5) | 2.5 s |
| G5-v2 (PC, RPi5) | + ball, cage | the same 9 | the same 5 | 3 | 675 | 2.5 s |

- **Same generator as today:** `generate_diagnosis_grid_dataset.py`, with the speed list extended by an option. **The default does not change.**
- **Noise and plant drift:** as in the current grid (one third noise-free; ψ_e × U(0.9, 1.1); J × U(0.8, 1.5)).
- **Split:** the same rule (by acquisition, seed 42).
- **The B4 grid seeds start at 80001.**

### 3.2 E1: test split of each grid

The recorded test set of each candidate, with the same cap and windowing as in training.

### 3.3 E3: regenerated generalization

`report_classifier_evaluation.CONDITIONS` (13 conditions), regenerated with v2 and with the same seeds and windows as `eval_cache_5class_dcm`.

### 3.4 E4: low speed, held out (AI-N14, AI-N29)

- **Conditions:** speeds **50 and 70 rad/s**, which are not in the grid; severities ×1.0 and ×1.5; every class of the tier.
- **Size:** 3 seeds per cell, so 2 × 2 × 3 = 12 runs per class.
- **Simulation:** 2.5 s per run, with the steady-state windows (t ≥ 0.5 s). Noise and drift are drawn as in the grid.
- **Seeds:** RNG seed 20261120, run seeds from 90001.

### 3.5 ESP32 only: healthy off-grid conditions in training (AI-N28) [D-B4-3]

- **What is added to the training split of G3-v2:** 60 **normal** runs outside the grid cells. These are 20 conditions × 3 seeds, listed exactly in `configs/b4_plan.yaml`:
  - speeds 30, 90, 140 and 210 rad/s;
  - extra passive loads `load_torque_nm` of 1, 2 and 4 N·m at 100 and 175 rad/s;
  - noise of 120 % and 150 % at 125, 150 and 200 rad/s;
  - plant drift ψ_e −10 % with J +50 % at 90, 140 and 210 rad/s;
  - a supply of 60 V and 70 V at 150 rad/s;
  - one combined load and noise condition.
- **Declared (2026-11-09, before approval and before any data):** the first draft listed **70 rad/s** among these speeds. That coincides with an E4 held-out speed, so it was replaced by 30 rad/s. `tests/test_b4_plan.py` now prevents any coincidence with 50 or 70 rad/s, with a grid cell or with an E3 condition.
- **Why:** in the repair round, normal recall fell from 0.994 to 0.687 on E3 (`esp32_repair.md`). The point is to teach the model what healthy looks like outside the grid.
- **Isolation:** none of these conditions coincides with an E3 or E4 condition.
- **E5 (ESP32, descriptive):** 30 new healthy runs under the same kinds of off-grid conditions, with seeds from 95001. Their false-alarm rate is reported, but it does not decide.

## 4. Candidates and exports (one run each, seed 42; a crash is rerun once and declared)

| # | Tier | Config | Data | Distillation | Deployed artifact |
|---|---|---|---|---|---|
| 1 | teacher (no tier) | `pc_server.yaml` (3-class) | G3-v2 (+ §3.5) | — | — |
| 2 | PC | `pc_server_5class_w2048.yaml` | G5-v2 | — | Keras |
| 3 | RPi5 | `rpi5_resnet1d_se_5class_w2048.yaml` | G5-v2 | α 0.5, teacher #2 | float16 TFLite |
| 4a | ESP32 | `esp32_dscnn.yaml` | G3-v2 + §3.5 | α 0.5, teacher #1 | int8 TFLite |
| 4b | ESP32 | `esp32_dscnn.yaml` | G3-v2 + §3.5 | none | int8 TFLite |

- **ESP32 int8:** each of 4a and 4b is exported in two ways:
  - per channel, the current default;
  - per tensor with noise-free stratified calibration: (C1, Qt), the configuration selected in the repair round.

  That gives **4 ESP32 variants**. One is selected on the **validation** split by the repair round's rule (`esp32_repair_preregistration.md` §4.1), and only the selected one is tested.
- **No other seeds, epochs or architectures.** No retraining to rescue a candidate.

## 5. Criteria, per tier (Δ = candidate − promoted, in percentage points)

### 5.1 The AI-N27 rule (unchanged), on E1, E3 and E4

A tier's candidate **MEETS THE CRITERION** if and only if all of these hold:
1. **D1 gate on E1:** macro-F1 ≥ 0.88, and recall ≥ 0.70 in every class.
2. **Non-inferiority on E1, E3 and E4:** Δ macro-F1 ≥ −1.0 point.
3. **No class drops more than 5 points** on E1, E3 and E4: Δ recall ≥ −5.0 points for every known class.
4. **Abstention** on E1, E3 and E4: |Δ abstention| ≤ 5.0 points.

The comparison is made with the currently promoted run of the tier, on the same windows and with each model's own preprocessing and calibration.

### 5.2 Low speed (AI-N29), absolute, on E4 [D-B4-2]

5. **PC and RPi5:** macro-F1 ≥ 0.80 and recall ≥ 0.50 in every class, on the E4 windows of known classes (argmax, as `test_macro_f1`). Abstention is reported.

### 5.3 ESP32 (AI-N28)

6. **Deployed int8 artifact against its own Keras model,** on E1, E3 and E4 (D-B4-2 as changed by Luciano):
   - recall drop ≤ **5** points in every class;
   - macro-F1 drop ≤ 3 points.

### 5.4 Descriptive only

- Accuracy, confusion matrices, start-up and steady-state windows.
- A bootstrap 95 % CI of Δ macro-F1, resampling acquisitions (1000 resamples, seed 0).
- E5 false alarms; the `ood_unseen_fault_type` runs.

## 6. If a tier does not meet the criterion

- **The promoted run stays.** The report lists the failing checks; nothing is weighted differently after the fact.
- **No rescue retraining in B4.** A further attempt needs a new pre-registration.
- **ESP32:** if it fails again, the next options go to Luciano: another architecture, a float model on a larger MCU, or accepting AI-N28 as a documented limitation. AI-N27 then stays open for the ESP32.
- **The tiers are decided separately.** Meeting the criterion is **not** a promotion: Luciano decides from the report (`promote_run.py` only with his OK).

## 7. The single C2000 visit (AI-N26) [D-B4-4, PENDING]

- **The window at 10 kHz:** 51.2 ms (512 samples), 25.6 ms (256 samples).
- **Criteria:** C1–C5 of `docs/audit/c2000/criteria.md`, with C5's window at 10 kHz, committed again before the visit. The verdicts of 2026-10-30 stay as recorded.
- **Kits:** exported with `fs_hz` in the manifest.
- **Note:** the board latency does not depend on the assumed rate, so for the three runs already measured, C5 at 10 kHz is already known post hoc (`c5_post_hoc_10khz.md`). A visit adds the measurement with the corrected manifest, and measures any **new** model.

**Options (pick one):**

| Option | Kits | What it closes | Luciano at the board | Claude Code |
|---|---|---|---|---|
| **A** | The ESP32 classifier that results from B4: the new one if it meets the criterion, else the current one | AI-N26 for the classifier; a new model measured on the board | ≈ 45 min (connect, flash 1 kit, collect, one retry allowed) | ≈ 15 min export + report |
| **B (recommended)** | A + `esp32_tiny/2026-10-19_run01` | AI-N26 for both models that met C5 at 5 kHz; `esp32_tiny` confirmed at 25.6 ms with the corrected manifest | ≈ 1 h 15 min | ≈ 25 min |
| **C** | B + `rpi5_tcn/2026-10-20_run01` | also re-measures `rpi5_tcn` (AI-N24: 520 ms, already 20× its window; it fails at either rate) | ≈ 1 h 45 min | ≈ 35 min |
| **D** | no visit | AI-N26 stays "mitigated (software)", closed only post hoc; a new ESP32 model would have no board measurement | 0 | 0 |

- **Not possible in any option:** the RPi5 classifier (AI-N25: 448 KB arena against 128 KB).
- **Recommendation: B.** It measures every model that can meet C5 and skips one whose result is already known to fail.

## 8. Cost (estimate)

| Item | Estimate | Basis |
|---|---|---|
| Simulation: G3-v2, G5-v2, E3, E4, §3.5, E5 | ≈ 1 080 grid runs + ≈ 330 evaluation runs ≈ 3 500 simulated s ≈ **3.2 CPU-h**; about **35–45 min of wall time** on 6 processes | v2 cost 3.3 CPU s per simulated s (`dc_perm_ex_v2_convergence.md`, dcm_exact) |
| Training | 5 Keras models × 30 epochs. **Earlier runs did not record their training time.** Planning assumption: ≤ 1 CPU-h per model, so ≤ 5 h, run overnight. The first B4 step records the real time | assumption, to be measured |
| Exports and evaluation | 4 int8, 1 float16; E1/E3/E4/E5 inference: < 1 h | earlier rounds |
| **Luciano, manual** | approve this file (~30 min); decide promotions from the report (~1 h); the C2000 session (~1.5–2 h at the board: connect, flash 3 kits, collect) | — |

## 9. Report

The report goes in `docs/audit/b4/b4_report.{md,json}` and gives:
- each check per tier, with its numbers and the command that produced them;
- the preconditions;
- the deviations;
- what was not executed;
- the C2000 table at 10 kHz.

The findings are updated (AI-N14, AI-N26, AI-N27, AI-N28, AI-N29) with **fixed / mitigated / open** as the numbers say.
