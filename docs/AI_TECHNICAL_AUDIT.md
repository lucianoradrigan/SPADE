# AI technical audit — SPADE (2026-10-18)

**Scope.** This audit covers:
- the diagnosis classifiers;
- the forecasting regressors;
- the DPC neural controller, insofar as it is a learned model;
- the transfer-learning / promotion flow;
- TFLite export;
- the monitoring agents and alerts.

**Base.** Commit `b053de7` plus this audit's working-tree changes. It builds on the previous audit, [`AUDIT_ANSWERS.md`](../AUDIT_ANSWERS.md) (Q01–Q30) and [`CHANGELOG_IMPLEMENTATION.md`](../CHANGELOG_IMPLEMENTATION.md), and does not repeat what those documents already resolved and test. New findings are numbered **AI-N01…AI-N12**.

**Environment.** Linux x86_64, Python 3.11.15, TensorFlow 2.21.0, Keras 3.15.1 (`requirements-lock.txt`), CPU only.

**Evidence files.**
- [`docs/audit/ai/classifier_evaluation.md`](audit/ai/classifier_evaluation.md) and `.json`. These are the generalization and calibration numbers, regenerated with `python experiments/report_classifier_evaluation.py`.
- [`docs/audit/validation/regression_tests_vs_prefix_code.txt`](audit/validation/regression_tests_vs_prefix_code.txt). It shows that each new regression test fails on the pre-fix code.
- Tests: `tests/ai/test_classifier_evaluation.py` and `tests/ai/test_monitoring_alerts.py`.

**Second pass.** §1–§6 describe the state found on 2026-10-18. The findings left open there were then fixed; §7 records each fix with its evidence, and the status markers of §2 point to it.

**No number in this report was invented.** Every figure comes from one of the commands named next to it, and every number is from **simulated** data. No real-machine data or hardware run exists in the repository.

---

## 1. What is actually implemented (inspected, not taken from the README)

| Model | Registry entry | Artifact(s) | Input → output | Recorded test metric |
|---|---|---|---|---|
| PC classifier (Conv1D+SE ×3) | `dc_motor/pc/classifier` → `pc_server/2026-09-01_run01` | `model.weights.h5` (Keras 3) | 512 × 6 (`acc_x, acc_y, acc_z, current_r, rpm, torque_nm`), per-window z-score → softmax over `[inner_race, normal, outer_race]` | acc 0.895 / macro-F1 0.894 on **38** windows (2 runs, **no inner_race**; Q15) |
| RPi5 classifier (ResNet1D-SE, distilled) | `dc_motor/rpi5/classifier` | `.weights.h5` + `model.tflite` (FP16 weights) | same | acc 0.868, same 38-window split |
| ESP32 classifier (DS-CNN, distilled) | `dc_motor/esp32/classifier` | `.weights.h5` + `model.tflite` (full INT8) | same | acc 0.868, same split |
| Candidate PC classifier (**not promoted**) | — `pc_server/2026-10-13_run01` | `.weights.h5` | same | acc 0.960 on 150 windows (3 runs, one per class) |
| PC / RPi5 / ESP32 forecasters | `vsc_dpc/{pc,rpi5,esp32}/regressor` | `.weights.h5` (+ `.tflite` for edge; the RPi5 GRU needs Flex ops) | 256 × 4 (`vc_real, vc_imag, i_f_real, i_f_imag`) → 64-step direct forecast | normalized RMSE 0.060 / 0.201 / 0.563 on 27 windows |
| DPC network | not in the registry; `configs/dpc_trained_v3_closed_loop.weights.h5` | `.weights.h5` | 15 inputs (state, 5-step reference, R) → 5 × (v_o real, imag) | closed-loop RMSE 1.18 V at R = 8.0064 Ω |
| Simulation-based anomaly agents | `agents/dc_motor_agent.py`, `agents/vsc_agent.py` | none (DTW against simulated hypotheses) | telemetry window → score in [0, 1] plus the closest hypothesis | no recorded metric |

**The GUI shows real model output.** `classify_last_window` runs the promoted model on the last window and labels the result *AI SUGGESTION — REQUIRES VERIFICATION*. When no model is promoted for a domain it returns `None`, and the UI says so; it never shows a placeholder value. This was checked in code (`viz/adapters/ai_adapter.py`, `viz/views/ai_diagnostics.py`), and the earlier T-UI tests cover it.

---

## 2. Findings

Each finding is classified as one of:
- **Defect (confirmed):** reproduced by a test or a measurement;
- **Risk (potential):** a plausible failure supported by evidence, not reproduced as a wrong output in a shipped flow;
- **Improvement (recommended):** not a defect.

### AI-N01 — Training/inference preprocessing mismatch in the GUI · **Medium** · Defect (confirmed) · **Fixed**

| Field | Content |
|---|---|
| Component | `viz/adapters/ai_adapter.normalize_window`, used by DC Motor → AI diagnosis, AI Diagnostics (classifier and regressor panels), and Monitoring's classifier-confidence drift. |
| Evidence | Training (`models/common/windowing.build_classification_windows`) **zeroes** a channel window whose σ ≤ 1e-3·\|μ\| (`is_live_window`). The GUI z-scored every window with σ > 1e-6. On fresh healthy runs at steady state, `current_r` and `torque_nm` were normalized differently in **19 of 19** windows: amplified numerical noise in the GUI, zeros in training. Probe, 2026-10-18. |
| Root cause | Two separately written normalization rules. |
| Impact | The model received inputs from a distribution it never saw. On the probed runs (57 windows, both PC classifiers) the predicted class did not change, so the measured impact was **0 changed predictions**. The mismatch is real all the same, and it grows with noise and drift. For the forecaster panel it also changed the de-normalization of dead channels. |
| Solution / fix | New `windowing.normalize_window_like_training()`, which applies exactly the training rule; dead channels get σ = 1 so that `y·σ + μ` maps back to the level. `ai_adapter.normalize_window` now delegates to it. |
| Validation | `tests/ai/test_classifier_evaluation.py::TestPreprocessingContract`: the GUI window equals the training tensor (atol 1e-5) on a real run, and the test checks that dead channels are present, so it is not vacuous. **Fails on the pre-fix code, passes after.** |

### AI-N02 — The rule's hysteresis runs on wall-clock time between clicks, not on telemetry time · **Medium** · Defect (confirmed) · **Fixed in the second pass (§7)**

| Field | Content |
|---|---|
| Component | `viz/adapters/monitoring_adapter.run_monitoring_pass` with `GatewayAgent`. |
| Evidence | The gateway is stepped **once per "Run monitoring" click** with `time.time()`. A 0.05 s VSC run at R = 2 Ω (unstable) never alerts on the first pass. It alerts only if the user clicks again at least 2 s later. The existing test `tests/test_live_monitoring_dashboard.py::TestRuleHysteresis` encodes this behavior. |
| Root cause | The hysteresis was designed for a continuous telemetry stream. The GUI has only discrete, user-paced simulated passes. |
| Impact | Detection latency depends on how fast the operator clicks, not on the process. A single pass on a dangerous condition shows "no rule alert". |
| Proposed solution | Step the gateway over the simulated run's own timeline. This needs runs at least as long as the hysteresis (2 s of VSC is about 30 s of computation), or a per-domain hysteresis decision. **Researcher decision**, so it is not changed here. Scenario D of `DEMO_VALIDATION_SCENARIOS.md` documents the two-click behavior. |
| Validation | Existing test (as above). |

### AI-N03 — The VSC alarm threshold did not match the physical instability region · **High** · Defect (confirmed) · **Fixed**

| Field | Content |
|---|---|
| Component | `monitoring/rules/vsc_dpc.yaml` (`dpc_load_resistance_divergence`). |
| Evidence | The condition was `1.0 ≤ R ≤ 3.0`. The plant's own open-loop spectral radius (`sim.vsc_system.load_feedback_spectral_radius`) exceeds 1 for **every R < R\* = 3.3707 Ω**: ρ ≈ 12 at R = 0.5 Ω, and ρ > 1 between 3.0 and 3.37 Ω. The rule stayed silent at R = 0.2, 0.5, 0.99, 3.2 and 3.36 Ω, all of which are divergent. |
| Root cause | A rounded "known divergent" range was used instead of the analytic boundary. |
| Impact | Missed detection of the most severe instability (R < 1 Ω, fastest divergence) and of the band just below R\*. |
| Fix | Condition changed to `load_resistance_ohm < 3.3707`. The GUI help text was updated to match. |
| Validation | `tests/ai/test_monitoring_alerts.py::TestVscRuleThresholdIsPhysical`: over 13 values of R, the rule fires **iff** ρ > 1. **5 cases fail on the pre-fix rule; all pass after.** |

### AI-N04 — Non-finite telemetry silently resolved active alerts · **High** · Defect (confirmed) · **Fixed**

| Field | Content |
|---|---|
| Component | `monitoring/agents/agent_gateway.GatewayAgent.step`. |
| Evidence | Every comparison with NaN is False. One NaN reading (a sensor dropout, or a diverged simulation, which the VSC at R < R\* produces) emitted `resolved=True`. `ServerAgent` then reported the domain **HEALTHY** while the condition persisted. |
| Root cause | No notion of an "indeterminate" sample. |
| Impact | Lost alarms exactly when the data path fails. |
| Fix | A tick whose referenced fields are non-finite is **indeterminate**: nothing fires and nothing clears; the hysteresis timer of a not-yet-active rule restarts; `indeterminate_rules()` exposes the data-quality problem. |
| Validation | `TestNonFiniteTelemetry` (3 tests). **They fail before the fix and pass after.** Existing gateway tests are unchanged and pass. |

### AI-N05 — The DC anomaly agent declared a feature that does not exist · **Medium** · Defect (confirmed) · **Fixed**

| Field | Content |
|---|---|
| Component | `agents/dc_motor_agent._DEFAULT_FEATURES`, `configs/agents.yaml`. |
| Evidence | The agent declared `"i_a"`, but the telemetry schema calls the armature current `current_r`. `detect_anomaly()` filters on `f in telemetry`, so the electrical channel was dropped **silently** in every monitoring pass. |
| Root cause | A naming mismatch with no schema check. |
| Impact | The agent ignored the MCSA (current) evidence it claimed to use. |
| Fix | The feature is renamed to `current_r` in the agent and in `configs/agents.yaml`. |
| Validation | `test_dc_agent_features_exist_in_the_telemetry_schema` and `test_dc_agent_score_is_finite_and_bounded_for_a_real_window`. **They fail before the fix (KeyError `i_a`) and pass after.** |

### AI-N06 — The promoted PC classifier never recognizes inner-race faults · **Critical** · Defect (confirmed, known since Q15) · **Fixed in the second pass (§7)**

| Field | Content |
|---|---|
| Component | `dc_motor/pc/classifier` = `pc_server/2026-09-01_run01`, and the two edge models distilled from it. |
| Evidence | `classifier_evaluation.md`, nominal steady state, 228 fresh windows: **recall inner_race = 0.00** (all 76 → outer_race), balanced accuracy 0.667, macro-F1 0.556, ECE 0.167, Brier 0.456. The same 0.00 appears under **every** shifted condition. |
| Root cause | Previous audit (Q15): the test split it was selected on contained no inner_race run. |
| Impact | One of the two fault classes the platform claims to diagnose is never detected by the model the GUI uses. |
| Proposed solution | Promote the candidate `2026-10-13_run01` through the gate (`experiments/promote_run.py`), then re-distil and re-export the edge models. Promotion changes the production model, so it is left to the researcher, as in Q15. |
| Validation | `test_promoted_classifier_recognizes_inner_race` is a **strict xfail**: it turns into a failure ("XPASS") the moment the promoted model is fixed, which forces the marker to be removed. |

### AI-N07 — Classifiers do not generalize across operating points; the candidate is confidently wrong · **High** · Defect (confirmed) · **Fixed in the second pass (§7)**

| Field | Content |
|---|---|
| Component | Both PC classifiers; training-data design (`experiments/generate_diagnosis_dataset.py`). |
| Evidence | Candidate `2026-10-13_run01`, fresh steady-state windows (`classifier_evaluation.md`): **1.000** accuracy at 150 rad/s, 200 rad/s, ×2 severity, 20 % noise and plant drift. But:<br>• **100 rad/s:** inner_race recall **0.00**, all predicted outer_race, with mean confidence **1.00** on the wrong windows (ECE 0.324);<br>• **half severity:** inner_race recall **0.00**, predicted as **normal**, so the missed-detection rate is **0.50**;<br>• **100 % sensor noise:** outer_race recall 0.00, false-alarm rate 0.32. |
| Root cause | **Hypothesis**, supported by the frequencies but not proven by an ablation. Training used one speed (150 rad/s), and per-window z-scoring removes the speed level, so the model cannot normalize fault frequencies by shaft speed. It appears to key on absolute spectral position. At 100 rad/s, BPFI = **78.5 Hz** lands next to the training-speed BPFO = **73.3 Hz**, and those windows are called outer_race. Separately, the dataset seed changes only the vibration channels (Q15; asserted in `test_seed_only_changes_vibration_channels`), so "new seeds" never vary the operating condition. |
| Impact | Accuracy measured at the training point says nothing about other speeds or severities. Confidence cannot warn of the error: the softmax is ≈1.0 when wrong. |
| Proposed solution | (1) Generate training data over a grid of speeds, loads and severities, and split by operating condition (leave-speed-out). (2) Give the model speed-invariant input: order-tracked / angle-resampled windows, or the absolute rpm as an extra non-z-scored feature. (3) Report per-condition metrics in `metrics.json` (now possible with `ai/evaluation.py`). (4) Add calibration (temperature scaling on validation) and an abstention threshold. |
| Validation | `test_candidate_generalizes_to_another_speed` (**strict xfail**, slow). Full table in `docs/audit/ai/classifier_evaluation.md`. |

### AI-N08 — Alert rules have an on-delay but no off-delay or deadband · **Low** · Risk (potential) · **Fixed in the second pass (§7)**

| Field | Content |
|---|---|
| Component | `GatewayAgent`. |
| Evidence | Two tests document the behavior. `test_chattering_condition_shorter_than_hysteresis_never_fires`: a condition toggling every 0.5 s with 2 s hysteresis produces **no** alert. `test_one_clean_sample_clears_an_active_alert`: a single clean sample resolves an active alert. |
| Impact | The alarm rate is bounded, so there is no alarm flooding (good). But an intermittent real fault that never holds for 2 s continuously is never reported, and a noisy signal at the threshold can clear an active alert right away. |
| Proposed solution | A separate `clear_hysteresis_seconds` and/or a numeric deadband per rule. This is a schema change, so it is left to the researcher. |

### AI-N09 — The anomaly score is confounded by the operating point · **Medium** · Risk (potential) · **Fixed in the second pass (§7)**

| Field | Content |
|---|---|
| Component | `agents/base.SimulationBasedAgent.detect_anomaly`. |
| Evidence | Hypotheses are always simulated at the default operating point (150 rad/s, seed 0). A **healthy** motor at 120 rad/s scored 0.573, and its closest hypothesis was "**moderate**" fault. The same healthy motor at 150 rad/s scored 0.32, closest "healthy". Probe, 2026-10-18. |
| Impact | Today's GUI always monitors at 150 rad/s, so it is not affected. Any other setpoint, or real telemetry, would make the "closest hypothesis" reflect the speed difference rather than the fault. |
| Proposed solution | Pass the observed operating point (setpoint, or measured mean speed and torque) into the hypothesis simulations and the cache key. |

### AI-N10 — The anomaly agent mixes unnormalized features and is CPU-heavy · **Medium** · Risk / improvement · **Fixed in the second pass (§7)**

| Field | Content |
|---|---|
| Evidence | Per-feature DTW distances on a 0.15 s outer_race window against the healthy hypothesis: rpm **3826.8**, acc_x 9.8, acc_y 7.4, acc_z 3.9, current_r 0.0 (current-limited start-up). The score is the mean of these raw distances, so speed in rpm dominates it, and the vibration channels, which carry the bearing signature, contribute less than 1 %. One `detect_anomaly` call takes **≈14–15 s**, even with the simulation cache warm, because the DTW is pure Python, O(n·m) per feature (5 features × 3 hypotheses × 1500² cells). |
| Impact | The score mostly reflects the speed trajectory. Each "Run monitoring" click blocks the UI for about 15 s. |
| Proposed solution | Normalize each feature (z-score with the healthy hypothesis's statistics) before DTW. Use a banded (Sakoe–Chiba) or vectorized/compiled DTW. Validate the score with labelled runs (ROC) before using the 0.70/0.85 thresholds, which remain uncalibrated (Q25). |

### AI-N11 — The forecaster test set is a near-duplicate of its training set · **Medium** · Risk (potential) · **Fixed in the second pass (§7.5)**

| Field | Content |
|---|---|
| Component | `vsc_dpc/*/regressor` and `data/vsc_dpc_dataset.parquet`. |
| Evidence | The 15 runs differ **only** in the reference start phase: the varying columns are timestamp, ids, seed and the signals. After the transient, run 0 and run 1 have a time-shifted correlation of **0.9999**. The run-grouped split therefore puts phase-shifted copies of training waveforms in the test set. There is no validation split and no naive (persistence) baseline in `metrics.json`. |
| Impact | The recorded test RMSE (0.060 / 0.201 / 0.563) measures interpolation of one periodic waveform. It gives no evidence of generalization to other loads, references or disturbances. |
| Proposed solution | Vary R, \|v_ref\| and ω across runs (within the stable region). Split by condition. Record a persistence baseline (the TL validator already computes one; `AUDIT_ANSWERS` Q11). |

### AI-N12 — Confidence is uncalibrated and there is no out-of-distribution abstention · **High** · Risk (confirmed numerically) · **Mitigated in the second pass (§7.2)**

| Field | Content |
|---|---|
| Evidence | The candidate classifies **ball** faults, a type with no class in the model, as outer_race with mean confidence **0.990** (100 % of windows above 0.9). It classifies cage as outer_race / normal with 0.81. The promoted model's mean confidence is about 0.5 on everything (ECE 0.167). |
| Impact | Confidence cannot flag unknown conditions or the AI-N07 errors. This was already labelled *uncalibrated* in the UI (Q14). |
| Proposed solution | Temperature scaling on the validation split, plus an abstention / "unknown" threshold checked against held-out fault types. `ai/evaluation.py` now gives ECE, Brier and OOD summaries to measure both. |

---

## 3. Dataset and evaluation review (prompt §2.2–2.3)

| Item | Status | Evidence |
|---|---|---|
| Class balance | Balanced by construction: one run per class per seed; the evaluation uses 76 windows per class. | `classifier_evaluation.md` |
| Train / val / test separation | By acquisition (run), per class; the train_model test set must contain every class. | Q08/Q15, `test_every_class_is_in_the_test_set` |
| Leakage through window overlap | None: non-overlapping windows inside one run, and runs never shared between splits. | Q05 |
| Leakage through near-duplicate runs | **Present.** Classifier: the seed changes only vibration (AI-N07). Forecaster: phase-shifted copies (AI-N11). | above |
| Normalization fitted on train only | Per-window statistics, so no cross-split leakage. Training and inference are now consistent (AI-N01). | Q06, AI-N01 |
| Overfitting / underfitting | Promoted model: underfit for inner_race (recall 0 even on training windows, Q15). Candidate: fits the training operating point, fails off it (AI-N07). | |
| Generalization to new conditions | **Measured for the first time**: speed, severity, noise, plant drift, start-up transient, unseen fault types. | `classifier_evaluation.md` |
| Simulated vs. real signals | All data is simulated. Only the vibration x axis is calibrated against real Paderborn PSDs, and that calibration does not generalize across bearings (Q16/Q29). | — |
| Independent validation on real data | **Not possible with the repository's data.** A real hold-out set is needed: other machines and bearings, at least two speeds and loads, labelled inner/outer faults (e.g. Paderborn KAt or CWRU), with a leave-machine-out split. | limitation |

**Metrics now available** (`src/driveflow/ai/evaluation.py`):
- accuracy, balanced accuracy, per-class precision/recall/F1/support, per-class one-vs-rest FPR and FNR, macro-F1, confusion matrix;
- fault-detection false-alarm and missed-detection rates;
- ECE, a reliability curve, Brier score, and confidence on correct vs. incorrect predictions;
- an out-of-distribution confidence summary.

Unit tests check them against hand-computed cases (`TestMetrics`).

## 4. Transfer learning, registry, TFLite (prompt §2.4)

Re-checked, not re-implemented. The previous audit's guarantees still hold, and their tests pass in the final run:
- the base model is not overwritten (SHA-256 fingerprint check, `TestAI04BaseModelImmutable`);
- base and candidate are evaluated on the same test windows (`TestAI07FairComparison`);
- promotion is fail-closed and `registry.promote()` re-checks persisted evidence (`TestAI06PromotionGate`);
- lineage, gate and split manifests are written per candidate;
- the registry is audited, with `rollback()`.

On TFLite, export success is **not** treated as hardware validation:
- The deployment state machine records only `exported` from disk.
- Host-CPU parity was measured earlier (`docs/audit/tflite_parity_report.txt`): edge classifiers agree with Keras top-1 1.000; the INT8 ESP32 regressor RMSE was 0.5629 → 0.5598.
- The RPi5 GRU needs Flex ops and is flagged.

**Not verified on any physical device:** no latency, RAM or flash measurements exist. The edge classifiers inherit AI-N06, because they were distilled from the promoted teacher. (Superseded: the edge models were re-distilled from the grid-trained teacher and re-exported; see §7.1 and §7.5.)

## 5. Monitoring summary (prompt §2.5)

**Detection, acknowledgement and resolution are separate and tested** (`TestDetectionAcknowledgementResolution`):
- acknowledgement does not change health;
- an acknowledgement does not carry over to a new occurrence of the alert;
- repeated reports update one active alert but are all logged, so no event is lost;
- two sources reporting the same rule stay independent.

Excessive alarming is bounded by the on-delay hysteresis. Detection gaps are AI-N02, AI-N03 (fixed), AI-N04 (fixed) and AI-N08.

## 6. Summary

| ID | Severity | Type | Status |
|---|---|---|---|
| AI-N06 | Critical | Defect | **Fixed + tested** (§7.1: grid-trained model promoted, edge models re-distilled) |
| AI-N03 | High | Defect | **Fixed + tested** |
| AI-N04 | High | Defect | **Fixed + tested** |
| AI-N07 | High | Defect | **Fixed + tested** on held-out simulated conditions (§7.1) |
| AI-N12 | High | Risk | **Mitigated** (§7.2): temperature scaling + abstention; does not catch every unseen fault type |
| AI-N01 | Medium | Defect | **Fixed + tested** |
| AI-N02 | Medium | Defect | **Fixed + tested** (§7.3) |
| AI-N05 | Medium | Defect | **Fixed + tested** |
| AI-N09 | Medium | Risk | **Fixed + tested** (§7.4) |
| AI-N10 | Medium | Risk / improvement | **Fixed + tested** (§7.4) |
| AI-N11 | Medium | Risk | **Fixed** (§7.5): condition-split data, persistence baseline recorded |
| AI-N08 | Low | Risk | **Fixed + tested** (§7.3) |
| AI-N13 | High | Negative result | **Open** (§8.1): a feature-space OOD detector does not catch ball/cage faults at a usable false-alarm rate |
| AI-N14 | Medium | Defect (generalization) | **Open** (§8.1): below the training speed range (50 rad/s) accuracy on answered windows is 0.58; at 150 % noise 0.68 |

---

## 7. Second pass: fixes of the open findings

### 7.1 AI-N06 and AI-N07 — classifier retrained on an operating-condition grid

**Data.** `experiments/generate_diagnosis_grid_dataset.py` → `data/diagnosis_grid_dataset.parquet` (not committed; regenerate with the command). It holds 315 runs of 2.5 s, 105 per class:
- speeds 80, 100, 125, 150, 175, 200, 220 rad/s;
- severity ×0.5, ×0.75, ×1, ×1.5, ×2 of the calibrated electrical and mechanical values;
- sensor noise: a third of the runs without noise, the rest 0–100 % drawn per run;
- plant drift ψ_e ×U(0.9, 1.1), J ×U(0.8, 1.5);
- seeds 1001+, disjoint from every evaluation seed.

A first version of the grid had no noise-free run and at most 40 % noise. Its model misclassified a third of the noise-free healthy windows (the GUI's default condition) and every healthy window at 100 % noise. The noise design was corrected before the model below was trained.

**Speed information.** Per-window z-scoring erased the speed level, which is what the model needs to tell BPFI at one speed from BPFO at another. Runs can now record `level_scaling` in `metrics.json`. For the promoted model, the rpm channel is scaled by a fixed (center 1432.4 rpm, scale 477.5 rpm) instead of z-scored, so its level survives. Every consumer applies the run's own rule: training, GUI, TL fine-tuning, TFLite export calibration, and the parity and evaluation reports. Runs without the key behave exactly as before; the old z-score path is bit-identical.

**Ablation (same grid, same seed, same epochs).**

| Run | rpm level | Test accuracy | Test macro-F1 | outer_race recall |
|---|---|---|---|---|
| `pc_server/2026-10-18_run01` (**promoted**) | kept | **0.914** | **0.914** | 0.89 |
| `pc_server/2026-10-18_run02` (ablation, not promoted) | z-scored | 0.838 | 0.834 | 0.68 |

The test split is by run, 30 of 315 runs, mixing every speed, severity and noise level. It is harder than the old single-point test, which is why 0.914 is below the old 0.960.

**Held-out conditions** (fresh runs, seeds 200–203, steady state; [`classifier_evaluation.md`](audit/ai/classifier_evaluation.md)). The new conditions are outside the training grid: 115 and 190 rad/s fall between grid speeds, 250 rad/s is beyond it, and ×0.6 severity falls between grid severities.

| Condition | 2026-09-01 (old production) | 2026-10-13 (single-point candidate) | **2026-10-18_run01 (promoted)** |
|---|---|---|---|
| nominal 150 rad/s | 0.667 (inner_race recall 0) | 1.000 | **1.000** |
| 100 rad/s | 0.667 | 0.667 (inner_race → outer_race) | **1.000** |
| unseen 115 rad/s | 0.667 | 0.667 | **1.000** |
| unseen 190 rad/s | 0.667 | 1.000 | **1.000** |
| extrapolated 250 rad/s | 0.667 | 0.333 | **1.000** |
| severity ×0.5 | 0.667 | 0.667 (missed-detection 0.50) | **1.000** |
| unseen severity ×0.6 | 0.667 | 1.000 | **1.000** |
| noise 100 % | 0.667 | 0.561 | **0.969** (false-alarm rate 0.092) |
| start-up transient (t < 0.5 s) | 0.500 | 0.778 | 0.815 |

**Edge models** were re-distilled from the promoted teacher on the same grid and re-exported. `level_scaling` is inherited from the teacher.

| Tier | Run | Test accuracy | TFLite parity (test runs, host CPU) |
|---|---|---|---|
| rpi5 | `rpi5_resnet1d_se/2026-10-18_run01` | 0.916 | FP16: accuracy 0.949 → 0.949, top-1 agreement 0.999 |
| esp32 | `esp32_dscnn/2026-10-18_run01` | 0.901 | INT8: accuracy 0.940 → 0.942, top-1 agreement 0.977 |

INT8 calibration fix found on the way: `export_tflite.py` calibrated on the first 100 windows of the builder output. That meant one class, and only start-up windows because of the 500-window cap, and it gave top-1 agreement 0.71. It now draws a seeded random subset of 1000 windows from all windows. Evidence: [`tflite_parity_report.txt`](audit/tflite_parity_report.txt).

**Tests.**
- `test_promoted_classifier_recognizes_every_class` and `test_promoted_classifier_generalizes_to_held_out_conditions[speed_115|speed_250|severity_x0.6]` are no longer xfail.
- `test_previous_production_model_still_shows_the_ai_n06_defect` and `test_single_speed_model_confuses_inner_race_at_100_rad_s` keep the original evidence reproducible.
- `TestLevelScalingContract` checks that GUI and training preprocessing are identical with `level_scaling`.

**Limits.** Every number is from the simulator. The model still has no class for ball or cage faults (§7.2), and no real-machine validation exists.

### 7.2 AI-N12 — calibration and abstention

`src/driveflow/ai/calibration.py` fits two things on the **validation** split at training time, and stores them in `metrics.json["calibration"]`:
- temperature scaling (NLL);
- an abstention threshold that keeps 95 % of the correct validation windows.

The GUI shows "uncertain" instead of a class below the threshold, and labels the confidence as calibrated. The promoted model has T = 1.005 (it was already close to calibrated on in-distribution data) and abstains below 0.594.

What it does and does not achieve, measured:
- In-distribution ECE stays 0.003–0.06.
- At 100 % noise, 16 % of windows are withheld; accuracy on the answered ones is 0.990, and 71 % of the errors are withheld.
- **Cage** faults (no class): 54 % withheld, the rest called normal.
- **Ball** faults (no class): called outer_race with confidence 0.993, **0 % withheld**.

Calibration fitted in-distribution does not make unseen fault types safe. A ball-fault class, or a dedicated out-of-distribution detector, is still needed. Tests: `TestCalibration`.

### 7.3 AI-N02 and AI-N08 — rule timing

- **Telemetry time (AI-N02).** The gateway is now stepped on every sample of the simulated run, on the run's own timeline, offset per domain so the clock keeps increasing across clicks (`monitoring_adapter.step_gateway_over_run`). Detection no longer depends on how long the operator waits.
- **Off-delay (AI-N08).** `Rule.clear_hysteresis_seconds` is an optional off-delay, default 0, so existing YAML is unchanged. `GatewayAgent` resolves an active alert only after the condition has been false that long.
- **VSC rule.** `hysteresis_seconds` 2 → 0.01 and `clear_hysteresis_seconds` 0.01. Justification, in the YAML: below R\* the plant diverges in milliseconds (spectral radius 1.25 per 0.1 ms step at 3 Ω), and R is a configured or slowly estimated parameter, not a noisy fast signal. 10 ms (100 samples) rejects isolated glitches, and fires within one 0.05 s dashboard run.
- **Tests:**
  - `TestOffDelay`;
  - `TestShippedVscRuleTiming`;
  - `test_fires_on_the_first_pass_and_resolves_on_a_stable_pass` (AppTest; it replaces the old two-click / 2.2 s sleep test).

### 7.4 AI-N09 and AI-N10 — simulation-based anomaly agent

Measured on the way: once rpm stopped dominating, the time-domain DTW score was ≈0.50 for healthy and faulty runs alike. Two vibration records are independent noise realizations, so a sample-by-sample alignment measures the noise, not the bearing.

Changes (`agents/base.py`, `agents/dc_motor_agent.py`, `agents/detector.py`):
- **Operating point (AI-N09).** Hypotheses are simulated at the observed operating point: the DC agent estimates the speed setpoint from the rpm tail, or takes it from the caller. It is part of the cache key.
- **Normalization (AI-N10).** Features are z-scored with the healthy hypothesis's statistics.
- **Spectral comparison (AI-N10).** Vibration and current are compared by log-PSD (Welch) over the second half of the window. rpm stays a time-domain DTW, decimated ×10.
- **Exact vectorized DTW.** Each row is a running-minimum recurrence, ≈30× faster, and equal to the reference implementation in tests.
- **Inner-race hypotheses.** With outer-race hypotheses only, inner_race runs scored 0.64–0.71.
- **Longer GUI window.** The DC monitoring window is now 0.6 s instead of 0.15 s. At 150 rad/s the motor is still accelerating at 0.15 s, and no method saw the fault there.

Labelled evaluation (`experiments/report_anomaly_agent_roc.py` → [`anomaly_agent_roc.json`](audit/ai/anomaly_agent_roc.json)): 60 fresh runs at 100/130/150/180 rad/s; healthy, plus outer and inner race at two severities each; seeds 31–33.

| Metric | Before (2026-10-18 audit) | After |
|---|---|---|
| ROC AUC (healthy vs. any fault) | not measured (healthy at 120 rad/s scored 0.573, closest "moderate") | **1.000** |
| False-alarm / missed-detection rate at 0.70 | — | **0.000 / 0.000** |
| Missed-detection rate at 0.85 (high severity) | — | 0.25 (inner_race ×0.5 scores 0.81–0.84) |
| Closest hypothesis = true fault type and severity | — | 60 / 60 |
| Time per `detect_anomaly` call, cache warm | ≈14–15 s | **0.057 s** |

The 0.70/0.85 thresholds now have a measured operating point on simulated data. Real telemetry would need the same check.

### 7.5 AI-N11 — forecasters retrained on distinct operating conditions

**Data.** `experiments/generate_vsc_dpc_grid_dataset.py` → `data/vsc_dpc_grid_dataset.parquet`: 36 runs of 0.3 s. Every run is a different condition, load resistance R ∈ {4, 5, 6, 7, 8.0064, 10, 13, 16, 20} Ω × |v_ref| ∈ {30, 50, 75, 100} V, simulated with the v4 DPC controller (S-05). The run-grouped split of `train_model.py` is therefore a split by condition. The 8 test conditions are listed in `metrics.json["test_groups"]`.

**Baseline.** `train_model.py` now records `persistence_test_rmse` (repeat the last context sample) next to `test_rmse`, in the same normalized space.

**Promoted runs (2026-10-19).**
- `pc_full`: trained directly.
- `rpi5_edge` and `esp32_tiny`: distilled from the PC run, 60 epochs.
- TFLite re-exported.

Results on the held-out conditions ([`forecaster_condition_split.json`](audit/ai/forecaster_condition_split.json), 72 windows, normalized RMSE):

| Run | RMSE on held-out conditions | Persistence | RMSE recorded by the run itself |
|---|---|---|---|
| `pc_full/2026-10-19_run01` (**promoted**) | **0.073** | 1.056 | 0.073 |
| `rpi5_edge/2026-10-19_run01` (**promoted**) | **0.063** | 1.056 | 0.063 |
| `esp32_tiny/2026-10-19_run01` (**promoted**) | **0.178** | 1.056 | 0.178 |
| `pc_full/2026-09-01_run01` (previous) | 0.149 | 1.056 | 0.060 |
| `rpi5_edge/2026-09-13_run01` (previous) | 0.213 | 1.056 | 0.201 |
| `esp32_tiny/2026-09-13_run02` (previous) | 0.627 | 1.056 | 0.563 |

**Reading the table:**
- The previous PC forecaster's recorded 0.060 came from a near-duplicate test set. On unseen conditions it measures 0.149, 2.5× worse.
- The comparison also mixes in the controller change: the previous runs were trained on v3-controlled data, the new grid uses v4.

**TFLite parity** ([`tflite_parity_report.txt`](audit/tflite_parity_report.txt)):
- ESP32 INT8: RMSE 0.178 → 0.184.
- The RPi5 GRU still needs Flex (SELECT_TF_OPS) kernels (Q22), so it cannot run on a builtin-only interpreter. An unrolled GRU or a TCN would be needed for TFLite Micro-class targets.

## 8. Third pass (2026-10-20, TODO_CLAUDE_CODE.md)

### 8.1 A1 — out-of-distribution detector for unseen fault types

**Setup.** `ai/ood.py` implements two pre-specified scores (TODO A1):
- the **Mahalanobis distance** of the classifier's features to the nearest class mean, with one shared covariance;
- the **energy score** of the logits.

**Anti-leakage rule.** The statistics come from the promoted run's own training windows and the thresholds from its validation windows. Both were rebuilt from the recorded dataset, and the split's group hashes were checked against `metrics.json`. Each threshold keeps 95 % of in-distribution validation windows, the same coverage as the abstention threshold. Ball and cage windows were only scored. The two pre-listed feature layers, `dense1` (the head's input) and `gap`, were both evaluated and both are reported. Nothing was tuned on the OOD results.

**Command and evidence.**
- `python experiments/report_ood_detector.py --cache data/ood_eval_cache.parquet [--layer gap]`
- [`ood_detector.md`](audit/ai/ood_detector.md) (`dense1`) and [`ood_detector_gap.md`](audit/ai/ood_detector_gap.md).
- The evaluation set was freshly simulated: ball and cage at 80, 125, 175 and 220 rad/s × severity ×0.5, ×1, ×2, with 2 seeds; benign shifts at 50 rad/s, 300 rad/s and 150 % noise.

| | Mahalanobis (dense1) | Mahalanobis (gap) | Energy | Target |
|---|---|---|---|---|
| Ball windows flagged unknown | 0.134 | 0.093 | 0.009 | ≥ 0.90 |
| Ball AUROC vs. in-distribution test | 0.898 | 0.866 | 0.623 | — |
| Cage: unknown or called a fault | 0.658 | 0.770 | 0.682 | ≥ 0.90 |
| Cage AUROC | 0.909 | 0.914 | 0.816 | — |
| In-distribution false-unknown rate (test) | 0.037 | 0.045 | 0.049 | ≈ 0.05 by construction |
| In-distribution accuracy on answered | 0.952 (0.954 without) | 0.952 | 0.955 | drop ≤ 1 pt |
| 300 rad/s (known classes) flagged unknown | 1.000 | 1.000 | 0.000 | low |

**AI-N13 (open, negative result).** No method meets the targets.
- **What does separate.** Mahalanobis separates ball and cage from in-distribution windows with AUROC ≈ 0.9. But at the 95 %-coverage threshold it flags only 9–13 % of ball windows.
- **What a usable threshold would cost.** Lowering the threshold enough to catch most ball windows would flag far more healthy windows; the AUROC bounds what any threshold can do.
- **False alarms where the classifier is right.** It flags every window at 300 rad/s, where the classifier is in fact correct.
- **The energy score** barely reacts to ball faults: the classifier's logits for a ball window look like a confident outer-race window.
- **Not wired into the model.** The detector is wired into `decide()` and the GUI (`Unknown condition — REQUIRES VERIFICATION`). It is **not** written into the promoted run's `metrics.json`, so the GUI's behavior is unchanged.
- **Next step.** Per TODO A1, the decision passes to A2 (adding ball/cage classes, with synthetic-only amplitudes) or to the physics cross-check with the simulation-based agent. The agent check was listed as optional and was not implemented.

**AI-N14 (open).** The benign-shift sets expose a generalization gap that the earlier held-out evaluation (115–250 rad/s) did not cover:
- **50 rad/s**, below the training grid's 80 rad/s floor: accuracy on answered windows is **0.58**;
- **150 % sensor noise**: accuracy on answered windows is **0.68**.

The earlier evaluation measured 1.000 at 115, 190 and 250 rad/s. Neither the abstention threshold nor the OOD detector catches these failures well: at 50 rad/s, 9 % of windows are flagged by Mahalanobis and 33 % by the energy score.

### 8.2 A3 — RPi5 forecaster without Flex ops

**The run.** `configs/regressors/rpi5_tcn.yaml` is a causal dilated TCN: 7 layers × 32 filters, kernel 3, receptive field 255 steps. It is built by the existing `recurrent_type: none` path. It was trained by distillation from the promoted `vsc_dpc/pc/regressor` with the same data, epochs and seed as `rpi5_edge/2026-10-19_run01`, and exported to float16 with `python -m experiments.export_tflite ... --run-dir` (a new option for exporting a run that is not promoted).

| | `rpi5_edge/2026-10-19_run01` (GRU, promoted) | `rpi5_tcn/2026-10-20_run01` (candidate) | Criterion |
|---|---|---|---|
| Flex ops | FlexTensorListReserve / SetItem / Stack | **none** (13 builtin ops) | 0 |
| RMSE on held-out conditions | 0.0634 | **0.0642** (×1.01) | ≤ 1.1 × GRU |
| Persistence baseline | 1.056 | 1.056 | better |
| TFLite vs. Keras (test RMSE) | not runnable on a builtin interpreter | 0.0642 → 0.0643, max \|diff\| 0.0024 | parity |
| File size | 38 KiB | 78 KiB | — |

Evidence:
- [`forecaster_condition_split.json`](audit/ai/forecaster_condition_split.json);
- [`tflite_parity_report.txt`](audit/tflite_parity_report.txt), produced with `python -m experiments.report_tflite_parity --candidate configs/regressors/rpi5_tcn/2026-10-20_run01`.

All A3 criteria are met. **Promoted on 2026-10-20** (confirmed by the user) as `vsc_dpc/rpi5/regressor`, replacing the GRU `rpi5_edge/2026-10-19_run01`; `configs/registry_history.yaml` records the change, and `rollback()` restores the GRU. The TFLite numbers are host-CPU measurements, not hardware.

### 8.3 A4 — VSC capacitor degradation, separability (Step 0 of D.2)

Evidence: [`vsc_capacitor_separability.md`](audit/ai/vsc_capacitor_separability.md).

**What was tested.** `sim/vsc_faults.py` makes the capacitor degradable, using the same exact ZOH as V0. With a pre-registered rule and cut (AUC ≥ 0.70 at C = 80 %), a logistic classifier grouped by operating point separates healthy from degraded windows in the DPC v4 closed loop:

| Level | AUC |
|---|---|
| C = 90 % | 1.000 |
| C = 80 % | 1.000 |
| C = 70 % | 1.000 |
| C = 80 %, with 5 % measurement noise (post-hoc) | 0.975 |
| C = 90 %, with 5 % measurement noise (post-hoc) | 0.888 |
| ESR of 0.05–0.2 Ω | 0.51–0.57 (not separable) |

**Verdict: proceed with capacitance; not with ESR.** The result holds only for the synthetic fault definition: no real dataset exists for this fault. Uncertainty in R, L and R_f was not tested. The classifier is proposed as a new task and was not built.

### 8.4 A2 — ball and cage classes (candidates, not promoted)

> **Synthetic-only, uncalibrated amplitude.** The ball and cage faults come from the simulator. Their mechanical amplitude (`CALIBRATED_MECHANICAL_SEVERITY` in `datagen/scenario.py`) is a placeholder: KAt has no clean single-defect code for these faults, so it cannot be calibrated without real data. Every run below records `"synthetic_only_classes": ["ball", "cage"]` in its `metrics.json` and carries the finding in `audit_findings.json`. When it answers ball or cage, the GUI shows "ball/cage: synthetic-only, uncalibrated amplitude".

**What was done.**
1. **Dataset.** The grid generator gained `--faults`. The 5-class grid `data/diagnosis_grid_dataset_5class.parquet` has 525 runs: 7 speeds × 5 severities × 3 seeds × 5 classes. It is not committed.
2. **PC model.** `pc_server_5class` was retrained with the promoted model's recipe: rpm level kept, 3000 windows per class, 30 epochs.
3. **RPi5 and ESP32 models.** They were distilled from that PC model. `train_model.py --teacher-run` is new and lets an unpromoted run act as teacher.
4. **Calibration.** Temperature and abstention were refitted on each model's validation split.

| Run | Recorded test acc. | Nominal 150 rad/s | 100 / 115 rad/s | 190 / 250 rad/s | 100 % noise | Start-up |
|---|---|---|---|---|---|---|
| `pc_server_5class/2026-10-20_run01` | 0.821 | **1.000** (all 5 classes) | 0.803 / 0.811 | 1.000 / 1.000 | 1.000 | 0.661 |
| `rpi5_resnet1d_se_5class/2026-10-20_run01` | 0.809 | 1.000 | 0.829 / 0.832 | 1.000 / 1.000 | 1.000 | 0.644 |
| `esp32_dscnn_5class/2026-10-20_run01` | 0.741 | 0.932 | 0.787 / 0.789 | 0.984 / 1.000 | 0.811 | 0.622 |
| promoted 3-class `pc_server/2026-10-18_run01`, for reference (its 3 classes only) | 0.914 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.969 | 0.815 |

Evidence: [`classifier_evaluation_5class.md`](audit/ai/classifier_evaluation_5class.md), from `python experiments/report_classifier_evaluation.py --extra-faults ball,cage …`. The held-out conditions are those of §7.1, simulated with all 5 classes.

**Limits found.**
- **Cage below nominal speed.** The models barely recognize cage faults at 100 and 115 rad/s. Cage recall is 0.01 / 0.05 for the PC model, 0.14 / 0.16 for RPi5 and 0.00 for ESP32. Those windows are called **ball**, so they are still flagged as a fault, and the missed-detection rate stays 0. At 150 rad/s and above, cage is recognized.
- **ESP32 INT8 export.** It loses accuracy: top-1 agreement with Keras is 0.73–0.76 and test accuracy drops from 0.785 to 0.68. With the 3-class model the agreement was 0.977. One declared retry with 3000 calibration windows instead of 1000 did not help ([`tflite_parity_report_5class.txt`](audit/tflite_parity_report_5class.txt)). The RPi5 float16 export matches Keras (agreement 1.000).
- **Recorded test accuracy.** It is lower than the 3-class model's (0.82 against 0.91). Most confusion is between the slower-rate faults during transients.

**Unseen faults (A1 rerun).** With ball and cage now classes, A1 needs a new set of unknown faults. Two custom bearing faults were used, at characteristic orders 1.3 and 6.5 × shaft speed; all built-in faults lie elsewhere (outer 3.07, inner 4.93, ball 2.04, cage 0.38). This is the DC Motor page's "custom fault type" mechanism. Evidence: [`ood_detector_custom_orders.md`](audit/ai/ood_detector_custom_orders.md).

| | Mahalanobis (dense1) | Energy |
|---|---|---|
| Order-6.5 fault flagged unknown | **0.970** (AUROC 0.992) | 0.000 |
| Order-1.3 fault flagged unknown | 0.298 (AUROC 0.897) | 0.107 |
| In-distribution false-unknown rate | 0.050 | 0.060 |
| Accuracy at 50 rad/s on answered windows (healthy / outer / inner) | 0.886 | 0.909 |

**Reading.**
- **High-order fault.** A fault whose impulse rate sits far above every class is caught.
- **Low-order fault.** One between cage (0.38) and ball (2.04) is mostly not caught.
- **Status.** The OOD detector stays off. At 50 rad/s the 5-class model answers healthy / outer / inner correctly in 0.89 of the windows, against 0.58 for the 3-class model (AI-N14).

**Status.** The candidates are ready and tested (`tests/ai/test_classifier_5class.py`). **Not promoted.** Promoting them replaces the 3-class models in the GUI and needs confirmation:

```bash
python experiments/promote_run.py configs/classifiers/pc_server_5class/2026-10-20_run01
```

