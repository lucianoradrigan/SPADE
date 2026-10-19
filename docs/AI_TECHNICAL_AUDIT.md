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

### AI-N02 — The rule's hysteresis runs on wall-clock time between clicks, not on telemetry time · **Medium** · Defect (confirmed) · **Open**

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

### AI-N06 — The promoted PC classifier never recognizes inner-race faults · **Critical** · Defect (confirmed, known since Q15) · **Open: researcher decision**

| Field | Content |
|---|---|
| Component | `dc_motor/pc/classifier` = `pc_server/2026-09-01_run01`, and the two edge models distilled from it. |
| Evidence | `classifier_evaluation.md`, nominal steady state, 228 fresh windows: **recall inner_race = 0.00** (all 76 → outer_race), balanced accuracy 0.667, macro-F1 0.556, ECE 0.167, Brier 0.456. The same 0.00 appears under **every** shifted condition. |
| Root cause | Previous audit (Q15): the test split it was selected on contained no inner_race run. |
| Impact | One of the two fault classes the platform claims to diagnose is never detected by the model the GUI uses. |
| Proposed solution | Promote the candidate `2026-10-13_run01` through the gate (`experiments/promote_run.py`), then re-distil and re-export the edge models. Promotion changes the production model, so it is left to the researcher, as in Q15. |
| Validation | `test_promoted_classifier_recognizes_inner_race` is a **strict xfail**: it turns into a failure ("XPASS") the moment the promoted model is fixed, which forces the marker to be removed. |

### AI-N07 — Classifiers do not generalize across operating points; the candidate is confidently wrong · **High** · Defect (confirmed) · **Open**

| Field | Content |
|---|---|
| Component | Both PC classifiers; training-data design (`experiments/generate_diagnosis_dataset.py`). |
| Evidence | Candidate `2026-10-13_run01`, fresh steady-state windows (`classifier_evaluation.md`): **1.000** accuracy at 150 rad/s, 200 rad/s, ×2 severity, 20 % noise and plant drift. But:<br>• **100 rad/s:** inner_race recall **0.00**, all predicted outer_race, with mean confidence **1.00** on the wrong windows (ECE 0.324);<br>• **half severity:** inner_race recall **0.00**, predicted as **normal**, so the missed-detection rate is **0.50**;<br>• **100 % sensor noise:** outer_race recall 0.00, false-alarm rate 0.32. |
| Root cause | **Hypothesis**, supported by the frequencies but not proven by an ablation. Training used one speed (150 rad/s), and per-window z-scoring removes the speed level, so the model cannot normalize fault frequencies by shaft speed. It appears to key on absolute spectral position. At 100 rad/s, BPFI = **78.5 Hz** lands next to the training-speed BPFO = **73.3 Hz**, and those windows are called outer_race. Separately, the dataset seed changes only the vibration channels (Q15; asserted in `test_seed_only_changes_vibration_channels`), so "new seeds" never vary the operating condition. |
| Impact | Accuracy measured at the training point says nothing about other speeds or severities. Confidence cannot warn of the error: the softmax is ≈1.0 when wrong. |
| Proposed solution | (1) Generate training data over a grid of speeds, loads and severities, and split by operating condition (leave-speed-out). (2) Give the model speed-invariant input: order-tracked / angle-resampled windows, or the absolute rpm as an extra non-z-scored feature. (3) Report per-condition metrics in `metrics.json` (now possible with `ai/evaluation.py`). (4) Add calibration (temperature scaling on validation) and an abstention threshold. |
| Validation | `test_candidate_generalizes_to_another_speed` (**strict xfail**, slow). Full table in `docs/audit/ai/classifier_evaluation.md`. |

### AI-N08 — Alert rules have an on-delay but no off-delay or deadband · **Low** · Risk (potential) · **Open**

| Field | Content |
|---|---|
| Component | `GatewayAgent`. |
| Evidence | Two tests document the behavior. `test_chattering_condition_shorter_than_hysteresis_never_fires`: a condition toggling every 0.5 s with 2 s hysteresis produces **no** alert. `test_one_clean_sample_clears_an_active_alert`: a single clean sample resolves an active alert. |
| Impact | The alarm rate is bounded, so there is no alarm flooding (good). But an intermittent real fault that never holds for 2 s continuously is never reported, and a noisy signal at the threshold can clear an active alert right away. |
| Proposed solution | A separate `clear_hysteresis_seconds` and/or a numeric deadband per rule. This is a schema change, so it is left to the researcher. |

### AI-N09 — The anomaly score is confounded by the operating point · **Medium** · Risk (potential) · **Open**

| Field | Content |
|---|---|
| Component | `agents/base.SimulationBasedAgent.detect_anomaly`. |
| Evidence | Hypotheses are always simulated at the default operating point (150 rad/s, seed 0). A **healthy** motor at 120 rad/s scored 0.573, and its closest hypothesis was "**moderate**" fault. The same healthy motor at 150 rad/s scored 0.32, closest "healthy". Probe, 2026-10-18. |
| Impact | Today's GUI always monitors at 150 rad/s, so it is not affected. Any other setpoint, or real telemetry, would make the "closest hypothesis" reflect the speed difference rather than the fault. |
| Proposed solution | Pass the observed operating point (setpoint, or measured mean speed and torque) into the hypothesis simulations and the cache key. |

### AI-N10 — The anomaly agent mixes unnormalized features and is CPU-heavy · **Medium** · Risk / improvement · **Open**

| Field | Content |
|---|---|
| Evidence | Per-feature DTW distances on a 0.15 s outer_race window against the healthy hypothesis: rpm **3826.8**, acc_x 9.8, acc_y 7.4, acc_z 3.9, current_r 0.0 (current-limited start-up). The score is the mean of these raw distances, so speed in rpm dominates it, and the vibration channels, which carry the bearing signature, contribute less than 1 %. One `detect_anomaly` call takes **≈14–15 s**, even with the simulation cache warm, because the DTW is pure Python, O(n·m) per feature (5 features × 3 hypotheses × 1500² cells). |
| Impact | The score mostly reflects the speed trajectory. Each "Run monitoring" click blocks the UI for about 15 s. |
| Proposed solution | Normalize each feature (z-score with the healthy hypothesis's statistics) before DTW. Use a banded (Sakoe–Chiba) or vectorized/compiled DTW. Validate the score with labelled runs (ROC) before using the 0.70/0.85 thresholds, which remain uncalibrated (Q25). |

### AI-N11 — The forecaster test set is a near-duplicate of its training set · **Medium** · Risk (potential) · **Open**

| Field | Content |
|---|---|
| Component | `vsc_dpc/*/regressor` and `data/vsc_dpc_dataset.parquet`. |
| Evidence | The 15 runs differ **only** in the reference start phase: the varying columns are timestamp, ids, seed and the signals. After the transient, run 0 and run 1 have a time-shifted correlation of **0.9999**. The run-grouped split therefore puts phase-shifted copies of training waveforms in the test set. There is no validation split and no naive (persistence) baseline in `metrics.json`. |
| Impact | The recorded test RMSE (0.060 / 0.201 / 0.563) measures interpolation of one periodic waveform. It gives no evidence of generalization to other loads, references or disturbances. |
| Proposed solution | Vary R, \|v_ref\| and ω across runs (within the stable region). Split by condition. Record a persistence baseline (the TL validator already computes one; `AUDIT_ANSWERS` Q11). |

### AI-N12 — Confidence is uncalibrated and there is no out-of-distribution abstention · **High** · Risk (confirmed numerically) · **Open**, partly mitigated

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

**Not verified on any physical device:** no latency, RAM or flash measurements exist. The edge classifiers inherit AI-N06, because they were distilled from the promoted teacher.

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
| AI-N06 | Critical | Defect | Open (promote candidate: researcher) |
| AI-N03 | High | Defect | **Fixed + tested** |
| AI-N04 | High | Defect | **Fixed + tested** |
| AI-N07 | High | Defect | Open (needs data design) |
| AI-N12 | High | Risk | Open (UI already labels confidence as uncalibrated) |
| AI-N01 | Medium | Defect | **Fixed + tested** |
| AI-N02 | Medium | Defect | Open (design decision) |
| AI-N05 | Medium | Defect | **Fixed + tested** |
| AI-N09 | Medium | Risk | Open |
| AI-N10 | Medium | Risk / improvement | Open |
| AI-N11 | Medium | Risk | Open |
| AI-N08 | Low | Risk | Open |
