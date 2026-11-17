# Claims audit (B2, 2026-11-16)

**All SPADE results are simulated.** 24 numeric claims from the README, `docs/VALIDATION_SUMMARY.md`, the v0.2.1 release notes and the GUI, chosen first among those the GUI or the release notes quote. Documents were **not** corrected (B2); a false statement in the GUI became a finding.

Result: **13 reproduced**, 9 evidence only, 1 stale, **1 unsupported** (→ AI-N43). Raw: `claims_audit_2026-11-16.json`. Scripts: `/mnt/datos/spade/b2/` (outside git).

Classes: reproduced = re-run here (< 10 min) and matched; evidence_only = an evidence file backs it, not re-run; stale = no longer true of the current code; unsupported = no evidence backs it as stated.

| # | Where | Claim | Status | Result (command in the JSON) |
|---|---|---|---|---|
| C01 | README §Diagnosis classifier (l.103) | PC model trained on a grid of 7 speeds × 5 severities with 2048-sample (205 ms) windows | **reproduced** | 7 speeds (80–220 rad/s), 5 severity scales; 2048 / 10 kHz = 204.8 ms |
| C02 | README l.289; GUI Monitoring help; GUI system diagram | VSC divergence below R* ≈ 3.37 Ω | **reproduced** | R* = 3.3707 Ω |
| C03 | README l.292; GUI Monitoring help | the vsc_dpc rule fires after a 10 ms on-delay and clears after 10 ms | **reproduced** | hysteresis_seconds 0.01, clear_hysteresis_seconds 0.01 |
| C04 | GUI Overview → system diagram (Classifier) | promoted PC classifier: test accuracy 93.0 %, macro-F1 0.93 on 600 test windows | **reproduced** | accuracy 0.930 on 600 windows (recorded 0.930) |
| C05 | GUI system diagram (Raspberry Pi 5) | RPi5 classifier 92.5 % (teacher 93.0 %, same 5 classes and test split) | **reproduced** | accuracy 0.925 on 600 windows (recorded 0.925) |
| C06 | GUI system diagram (Regressor) | vsc_dpc PC forecaster: normalized test RMSE 0.073 (persistence baseline 1.06) | **evidence_only** | test_rmse 0.0730 in metrics.json; the 1.056 baseline is in the audit, not in metrics.json |
| C07 | GUI system diagram (RPi5, ESP32) | vsc_dpc regressors: RPi5 RMSE 0.064, ESP32 RMSE 0.18 | **evidence_only** | 0.0642 and 0.1784 |
| C08 | Release notes v0.2.1, G6 | AI Diagnostics defaults: healthy → normal at 80, 300 and 350 rad/s (0.999 / 0.955 / 0.909); each fault at 300 rad/s detected | **reproduced** | 0.999 / 0.955 / 0.909; outer 0.95, inner 1.00, ball 1.00, cage 1.00 — Covers only the promoted PC model: in v0.2.1 the GUI could not select another one (AI-N42) |
| C09 | GUI notice AI-N40 (ABOVE_GRID); AI audit §13.13 | at 350 rad/s a healthy motor was read as 'cage' (RPi5 model, 1.00) | **reproduced** | cage 1.000 (known condition) |
| C10 | AI audit §13.13 | shipped RPi5: outer_race at 300 rad/s read as 'ball' 0.928 | **reproduced** | ball 0.928 |
| C11 | Release notes v0.2.1 (AI-N29); AI audit §13.3 | the shipped PC is confidently wrong at 50 rad/s: 54 windows, 44 % abstained, 0.067 accuracy on answered | **reproduced** | the whole 'eval' block is identical to the evidence |
| C12 | AI audit §13.3 | at 300 rad/s the shipped PC answers correctly (1.000 on answered) | **reproduced** | 1.000 |
| C13 | GUI ESP32 low-speed notice (Edge Deployment, Model Registry, AI Diagnostics); in v0.2.1 | the shipped ESP32 int8 misses most inner-race faults at 50–70 rad/s (int8 recall 0.35 where its float model gets 1.00) | **unsupported** | 0.35 / 1.00 are the never-shipped B4 candidate's numbers; the shipped int8 gets inner_race 1.00 and outer_race 0.28 (float 0.22) — False statement in the GUI → finding AI-N43, corrected on dev (54a7606) |
| C14 | README l.263; VALIDATION_SUMMARY §2 (Monitoring) | the DC anomaly score separates healthy from faulty with ROC AUC 1.0 on 60 simulated runs; 0.057 s per call | **reproduced** | AUC 1.0; false alarm / missed at 0.70: 0 / 0; missed at 0.85: 0.25; 0.058 s per call (machine busy with G1) |
| C15 | VALIDATION_SUMMARY §14.7 (B4c) | B4c: PC and RPi5 candidates macro-F1 0.994, min recall 0.970, selective risk 0.004 (promoted PC 0.275, RPi5 0.024) | **reproduced** | JSON identical except evaluation CPU time (23.1 s vs 23.2 s) |
| C16 | VALIDATION_SUMMARY §2 (PMSM / FOC) | S-01 fixed: default Euler with 3 substeps, stable up to ≈460 rad/s | **stale** | EULER_NSTEPS = 10, the default since 2026-10-20 (S-13), as the same row's last column says — The first column describes the 2026-10-19 state; not corrected here (B2 does not edit documents) |
| C17 | VALIDATION_SUMMARY §2 (VSC + DPC); README l.625–632 | scheduled DPC: 1.18 V at 8 Ω; v4 within 3.1 V from 4 Ω to 1 kΩ | **evidence_only** | the slow test runs in G1; it checks thresholds, not the quoted numbers |
| C18 | Release notes v0.2.1 (AI-N26) | the ESP32 classifier takes 68.5 ms for a 51.2 ms window on the C2000 | **evidence_only** | 51.2 ms = 512 samples / 10 kHz (reproduced); 68.5 ms needs the board |
| C19 | Release notes v0.2.1 (AI-N39); G5 | a real turn in Chrome gets its reply in 13–18 s | **evidence_only** | see A4 |
| C20 | Release notes v0.2.1 (G1) | 2045 passed, 1 skipped, 9 xfailed, 0 failed | **evidence_only** | the suite has grown since; the count is a fact about e804fc4 |
| C21 | VALIDATION_SUMMARY §1 | baseline before any change: 540 passed, 7 min 56 s | **evidence_only** | log present |
| C22 | VALIDATION_SUMMARY §2 (Installation) | CI smoke green on Ubuntu / Windows / macOS | **reproduced** | 5/5 jobs success, including the three smoke jobs |
| C23 | VALIDATION_SUMMARY §14.7 | B4 cost: simulation 5.66 CPU-h in 35 min; training 2.10 CPU-h in 23 min | **evidence_only** |  |
| C24 | VALIDATION_SUMMARY §2 (DC motor + MPC) | default exact QP: ≈ 7.6× the PI's CPU cost | **evidence_only** |  |

## What it changed

- **AI-N43 (C13):** the ESP32 low-speed notice in the GUI quoted the never-shipped B4 candidate's int8 numbers as the shipped model's. Registered and corrected on `dev` (`54a7606`, `tests/test_ai_n43_esp32_notice.py`) before the v0.2.2 gates.
- **C08 caveat (AI-N42):** the v0.2.1 G6 is reproduced, but it covered only the promoted PC model.
- **C16 (stale):** `VALIDATION_SUMMARY.md` §2 still opens the PMSM row with the 3-substep default; left for Luciano (B2 does not edit documents).

Machine: Linux, Python 3.12, while the v0.2.2 gates ran (timings not comparable).
