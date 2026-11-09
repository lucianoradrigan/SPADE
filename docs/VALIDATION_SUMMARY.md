# Validation summary — SPADE (2026-10-18, second pass 2026-10-19, third pass 2026-10-20, rounds 2026-10-21 and 2026-10-22)

This page consolidates the technical state after the stability / AI / installation audit, which started from commit `b053de7`.

The **second pass** (2026-10-19) fixed the findings the first pass had left open: AI-N02, N06, N07, N08, N09, N10, N11, S-01 and S-05; AI-N12 is mitigated. §2–§4 and §6 describe the state after it. The detailed reports are:
- [Simulation stability](SIMULATION_STABILITY_REPORT.md), findings **S-01…S-08**;
- [AI technical audit](AI_TECHNICAL_AUDIT.md), findings **AI-N01…AI-N12**, which builds on the earlier [`AUDIT_ANSWERS.md`](../AUDIT_ANSWERS.md);
- installation guides for [Ubuntu](INSTALL_UBUNTU.md), [Windows](INSTALL_WINDOWS.md) and [macOS](INSTALL_MACOS.md);
- [demo scenarios](DEMO_VALIDATION_SCENARIOS.md).

The **third pass** (2026-10-20) ran the task list in [`TODO_CLAUDE_CODE.md`](../TODO_CLAUDE_CODE.md). It covered:
- **Cross-validation against the reference implementations (§6):** DPC4PowerElectronics and gym-electric-motor 3.0.3. These comparisons are against implementations, not measured data.
- **Control work:** C1–C3.
- **AI work:** A1, A3, A4.
- **Platform work:** P1–P3.

It added findings **S-09…S-13** and **AI-N13…AI-N14**. Results that change a default or promote a model are implemented and documented, but **not switched on**; they wait for confirmation (§7).

## 1. Test runs

| Run | Command | Result |
|---|---|---|
| Baseline, before any change | `python -m pytest -q` | **540 passed**, 0 failed, 7 min 56 s ([log](audit/validation/baseline_pytest_output.txt)) |
| New battery only | `pytest tests/simulation tests/control tests/ai tests/compat` | all pass; 5 strict xfails (known open defects) ([final log](audit/validation/final_pytest_output.txt)) |
| Final, everything | `python -m pytest -rxXs` | **638 passed, 5 xfailed, 0 failed**, 9 min 41 s ([final log](audit/validation/final_pytest_output.txt)) |
| **Second pass, everything** | `python -m pytest -rxXs` | **663 passed, 0 xfailed, 0 failed**, 9 min 51 s. The 5 strict xfails of the first pass now pass and are ordinary tests ([log](audit/validation/second_pass_pytest_output.txt)) |
| Smoke test, second pass | `python scripts/smoke_test.py --launch` | 6/6 PASS, with the promoted classifier's calibrated output |
| **Third pass, everything** (Python 3.12) | `python -m pytest -rxXs` | **733 passed, 1 xfailed, 0 failed**, 13 min 44 s; the strict xfail is S-13 ([log](audit/validation/third_pass_pytest_output.txt)) |
| Smoke test, third pass | `python scripts/smoke_test.py --launch` | 6/6 PASS |
| Intel-Mac pin emulated on Linux (P2) | `pytest -m "not slow"` with `constraints/macos-intel.txt` | 687 passed, 1 xfailed, 0 failed ([log](audit/install/macos_intel_emulated_on_linux_p2.txt)) |
| Do the new tests detect the defects? | fixed files reverted to pre-fix versions | **25 of 43** targeted tests fail on the pre-fix code and pass after; the other 18 pin behavior that was already correct ([evidence](audit/validation/regression_tests_vs_prefix_code.txt)) |
| Stability battery | `python experiments/run_stability_battery.py` | executed, 731 s ([results](audit/stability/results.md)) |
| Classifier evaluation | `python experiments/report_classifier_evaluation.py` | executed, 98 s ([results](audit/ai/classifier_evaluation.md)) |

**Test organization.** The new tests are grouped by pytest markers: `simulation`, `control`, `ai`, `compat` and `slow`. For example, `pytest -m "control and not slow"`.

**Strict xfail.** It marks a test that asserts a property the platform does **not** have yet. Each one is tied to an open finding. If the defect gets fixed, the test reports XPASS and fails the run, so the marker cannot silently outlive the defect.

## 2. Status by module

| Module | Numerical | Reproducible | Dynamic behavior | Open issues |
|---|---|---|---|---|
| DC motor + PI | ✅ first-order convergence to RK45; Euler limit 26× above the production step; 60 s with no drift | ✅ bitwise | ✅ matches the linear design model; reaches the current-limited bound; robust to the 4 parameter drifts | S-04 (≈15 % small-signal overshoot, documented) |
| DC motor + MPC | ✅ | ✅ | ✅ stable; small steady-state offset 0.04–0.09 % (no integrator) | default `qp_solver="exact"` since 2026-10-20 (C3): ≈ 7.6× the PI's CPU cost (was ≈ 330×) |
| PMSM / FOC | ✅ **S-01 fixed**: default Euler with 3 substeps, stable up to ≈460 rad/s (1.47× nominal); bit-identical to GEM 3.0.3 (V4) | ✅ | ✅ Euler = RK45 within 0.5 % of i_s in the closed-loop demo; MTPA = GEM's table within 1 A (V5); now produces datasets (P1) | S-13 fixed (10 substeps: 5.3 % at 314 rad/s, stable to ≈ 841 rad/s) |
| VSC + DPC | ✅ bounded; divergence below R\* is physics; ADF/BDF = exact ZOH of the upstream L-C filter (V0, V1) | ✅ | ✅ **S-05 fixed**: DPC v4 tracks within 3.1 V from 4 Ω to 1 kΩ. C1: the v3/v4 operating-point selection is the default since 2026-10-20: 1.18 V at 8 Ω. The ported upstream MPC reaches 0.14 V (V2) | **S-09**: no converter voltage limit (120/150 V points are idealized); S-10 input convention |
| Fault injection / signals | ✅ S-06 fixed | ✅ | n/a | — |
| Diagnosis classifiers | n/a | ✅ | ✅ **AI-N06/N07 fixed** with the grid-trained `pc_server/2026-10-18_run01` (3 classes, promoted 2026-10-18 to 2026-10-20): 1.000 on held-out speeds (115, 190, 250 rad/s) and severity ×0.6; 0.969 at 100 % noise. **Promoted since 2026-11-04: `pc_server_5class_w2048/2026-11-04_run01`** (5 classes, trained on the corrected-converter grid, AI-N27 option 2): 1.000 in every steady-state condition of the regenerated evaluation cache (held-out speeds and severity, 100 % noise, plant drift) and 0.55 accuracy during start-up ([`classifier_evaluation_5class_w2048_dcm.md`](audit/assistant/classifier_evaluation_5class_w2048_dcm.md)); it replaced `pc_server_5class_w2048/2026-10-20_run01` (promoted 2026-10-20 to 2026-11-04: 1.000 on the held-out conditions, 0.963 at 100 rad/s, 0.60 during start-up, [`classifier_evaluation_5class_w2048.md`](audit/ai/classifier_evaluation_5class_w2048.md)) (caveat X1: 4 seeds per condition and class, 16 windows per class; the seed changes only the vibration channels and the noise realisation, not the simulated current, rpm or torque, so each condition rests on one current/rpm/torque trajectory per class — [x1_eval_independence.md](audit/ai/x1_eval_independence.md)); calibrated, with abstention | AI-N12 / **AI-N13**: no OOD method catches ball faults (A1 negative); **AI-N14**: 0.58 at 50 rad/s, 0.68 at 150 % noise |
| Forecasters | n/a | ✅ | ✅ **AI-N11 fixed**: condition-split test, RMSE 0.073 vs persistence 1.056 | RPi5: the builtin-only TCN (A3) is promoted since 2026-10-20 |
| Monitoring / alerts | — | ✅ | ✅ AI-N02/N03/N04/N08 fixed: telemetry-time evaluation, off-delay; anomaly agent AUC 1.0 and 0.057 s per call (AI-N09/N10) | thresholds measured on simulated runs only |
| Installation | — | — | ✅ Ubuntu 26.04 executed (uv + Python 3.11, venv/pip + Python 3.12): smoke 6/6; CI smoke green on Ubuntu / Windows / macOS; Intel-Mac pin emulated: 687 passed, 0 failed (P2) | manual runs on Windows, macOS, Ubuntu 22.04/24.04 not done |

## 3. Critical and high issues

| ID | Severity | Status |
|---|---|---|
| AI-N06 — the promoted classifier never detected inner_race | Critical | **Fixed and tested.** `pc_server/2026-10-18_run01` promoted, edge models re-distilled and re-exported. Superseded on 2026-10-20 by `pc_server_5class_w2048/2026-10-20_run01`. |
| AI-N03 — VSC alarm missed the unstable bands R < 1 and 3.0–3.37 Ω | High | **Fixed and tested** |
| AI-N04 — NaN telemetry silently resolved alerts | High | **Fixed and tested** |
| AI-N07 — classifiers did not generalize to other speeds or severities | High | **Fixed and tested** on held-out simulated conditions (speed × severity × noise × plant-drift grid, rpm level kept). Measured on `pc_server/2026-10-18_run01` (3 classes); the promoted `pc_server_5class_w2048/2026-10-20_run01` was evaluated on the same conditions (§6 A2 follow-up). |
| AI-N12 — uncalibrated confidence, no abstention | High | **Mitigated**: temperature scaling and an abstention threshold, fitted on validation. Not solved for unseen fault types (ball). |
| S-01 — Euler integration of the PMSM at high speed | High (latent) | **Fixed and tested** |
| macOS Intel — an unpinned install breaks | High (installation) | **Mitigated**: `constraints/macos-intel.txt`. The pinned set was checked by emulation on Linux (third pass: 0 failures), not on a Mac. |
| AI-N13 — no OOD detector catches unseen ball faults at a usable false-alarm rate | High (negative result) | **Open**: Mahalanobis / energy tried (A1); decision passes to A2 (ball/cage classes) |

## 4. Corrections implemented

**First pass (commit `be577b9`)**

| Change | Files |
|---|---|
| GUI inference uses the training normalization (AI-N01) | `models/common/windowing.py`, `viz/adapters/ai_adapter.py` |
| VSC alarm threshold = analytic instability boundary (AI-N03) | `monitoring/rules/vsc_dpc.yaml`, help texts |
| Non-finite telemetry is *indeterminate*, never a resolution (AI-N04) | `monitoring/agents/agent_gateway.py` |
| DC agent uses the real current channel `current_r` (AI-N05) | `agents/dc_motor_agent.py`, `configs/agents.yaml` |
| `Scenario` rejects invalid numeric inputs (S-03) | `datagen/scenario.py` |
| Corrected PI overshoot docstring (S-04); finite unit limit for fault-phase states (S-06) | `control/classical/pi_controller.py`, `datagen/fault_injection.py` |

**Second pass (2026-10-19)**

| Change | Files |
|---|---|
| PMSM default solver: Euler with 3 substeps (S-01) | `control/classical/pmsm_foc.py`, `validation/benches.py` |
| DPC v4, fine-tuned over R = 3.5–1000 Ω and \|v_ref\| = 25–100 V, now the default (S-05) | `experiments/finetune_dpc_r_range.py`, `configs/dpc_trained_v4_r_range.weights.h5`, `datagen/runner.py` |
| Operating-condition grid dataset; `level_scaling` (rpm level kept) as a recorded, consumer-wide preprocessing option; grid-trained classifier promoted; edge models re-distilled (AI-N06/N07) | `experiments/generate_diagnosis_grid_dataset.py`, `models/common/windowing.py`, `experiments/train_model.py`, GUI adapters, TL workflow, `configs/classifiers/*/2026-10-18_run01` |
| Temperature calibration + abstention, stored per run and shown in the GUI (AI-N12) | `ai/calibration.py`, `viz/views/*` |
| INT8 export calibrated on a random sample of all windows, not on the first 100 (top-1 agreement 0.71 → 0.977) | `experiments/export_tflite.py` |
| Rules evaluated on telemetry time; optional off-delay; VSC rule 10 ms on/off-delay (AI-N02/N08) | `monitoring/rules/schema.py`, `monitoring/agents/agent_gateway.py`, `viz/adapters/monitoring_adapter.py`, `vsc_dpc.yaml` |
| Anomaly agent: operating point in hypotheses, healthy-normalized features, spectral comparison of vibration/current, exact vectorized DTW, inner-race hypotheses, 0.6 s GUI window (AI-N09/N10) | `agents/base.py`, `agents/dc_motor_agent.py`, `agents/detector.py`, `viz/views/monitoring.py` |
| Condition-split VSC dataset, persistence baseline in metrics, forecasters retrained and re-exported (AI-N11) | `experiments/generate_vsc_dpc_grid_dataset.py`, `experiments/train_model.py`, `configs/regressors/*/2026-10-19_run01` |
| Monitoring KPI read the calibration flag from the wrong dict (found by a test) | `viz/views/monitoring.py` |

New evidence scripts:
- `experiments/report_anomaly_agent_roc.py`;
- `experiments/report_forecaster_condition_split.py`;
- `experiments/finetune_dpc_r_range.py` (writes `docs/audit/stability/dpc_r_range_finetune_v*.json`).

**Datasets** are not committed (`.gitignore`). Each one is regenerated by the generator script named in the table above.

## 5. Installation verification

| OS | Executed here? | Evidence | Status |
|---|---|---|---|
| Ubuntu 26.04 (x86_64), Python 3.11 via uv | **yes**, fresh copy, documented commands | [log](audit/install/ubuntu_26.04_fresh_install_log.txt): smoke 6/6 including the real server health check | **Verified** |
| Ubuntu 26.04, Python 3.12 via venv + pip | **yes** | same log, 6/6 | **Verified** |
| Ubuntu 22.04 / 24.04 | no | resolution OK; CI job `ubuntu-latest` | Pending |
| Windows 10/11 | no | resolution OK for 3.11/3.12 ([summary](audit/install/resolution_summary.txt)); CI job `windows-latest` | Pending |
| macOS Apple Silicon | no | resolution OK (TF 2.21); CI job `macos-latest` | Pending |
| macOS Intel | no | unpinned resolution **broken**; constrained set emulated on Linux: [smoke 6/6; non-slow suite 631 passed, 3 failed](audit/install/macos_intel_emulated_on_linux.txt). The 3 failures are specific to that set: a TFLite size heuristic, and a Transfer Learning `AppTest` flow under Streamlit 1.60, which is capped there by protobuf | Pending (constraints shipped) |

**CI (`.github/workflows/`).** First GitHub run (commit `1842121`, 2026-10-19):
- the **smoke jobs passed on `windows-latest`, `macos-latest` (Apple Silicon) and `ubuntu-latest`**, Python 3.12. This is the first execution of the install path on Windows and macOS. It is CI, not a manual run of the guides.
- the `pytest` jobs failed on one test, `test_registry_lists_only_existing_artifacts_with_correct_hashes`. Two historical runs (`esp32_dscnn/2026-09-01_run01`, `esp32_tiny/2026-09-13_run01`) had been committed without their `model.weights.h5`, because `*.h5` is git-ignored. This had made CI red since `a6f8894`. Reproduced in a clean checkout and fixed by adding the two weight files.

| Workflow | Trigger | What it runs |
|---|---|---|
| `tests.yml` | each push / PR | `pytest -m "not slow"` on Ubuntu (3.11, 3.12), `pip check`, and the smoke matrix on Ubuntu / Windows / macOS (3.12) with `smoke_test.py --launch` and `pytest -m compat` |
| `validation-scheduled.yml` | weekly + manual | the slow tests, the stability battery and the classifier evaluation; results uploaded as artifacts |


## 6. Third pass: cross-checks and TODO tasks (2026-10-20, TODO_CLAUDE_CODE.md)

One row per task of [`TODO_CLAUDE_CODE.md`](../TODO_CLAUDE_CODE.md).

**The V rows.** They compare SPADE with the implementations it was ported from: DPC4PowerElectronics (MATLAB/Simulink) and gym-electric-motor. They are not checks against measured data. They show whether the port is faithful, not that the models represent a real machine.

**Every other row.** It measures simulated behavior only.

| Task | What was checked | Result | Evidence |
|---|---|---|---|
| P0 | Baseline on Python 3.12 before the block | 655 passed (not slow), smoke 6/6, CI green on `5427819` | [`p0_baseline.txt`](audit/validation/p0_baseline.txt) |
| V0 | ADF/BDF vs. the ZOH of a physical L-C filter | exact to float32 (L = 5.02 mH, R_f = 2 Ω, C = 15 µF, Ts = 100 µs); "LCL" → L-C | `tests/simulation/test_vsc_params.py` |
| V1 | Parameters in `CCS_MPC.slx` / `DPCvalidation.slx` vs. SPADE | L, R_f, C, Ts, the reference and ADF/BDF match. New findings: S-09 (no converter voltage limit in SPADE; upstream V_dc = 200 V and a ±134 V clamp), S-10 (the data-collection MPC is 5-step), S-11, S-12 | [`dpc4pe_parameters.md`](audit/upstream/dpc4pe_parameters.md) |
| V2 | Upstream CCS-MPC ported (`control/mpc/vsc_ccs_mpc.py`) vs. DPC v3/v4, R = 3.5 Ω–5 kΩ × \|v_ref\| = 10–150 V | MPC N=5: 0.14 V at 8 Ω/50 V (v3 1.18, v4 2.52) and < 1 V on the whole grid; with the ±134 V clamp it fails exactly where S-09 predicts; v4 reaches 14.7 V at 5 kΩ | [`dpc_vs_mpc.md`](audit/stability/dpc_vs_mpc.md) |
| V3 | `train.py` vs. `Main.txt`; any recoverable MATLAB-trained network | port faithful except Adam's step counter and ε; no comparable MATLAB network recoverable (`trainedNet1nDPCv4_1.mat` is 7→80→80→2; the InitFcn 15→16→16→10 network has holdout loss 1325 vs. 1.15 for v1) | [`dpc4pe_parameters.md`](audit/upstream/dpc4pe_parameters.md) §V3 |
| V4 | DC motor (Euler, solve_ivp; new load and voltage steps) and PMSM at 400 rad/s against GEM 3.0.3 | same solver: identical to the last bit (4 cases); DC Euler vs. RK45 0.19 % speed / 2.6 % current; GEM's own 1-step Euler diverges at 400 rad/s (S-01 confirmed independently); **S-13**: the default 3-substep Euler is 24.6 % off RK45 at 314 rad/s open loop | `tests/simulation/test_gem_equivalence.py`, `tests/data/gen_ref_data_v4.py` |
| V5 | GEM 3.0.3's classical controllers (gem_controllers) vs. SPADE's on the same DC plant; MTPA curve | MTPA: closed form = GEM's table within 1.0 A over 2.9–716 N·m (GEM grid resolution); DC small step: GEM PI overshoot 33.6 % vs. SPADE 15.1 % (b = 1) / 3.1 % (b = 0.7); start-up nearly identical. gem_controllers 3.0.3 needs NumPy < 2 | [`gem_controller_comparison.md`](audit/upstream/gem_controller_comparison.md) |
| V6 | GEM upstream drift after 3.0.3 in the ported physics | none: `master` = tag v3.0.3 = latest PyPI release (2025-12-19); the only unmerged branch touching `physical_systems/` (EESM defaults, 2024) does not affect SPADE | [`gem_upstream_drift.md`](audit/upstream/gem_upstream_drift.md) |
| C1 | DPC nominal accuracy, option 1: v3/v4 selected by operating point (`ScheduledDpcController`), region chosen on a design grid | all C1 targets met (8 Ω: 1.18 V at 50 V, 1.75 V at 10 V; 4/20/1000 Ω ≤ 3.10 V); never worse than v4 on 44 held-out points; **default unchanged** (needs confirmation) | [`dpc_operating_point_selection.md`](audit/stability/dpc_operating_point_selection.md) |
| C2 | Optional PI setpoint weight b (S-04) | b = 1 unchanged (byte-identical datagen tests pass); small-step overshoot 15.1 % → 3.1 % (b = 0.7) → 0.4 % (b = 0.5), matching the linear model; disturbance rejection unchanged | `tests/control/test_controller_performance.py::TestPiSetpointWeighting` |
| C3 | MPC CPU cost: `qp_solver="exact"` (closed form + exact piecewise-QP via BVLS, L-BFGS-B as last resort) | 8–190× faster on 4 benchmark runs (MPC ≈ 7.6× PI instead of ≈ 330×); differences come from L-BFGS-B's 50-iteration cap (exact holds the 178.5 A limit, same settled tracking); **default unchanged** (bit-identical; switching needs confirmation) | [`mpc_exact_solver.json`](audit/stability/mpc_exact_solver.json), `TestExactQpSolver` |
| A1 | OOD detector (Mahalanobis on two layers, energy) for ball/cage faults, fitted in-distribution only | **targets not met** (ball flagged 9–13 %, cage unknown-or-fault 66–77 %; AUROC ≈ 0.9). Negative result AI-N13; not activated in the promoted run. New AI-N14: 50 rad/s accuracy 0.58, 150 % noise 0.68 | [`ood_detector.md`](audit/ai/ood_detector.md) |
| A2 | Ball/cage classes: 5-class grid, PC model retrained, RPi5/ESP32 distilled, recalibrated, A1 rerun on custom-order unknown faults | PC: 1.000 on all 5 classes at 150 rad/s, 0.80–0.81 at 100/115 rad/s (cage → ball); ESP32 INT8 export degrades (agreement 0.73); order-6.5 unknown fault caught 97 %, order-1.3 30 %. **Synthetic-only, uncalibrated amplitude; not promoted** | [`AI_TECHNICAL_AUDIT.md`](AI_TECHNICAL_AUDIT.md) §8.4 |
| A3 | RPi5 forecaster without Flex ops: TCN (7×32, kernel 3) distilled from the PC model | 0 Flex ops (GRU: 3); held-out RMSE 0.0642 vs. 0.0634 GRU (×1.01) and 1.056 persistence; TFLite/Keras parity 0.0643/0.0642; **not promoted** (needs confirmation) | [`AI_TECHNICAL_AUDIT.md`](AI_TECHNICAL_AUDIT.md) §8.2 |
| A4 | Capacitor degradation separability, pre-registered cut AUC ≥ 0.70 at C = 80 % (DPC v4 closed loop, logistic regression grouped by operating point) | **passes**: C 70–90 % AUC 1.000 (noise-free; 0.975 at C = 80 % with 5 % measurement noise); ESR 0.05–0.2 Ω not separable (0.51–0.57). Synthetic fault definition only; classifier proposed as a new task, not built | [`vsc_capacitor_separability.md`](audit/ai/vsc_capacitor_separability.md) |
| P1 | PMSM in the data pipeline (`plant_config_id="pmsm_v1"`, FOC_MTPA / FOC_naive) | records with the shared schema + `i_d, i_q, i_d_ref, i_q_ref, u_d, u_q, omega_rad_s` (optional columns, so DC/VSC exports are unchanged: existing datagen tests pass); MTPA 119.9 N·m vs. 59.7 N·m at 200 A; generator `experiments/generate_pmsm_dataset.py` (80 scenarios). No fault model, no diagnosis | `tests/test_datagen_pmsm.py` |
| P2 | Transfer Learning page and TFLite size test under the Intel-Mac pin (Streamlit 1.60, TF 2.16), emulated on Linux | page keeps step widget state (fixes the AppTest `KeyError` after jumping to step 4; values now survive step navigation); size test compares a model whose weights dominate the file. Non-slow suite: **687 passed, 0 failed** (was 631 / 3 failed) | [`macos_intel_emulated_on_linux_p2.txt`](audit/install/macos_intel_emulated_on_linux_p2.txt) |
| P3 | Edge benchmark kit, preparation only (`scripts/edge_benchmark/`): reference vectors, RPi5 script (LiteRT / tflite-runtime), ESP32 TFLM skeleton | **not executed on hardware**; host run only: RPi5 script reports `HOST FALLBACK`, top-1 agreement 1.0 with Keras on 64 windows; ESP32 sources generated, not compiled | `scripts/edge_benchmark/README.md`, `tests/test_edge_benchmark_kit.py` |

## 7. Pending actions

### 7.1 Switched on, 2026-10-20 (confirmed by the user, on `dev`)

- **C1:** `ScheduledDpcController` is the default DPC of every Scenario. Robustness thresholds were re-measured: 1.18 V at the nominal point.
- **A3:** `rpi5_tcn/2026-10-20_run01` is promoted as `vsc_dpc/rpi5/regressor`. It has no Flex ops.
- **C3:** `qp_solver="exact"` is the MPC default.
- **S-13:** the PMSM uses `EULER_NSTEPS = 10`: 5.3 % at 314 rad/s, stable to ≈ 841 rad/s.
- **S-09 (mitigated):** optional ±134 V converter limit (`Scenario.vsc_voltage_limit_v` and a toggle on the VSC page), plus a warning when a run needs more than the original converter can give. The default stays unlimited.
- **A4 follow-up (capacitor):**
  - **Separability with uncertainty:** with ±10 % R/L/R_f uncertainty it still passes the pre-registered cut (AUC 0.96 at C = 80 %).
  - **Agent hypotheses:** the VSC anomaly agent now has `capacitor_80pct` / `capacitor_70pct` hypotheses, and `Scenario.vsc_capacitor_scale` / `vsc_esr_ohm` simulate the fault.
  - **Result:** AUC 1.0, with the cause named correctly for C ≤ 80 %.
  - **Limit:** synthetic-only; no trained classifier.
- **A2 (promoted):** PC and RPi5 run the 5-class models with 2048-sample windows (test accuracy 0.90; cage recognized at 100 rad/s). Ball and cage are synthetic-only. The ESP32 keeps its 3-class model. **Warning (X4, 2026-10-25):** the promoted ESP32 classifier (`esp32_dscnn/2026-10-18_run01`) has 3 classes while PC and RPi5 have 5. A ball or cage fault on the ESP32 is assigned to one of the 3 known classes, probably with high confidence (ball windows: outer_race at mean calibrated confidence 0.99), and no OOD method catches it (AI-N13). The Model Registry shows this next to the entry, and the run's `audit_findings.json` records it.

### 7.2 Waiting for your decision (implemented, evidence written, not switched on)

1. **dc_motor regressor (E14, 2026-10-21): use case defined and evaluated, not promoted.**
   - **What exists:** forecast-residual anomaly detector trained on healthy runs only. Runs `configs/regressors/dc_motor_residual_{pc,pc_novv,rpi5,esp32}/2026-10-21_run01`. Report in [`dc_motor_residual_anomaly.md`](audit/ai/dc_motor_residual_anomaly.md).
   - **PC variant chosen for the GUI (inputs current_r, rpm, voltage_v):**
     - Criterion 1 passes on the evaluable readings. This is a declared deviation: persistence is exact on constant steady-state signals.
     - Criterion 2 passes (AUROC 1.000 outer and inner race).
     - Criterion 3 passes (0 % false alarms in every held-out condition).
   - **Limits:**
     - At the fixed threshold it flags only 51 % / 48 % of faulted windows: start-up windows inflate the threshold (AI-N18).
     - Reference steps give 8.3 % false alarms.
     - The perfect AUROC reflects noise-free simulated signals (AI-N19).
     - The distilled RPi5/ESP32 detectors are not usable (AI-N20).
   - **Integrated, but inactive until promoted:**
     - Monitoring consensus, as a 4th detector that never alerts on its own.
     - AI Diagnostics "Forecast residual" panel.
     - Model Registry.
     - With nothing promoted, the GUI shows "Residual detector not available".
   - **Decisions taken by the user (2026-10-21):** threshold first, then variant, then promotion.
   - **Done (declared design change, made after seeing test results):** steady-state gate of 411 ms from nameplate physics; threshold refitted on gated healthy validation windows; runs `*_run02`; re-evaluated once. Evidence: [`dc_motor_residual_gate.md`](audit/ai/dc_motor_residual_gate.md).
   - **Result:**
     - detection at ×1 is now 100 %;
     - criterion 3 **fails** for both variants: 58.3 % / 71.2 % false alarms at 100 % noise (AI-N21);
     - no variant qualifies, so **nothing promoted**;
     - the threshold was not retuned on the test.
   - **Closed as a negative result (2026-10-22, D2, AI-N22).** The last attempt was a pre-registered noise-aware threshold ([`e14_noise_aware_preregistration.md`](audit/ai/e14_noise_aware_preregistration.md), [`e14_noise_aware_results.md`](audit/ai/e14_noise_aware_results.md)).
     - **Criterion 2:** passes (AUROC 1.000).
     - **Criterion 3:** fails at 100 % noise: 6.7 % false alarms, and 23.1 % of healthy windows out of the calibrated noise range.
     - **Outcome:** not promoted, not integrated, no further attempts.

### 7.3 Open, not blocked on a decision

1. **Real data.** No real measurements exist in this project, so every model number above comes from the simulator. A real hold-out set is the only way to validate any diagnosis claim: another machine or bearing, at least two speeds, inner and outer race faults.
2. **AI-N14.** Measured on `pc_server/2026-10-18_run01` (3 classes): accuracy on answered windows 0.58 at 50 rad/s, below the training grid, and 0.68 at 150 % noise. On the promoted `pc_server_5class_w2048/2026-10-20_run01` (round 2026-10-22, [`promoted_pc_server_5class_w2048/ood_detector_custom_orders.md`](audit/ai/promoted_pc_server_5class_w2048/ood_detector_custom_orders.md)): 0.818 at 50 rad/s, but on only 41 % of the windows (the rest abstained or flagged unknown); 1.000 at 150 % noise; 54 windows each (caveat X1: 2 seeds per condition and class; the seed does not change the simulated current, rpm or torque — [x1_eval_independence.md](audit/ai/x1_eval_independence.md)). The OOD filter in these numbers is not active in the GUI (AI-N17). Extending the training grid down in speed is still open.
3. **Installation.** Run the guides once by hand on Windows 11, an Apple-Silicon Mac, an Intel Mac and Ubuntu 22.04. CI covers the smoke test only.
4. **Hardware.** No `.tflite` model has been run on a Raspberry Pi 5 or an ESP32. `scripts/edge_benchmark/` (P3) is ready for when one is available.

## 8. Expansion (E0–E14), round of 2026-10-21 — partial closure

Full detail, with the evidence for each row: [`audit/expansion/t9_closure.md`](audit/expansion/t9_closure.md) (and `.json`). There is no PR from `dev` to `main`.

### 8.1 Tests per task

Each count is a full suite run with `PYTHONPATH=src python -m pytest -q`. The logs are `docs/audit/expansion/*_branch_pytest.txt`.

| Task | Suite |
|---|---|
| Baseline (before E0) | 760 passed |
| E0 jobs and runs | 823 passed |
| E1 Gym envs | 845 passed |
| E6 buck | 871 passed |
| E10a Unknown condition | 890 passed |
| E10b PINN | 905 passed |
| E12 Algorithm Guide | 949 passed |
| E2 PLECS RPC | 1002 passed |
| T1 findings | 1007 passed, 2 xfailed |
| T2 E14 residual | 1023 passed, 4 xfailed |
| T3 E7 Tuning Studio | 1038 passed, 4 xfailed |
| T4 E5 surrogate | 1050 passed, 4 xfailed |
| T5 E8 RL Lab | 1063 passed, 4 xfailed |
| T6 Pmsm-v0 | 1070 passed, 4 xfailed |
| T7 E11 IGBT | no code (stopped at the feasibility check) |
| T8 E9 DAB prep | 1083 passed, 4 xfailed |
| T9 closure | 1083 passed, 4 xfailed |

The 4 xfails are strict and tied to AI-N15, AI-N16, AI-N18 and AI-N20.

### 8.2 PLECS

Nothing ran with PLECS. No machine with PLECS 5.0 and a license was available. E2 is tested only against the local test double. E3/E4 were not started (D7), and the real 3×3 sweep was not run.

### 8.3 Findings of the round

| ID | Summary |
|---|---|
| AI-N15 | PINN estimates C low (−1.3 to −9.1 %) |
| AI-N16 | PINN interval coverage 52–78 % vs 90 % nominal |
| AI-N17 | No OOD block in the promoted classifier (D6) |
| S-14 | DPC divergence vs capacitance, per load |
| AI-N18 | Residual threshold inflated by start-up windows |
| AI-N19 | Residual AUROC optimistic on noise-free signals |
| AI-N20 | Edge residual detectors not usable |

Negative results without an ID:

| Task | Result |
|---|---|
| E7 | The bandwidth goes to its bound (no effort term). |
| E8 | RL loses to the PI in steady state. |
| T7 | Too few IGBT run-to-failure devices. |

### 8.4 Spec §14 matrix and §20 criteria

§14, 18 rows:

| Result | Rows |
|---|---|
| Passed | 14 |
| Partial | 3: import mapping wizard; link from an out-of-range surrogate to a real sweep; mobile/keyboard/contrast |
| Not run | 1: RUL license label (no RUL screen) |
| Failed | 0 |

In §20, the PLECS Gym criteria that depend on E3/E4 are not run.

### 8.5 Screenshots

`docs/screenshots/expansion/` holds 25 screen-state pairs, light and dark. Its README says, per screen, which of the 5 states could not be reached and why.

### 8.5b E7 follow-up (2026-10-21): effort term, current constraint, noise, and comparison with C2

Evidence: [`audit/expansion/e7_tuning_effort.md`](audit/expansion/e7_tuning_effort.md).

| Controller | Result on the evaluation grid with 0.6 rad/s speed-sensor noise |
|---|---|
| High bandwidth (≥ 142 rad/s): the first E7 candidate, and the PSO and BO picks on J_effort | Limit cycle: 52 % overshoot and 53 V RMS change of the action |
| C2, b = 0.5 | Overshoot 14.8 % → 2.4 % with the same effort and lower current peaks |

- PSO and BO still went to high bandwidth because the tuning grid has no noise.
- **No default changed.**

### 8.6 Pending

- Measure the contrast of the colour tokens.
- Run the worker heartbeat path on Windows. There, a dead worker is detected by heartbeat age only, and this has never run on Windows.
- Make My Plants show a failed sweep.
- Decisions listed in §7.2 and in `t9_closure.md` §6.

## 9. Custom classifier training and transfer-learning modes (2026-10-21)

Evidence: [`audit/ai/custom_training_tl.md`](audit/ai/custom_training_tl.md) (`python experiments/report_custom_training_tl.py`). Simulated data only. Nothing was promoted, and no default changed.

| What | Result |
|---|---|
| A. AI Diagnostics → Train a model. 4 classes (normal, outer race, inner race, user-defined unbalance); PI + MPC; 3 speeds; 3 severities; noise and drift; 72 runs; window 1024 | **Test macro-F1 0.417.** inner_race recall 0, all confused with outer_race and unbalance; normal 1.0. The pipeline works end to end, but this small grid is **not** a usable model. |
| B. Full vs. feature transfer from the promoted 5-class model on MPC + 0-60 % noise | The base already scores macro-F1 1.000 on this condition, so there was nothing to gain. FULL_FINETUNE and FEATURE_EXTRACTOR kept 1.000. NEW_HEAD fell to 0.556 with 72 training windows, and the gate reports it (macro_f1_vs_base FAIL). |
| C. New class (unbalance) on the 3-class `pc_server/2026-10-18_run01` | Full fine-tuning is refused. NEW_HEAD reaches macro-F1 1.000 on 228 test windows. The gate stays BLOCKED until a reviewer approves. |
| Simulation time | 398 s for 108 simulated seconds with 4 processes, 3.7 s per simulated second. One process measured about 3. Extra processes gave no speed-up, so the GUI now estimates without dividing by processes and defaults to 1. |

**Decided (D1, 2026-10-22):** floors macro-F1 ≥ 0.88 and recall ≥ 0.70 for every class, applied to every classification candidate (raise-only). The 0.8 floor in this report predates D1. Re-evaluation of these candidates: [`audit/ai/tl_gate_floors.md`](audit/ai/tl_gate_floors.md).

## 10. Round of 2026-10-22: closing decisions D1–D5

Instructions: the user's round document of 2026-10-22 (decisions D1–D5). Each task ran on its own branch, was merged into `dev` after a green full suite, and its branch was deleted. Nothing was promoted and `main` was not touched.

### 10.1 Tests per task

Each count is a full suite run with `PYTHONPATH=src python -m pytest -q`. The logs are `docs/audit/**/t*_branch_pytest.txt`.

| Task | Suite |
|---|---|
| Baseline (`bca4396`) | 1131 passed, 5 xfailed ([log](audit/expansion/round_2026-10-22_baseline.txt)) |
| T1 "live" wording | 1133 passed, 5 xfailed |
| T2 model attribution | 1133 passed, 5 xfailed |
| T3 ball/cage labelling | 1145 passed, 5 xfailed |
| T4 xfail inventory | 1147 passed, 5 xfailed |
| T5 TL gate floors (D1) | 1156 passed, 5 xfailed |
| T6 E14 last attempt (D2) | 1157 passed, 6 xfailed (new strict xfail AI-N22) |
| T7 E11a IGBT data audit (D3) | 1159 passed, 6 xfailed |
| T8 closure | see [`round_2026-10-22_closure.txt`](audit/expansion/round_2026-10-22_closure.txt) |

### 10.2 What changed

| Task | Result | Evidence |
|---|---|---|
| T1 (D4) | Visible GUI text no longer says "live". A guard test scans every visible string. | [`t1_gui_wording.md`](audit/expansion/t1_gui_wording.md) |
| T2 (D5) | AI-N07 and AI-N14 figures name their run (`pc_server/2026-10-18_run01`). The promoted 5-class model at the AI-N14 conditions: 0.818 at 50 rad/s on 41 % of the windows answered; 1.000 at 150 % noise. | [`t2_model_attribution.md`](audit/ai/t2_model_attribution.md) |
| T3 (D5) | "ball/cage: synthetic-only, uncalibrated amplitude" is shown on every ball/cage run and next to every ball/cage prediction (AI Diagnostics, DC Motor, Monitoring, Model Registry). New runs inherit it. | [`t3_ball_cage_labelling.md`](audit/ai/t3_ball_cage_labelling.md) |
| T4 | Every xfail is strict and cites a finding; a guard test enforces it. | [`xfail_inventory.md`](audit/expansion/xfail_inventory.md) |
| T5 (D1) | TL gate floors are macro-F1 ≥ 0.88 and recall ≥ 0.70 per class, always applied and raise-only, recorded with their origin. | [`t5_tl_gate_floors.md`](audit/ai/t5_tl_gate_floors.md), [`tl_gate_floors.md`](audit/ai/tl_gate_floors.md) |
| T6 (D2) | **E14 closed as a negative result (AI-N22).** The pre-registered noise-aware threshold passes criterion 2 (AUROC 1.000) and fails criterion 3 at 100 % noise. | [`e14_noise_aware_preregistration.md`](audit/ai/e14_noise_aware_preregistration.md), [`e14_noise_aware_results.md`](audit/ai/e14_noise_aware_results.md) |
| T7 (D3) | **E11 closed as a negative result (AI-N23).** At most 2 run-to-failure devices on a comparable protocol. E11b was not started. | [`e11a_igbt_data_audit.md`](audit/ai/e11a_igbt_data_audit.md) |

### 10.3 Findings of the round

| ID | Summary |
|---|---|
| AI-N22 | E14 last attempt: the noise-aware threshold fails criterion 3 at 100 % noise. Closed, negative. |
| AI-N23 | The IGBT aging archive supports at most 2 run-to-failure devices on comparable protocols; no validated RUL model is feasible. Closed, negative. |

### 10.4 Status of E11 and E14

- **E14:** closed as a negative result. The residual detector is not promoted, not integrated, and gets no further attempts. The edge distillates stay discarded (AI-N20).
- **E11:** closed as a negative result.
  - E11b is not started.
  - Option 3 (another NASA PCoE dataset) is postponed: not rejected.

### 10.5 Still pending (not part of this round)

- **PLECS:** E3/E4, the real E2 3×3 sweep and the DAB dataset need a machine with PLECS 5.0 and a license.
- **E11 option 3.**
- **OOD detector for the promoted classifier** (D6).
- **Installations and UI checks:** manual installs on Windows, macOS and Ubuntu 22.04; colour contrast; the worker heartbeat on Windows; My Plants showing a failed sweep.
- **AI-N14:** extending the training grid below 80 rad/s.


## 11. External circuit plants (PLECS / OpenModelica) — rounds of 2026-10-22 and 2026-10-25

Evidence lives in `docs/audit/plecs_om/`. Each `.md` there gives the command that produced its numbers.

### 11.1 What was executed and what was not

| Item | Status | Evidence |
|---|---|---|
| PLECS (real) | **Not executed: real PLECS needs the researcher's machine** (PLECS 5.0 with licence). The researcher kit (A2) is ready; it was not sent (G6). | `a2_researcher_kit.md` |
| A1 surrogate from a PLECS sweep | Executed against the PLECS test double only | `a1_surrogate_from_plecs.md` |
| B0–B2 OpenModelica toolchain, models, FmuAdapter | Executed (OpenModelica 1.27.1, pinned image digest, MSL 4.1.0+maint.om) | `b0_setup.md`, `b1_modelica_plants.md`, `b2_fmu_adapter.md` |
| B3 GO/NO-GO (pre-registered) | Executed | `b3_preregistration.md`, `b3_om_fmu_spike.md` |
| R1 post-hoc analysis of B3 + lesson G7 | Done (documentation) | `b3_om_fmu_spike.md` § Post-hoc analysis; `CLAUDE.md` |
| R2 S-16 sensitivity | Executed with the RK45 reference (user decision a): **S-16 affects diagnosis** (S-18) | `r2_convergence.md`, `r2_preregistration.md`, `r2_s16_sensitivity.md` |
| B4 `SPADE/OmBuck-v0` | Executed | `b4_fmu_gym_env.md` |
| B5 Plant Studio "OpenModelica FMU" engine | Executed (AppTests + screenshots) | `tests/fmu/test_plant_studio_fmu.py`, `docs/screenshots/plecs_om/` |
| B6 RL Lab / Tuning Studio on `SPADE/OmBuck-v0` | Executed (jobs through the worker) | `b6_rl_on_fmu.md` |
| B7 weekly FMU CI | Job added and dispatched once: [run 40225897904](https://github.com/lucianoradrigan/SPADE/actions/runs/40225897904), `fmu` and `slow-and-reports` green (18 passed, 2 xfailed S-15; same B3 verdicts on the CI build) | `b7_fmu_ci.md` |
| B8 FMU distribution | Licence review settled from the OpenModelica 1.27.1 source (user option b): redistribution permitted with the notices in `licenses/fmu/`. `fmu-release.yml` (release with GO FMUs only), `python -m driveflow.fmu fetch`, Plant Studio download job. Release **`fmu-v1` published 2026-10-26** at the user's request (BuckAveraged-linux64 + notices); `fetch` verified against it: see `b8_license_review.md` §7 | `b8_license_review.md`, `docs/guides/BUILD_FMUS_LOCALLY.md` |
| Windows / macOS FMUs | Not executed (Linux x86_64 builds only) | — |

### 11.2 B3 verdicts (pre-registered, unchanged)

| Model | Verdict | Note |
|---|---|---|
| BuckAveraged | **GO** | NRMS 1.7e-6 / 2.1e-6 vs the native buck |
| DcMotorAveraged | **NO-GO** (criterion 4: i_a 2.6 % > 1 %) | **Post-hoc analysis (not pre-registered):** the reference, the native plant with one Euler step per τ, is not converged; the FMU matches Euler × 100 to 2e-5 and GEM RK45 to 5e-7. Reference use only (G2). |
| BuckSwitching, DcMotorSwitching | **NO-GO** | S-15: CVODE aborts at the PWM period |

### 11.3 R2 and S-16

- **R2.1:** Euler substeps do not converge in a load-step scenario. Cause: S-17, the Euler substep time offset.
- **Reference (user decision a, 2026-10-25):** RK45 at 1e-9, shown converged within 6.4e-6 of RK45 at 1e-11.
- **Pre-registered study, NAT vs REF, 60 + 60 runs:** **S-16 affects diagnosis** (finding S-18).
  - The current_r fault-band amplitudes differ by up to 16.5 %. This is in noisy runs, at the 2nd and 3rd harmonics; noise-free runs stay ≤ 2.4 %.
  - Start-up top-1 agreement is 0.92–0.97. REF is more accurate there.
  - In steady state, PC and RPi5 pass their own conditions. The ESP32 changes only on ball/cage windows.
- S-16 stays Open.
- **User decision (2026-10-25): regenerate the training datasets with the reference.** Done in R3 (`r3_datasets_rk45.md`):
  - The Euler generator first reproduces the stored datasets: plant channels bit-identical, vibration within 5e-15.
  - The new files are `data/diagnosis_grid_dataset_5class_rk45.parquet` and `data/diagnosis_grid_dataset_rk45.parquet`. The originals are kept.
  - Per run, the current differs by a median of 0.5–0.8 % NRMS. The operating point changes in 4 of 525 runs.
- **Retrained (R4, user decision 2026-10-25), candidates only:**
  - Runs: `pc_server_5class_w2048/2026-10-25_run01`, `rpi5_resnet1d_se_5class_w2048/2026-10-25_run01`, `esp32_dscnn/2026-10-25_run01`, and the 3-class teacher `pc_server/2026-10-25_run01`.
  - **No improvement on either plant.** PC start-up is slightly worse (0.500 vs 0.600 on the evaluation report's start-up condition); one training seed each.
  - Nothing promoted.
- **User decision (2026-10-26): keep the promoted models** (trained on the Euler datasets). The RK45 candidates and datasets stay as evidence only.
- Evidence: `r2_s16_sensitivity.md`, `r3_datasets_rk45.md`, `r4_retrained_rk45.md`.

### 11.4 New findings

| ID | Severity | Summary | Status |
|---|---|---|---|
| S-17 | Low | `sim/solvers.EulerSolver(nsteps > 1)` evaluates f at t_end + k·h/n (inherited from GEM); latent for the production plants | Open; strict xfail |
| S-18 | Medium | S-16 changes fault-band amplitudes (up to 16.5 %, noisy runs, harmonics) and start-up decisions (agreement 0.92–0.97) | Documented; datasets regenerated with RK45 (R3); retrained candidates (R4) do not improve; promoted models kept (user, 2026-10-26) |
| S-19 | Info | 15 % (5-class) / 7 % (3-class) of training runs do not hold the speed reference (one-quadrant converter vs fault ripple), with both integrations | Documented |

AI-N24 was not used.

### 11.5 Capability table (spec §3.15) with the OpenModelica FMU column

| UI function | RPC batch (PLECS) | FMU stepping (PLECS) | Surrogate Gym | **OpenModelica FMU** (state on `dev`) |
|---|---|---|---|---|
| Start sweep | Yes (test double only) | — | No | Not implemented (there is no FMU sweep route) |
| Fit surrogate | With a valid dataset (A1, test double) | Not available | n/a | Not implemented |
| Run step test | No | **Implemented (P1), not run with PLECS**: an uploaded PLECS FMU (FMI 2.0/3.0, CS or ME) goes through the same 13-check job | No | **Yes**: stage 6 "Run verification" job, 13 checks |
| Publish Gym environment | No | **Implemented (P1), not run with PLECS**: after a passing verification, labelled "no native comparison" | Yes (A1) | **Yes, for a verified B3-GO FMU only** (BuckAveraged). NO-GO models are blocked |
| Train agent | No | Through the published environment (P1), not run with PLECS | Yes | **Yes**: `SPADE/OmBuck-v0` in RL Lab (B6), by capability |
| Tune a controller | — | — | Yes (buck PI) | **Yes**: buck PI on `SPADE/OmBuck-v0` (B6) |
| Compare with native SPADE | Yes, with aligned signals | Not available | Yes, own label | **Yes**: B4 report and verification check 4b (B3 criterion 4) |

### 11.5b P1 (2026-10-26): PLECS FMUs as Gym plants — implemented, not run with PLECS

- **`FmuAdapter`:** now drives FMI 2.0 and 3.0 in Co-Simulation and Model Exchange. In Model Exchange SPADE integrates with RK4, locates switching events by bisection and enters Event Mode on discrete inputs.
- **Plant Studio:** offers a "PLECS FMU" engine (upload, export guidance, interface choice).
- **Evidence (`p1_fmu_me_fmi3.md`):**
  - the averaged buck in Model Exchange matches the native buck to 5e-11;
  - **the switching buck, which aborts in Co-Simulation (S-15), runs in Model Exchange and converges.**
- **FMI 3.0 (and 2.0) with real binaries, run with the user's permission (Modelica Association Reference FMUs v0.0.42):**
  - Dahlquist in Model Exchange is within 3.1e-11 of exp(−t);
  - the Stair time events are exact;
  - BouncingBall's state events agree with fmpy's CVode to 1.0e-7 m.
  - A communication step below an FMU's fixedInternalStepSize is now explained and refused.
- **No PLECS here:** no PLECS-exported FMU has been run, so the reviewer's request (PLECS circuits as Gym environments) is implemented end to end but not demonstrated with PLECS.

### 11.6 B8

- **First pass:** stopped by the licence review.
- **Second pass (user option b):** the OpenModelica 1.27.1 tree and its 3rdParty submodule identify every component compiled into the FMUs, and all are permissive with notices:
  - MINPACK (with its acknowledgement);
  - CLAPACK 3.2.1 / libf2c;
  - SUNDIALS, FMI, MSL C and the runtime under BSD New.
  - The only LGPL runtime file (`libcsv.c`) is not in the FMUs.
- **Notices:** `licenses/fmu/`.
- **Release:** `fmu-release.yml` builds, re-runs B3 and creates a draft release with GO FMUs only (BuckAveraged; DcMotorAveraged excluded by G2). **`fmu-v1` was published on 2026-10-26 at the user's request.** `python -m driveflow.fmu fetch` against GitHub installs it (SHA-256 checked), and `SPADE/OmBuck-v0` on it matches the native buck to 6.0e-6.

### 11.7 Screenshots

`docs/screenshots/plecs_om/`: Plant Studio → Use as a Gymnasium plant, with the FMU engine in five readiness states, light and dark. Index in its `README.md`.


## 12. Assistant, phase 0 (simulator prerequisites) — round of 2026-11-02

Instructions: the user's round document `INSTRUCCIONES_CLAUDE_CODE_2026-11-02.md` (decisions A1–A7); planning documents in [`docs/assistant/`](assistant/). **No assistant exists and no language model is used**: this phase only adds optional simulator inputs. Evidence: `docs/audit/assistant/`. Every task ran on its own branch, was merged fast-forward into `dev` after a green full suite, and its branch was deleted. `main` was not touched.

### 12.1 Tests per task (full suite, `PYTHONPATH=src python -m pytest -q`, Python 3.12, fmpy and data/ present)

| Task | Suite |
|---|---|
| Baseline (`df1950f`) | 1318 passed, 9 xfailed; smoke 6/6 (`f0_baseline.txt`) |
| F0.0 planning documents | 1318 passed, 9 xfailed |
| F0.1 `load_torque_nm` | 1331 passed, 9 xfailed |
| F0.2 supply, limits, nominals, design check | 1346 passed, 9 xfailed |
| F0.3 `pmsm_motor_parameters` | 1361 passed, 9 xfailed |
| F0.4 `describe_plant()` | 1418 passed, 9 xfailed |
| F0.5 simulation cost | 1418 passed, 9 xfailed |
| F0.6 tuning on another motor | 1426 passed, 9 xfailed |
| F0.7 S-20 and closure | 1426 passed, 9 xfailed; smoke `--launch` 6/6 |

No xfail was added; the inventory (`docs/audit/expansion/xfail_inventory.md`) is unchanged and `tests/test_xfail_inventory.py` passes.

### 12.2 New `Scenario` fields (all optional; `None` takes the original code path)

| Field | Plant | Meaning |
|---|---|---|
| `load_torque_nm` | dc_perm_ex_v1 | Constant passive load torque (N·m, ≥ 0), added to the stock friction constant `a` |
| `dc_supply_v` | dc_perm_ex_v1 | DC link voltage (default 65 V) |
| `dc_limit_overrides` / `dc_nominal_overrides` | dc_perm_ex_v1 | Motor limits / rated values (`omega`, `torque`, `i`, `u`); `u` limit ≤ supply |
| `pmsm_motor_parameters` | pmsm_v1 | `p`, `l_d`, `l_q`, `r_s`, `psi_p`, `j_rotor`, `u_supply`, merged over `MOTOR_PARAMS` |

Each is validated in `__post_init__` (finite, range, keys) and rejected on any other plant. **Defaults bit for bit:** the existing datagen and `tests/test_scml_system_regression.py` tests were not modified and pass; explicit stock values (zero load, 65 V, stock limits and nominals, explicit `MOTOR_PARAMS`) give bit-identical records (new tests). The tuning with the stock motor gives the same numbers as before the change (F0.6).

Other additions: `control/design_check.py` (diagnostic only), `datagen/plant_catalog.describe_plant()` with `configs/plant_catalog.yaml`, `experiments/measure_sim_cost.py` with `configs/sim_cost.yaml`, an optional `params.motor` block in `controller_tuning`. One function changed for a case it could not handle: `pmsm_foc.mtpa_id_iq` returns i_d = 0 for a non-salient machine (it divided by `l_d − l_q`).

### 12.3 Controller-design check

| Motor | τ_m/τ_a | PI | MPC (horizon/τ_a) |
|---|---|---|---|
| Stock (GEM PMG 132) | 13.6 | ok | ok (1.68) |
| Case B, 220 V (R_a 1.6 Ω, L_a 16 mH, ψ_e 1.146, J 0.05) | 6.4 | **warning** | **warning** (0.20) |

Despite the warnings, PI and MPC bring the 220 V motor to rated speed without extra load (PI error 0.000 rad/s, 0.31 % overshoot; MPC 0.19 rad/s offset). **At rated load torque neither reaches rated speed** (148 rad/s, voltage at the 220 V supply): the stock load's friction (a + b·ω = 7.9 N·m at 157 rad/s), sized for the 60 V PMG 132, adds to the rated torque. Not a finding against the controllers; recorded as an open point for the assistant (§12.6).

### 12.4 Simulation cost (this machine only: Ryzen 5 5600X, idle)

| Plant | Median CPU s per simulated s |
|---|---|
| DC + PI | 3.16 |
| DC + MPC | 5.41 |
| PMSM FOC_MTPA | 1.14 |
| VSC + DPC | 13.75 |
| Buck + PI | 0.67 |

### 12.5 Findings and observations

- **S-20** (Info, Documented): the stock DC motor matches the PMG 132 datasheet in R, L, J and rated speed, not in the 60 V S1 rating (110 A vs 97 A, 20 N·m vs 16 N·m, ψ_e 0.181–0.194 vs 0.165). SPADE validates its port of GEM, not the real motor. No default changed.
- GEM keeps an explicit current limit, but changing motor parameters alone keeps the stock 210 A limit (documented in `Scenario`).
- The stock motor needs 188 A for rated torque at 300 rad/s (stock friction b·ω = 15 N·m there), above the PI's 168 A cap (F0.1).
- AI-N26 had already been fixed earlier on 2026-11-02 at the user's request (`df1950f`); decision A6 was superseded by that. The next free IDs are S-21 and AI-N27.

### 12.6 Not executed / pending decisions

- A load ramp within a run, the induction motor, GUI changes, any LLM: no ejecutado (decisions A1, A5, A7; Phase 1+).
- Default load friction for another motor, tuning grids for another motor, PMG 132 values: decided by the user on 2026-11-02 and done in round 0b (§12.7).
- Simulation cost on other machines: no ejecutado.

### 12.7 Round 0b (2026-11-02): the user's decisions on §12.6

Same method: one branch per decision from `dev`, green full suite, fast-forward merge, branch deleted; `main` not touched.

| Decision | What was done | Evidence |
|---|---|---|
| D1. Default load according to the motor | Stock motor keeps a = 0.01, b = 0.05 (bit for bit). Another motor with known rated torque and speed gets a + b·ω_rated = **2 %** of its rated torque (half constant, half viscous); without those ratings the stock load stays. One rule for simulation and tuning (`runner.dc_default_load_parameter`). The proportional option was dropped after showing the stock friction is 94 % of the stock rated torque at rated speed. Case B at rated load: 156.4 rad/s (0.45 % below rated, limited by its supply) instead of 148.0. | [`f0b_1_load_default.md`](audit/assistant/f0b_1_load_default.md) |
| D2. Tuning grids from the base motor's ratings | `objectives.grids_for(motor)`: speeds, steps and noise × ω_rated/300, loads × T_rated/16; stock grids unchanged; a changed motor without rated speed and torque is refused. Case B's fastest condition is 104.7 rad/s instead of 200. | [`f0b_2_tuning_grids.md`](audit/assistant/f0b_2_tuning_grids.md) |
| D3. PMG 132 datasheet values | Separate plant `dc_pmg132_v1` (ψ_e 0.181, rated 110 A / 20 N·m / 2870 min⁻¹, limits 200 A / 38 N·m), PI and MPC; `dc_perm_ex_v1` unchanged, so no trained model, dataset or regression reference is affected. Holds rated speed from 0 to 100 % rated torque. Not used by datasets, Gym environments or the GUI. **Exception to Phase 0 decision A3** ("no new `plant_config_id`"): A3 kept the new Phase 0 fields as optional overrides of the existing plants; the user chose a separate plant for D3 on 2026-11-02 (commit `409b3f7`) so that the datasheet values would not change `dc_perm_ex_v1`, the plant of every trained model, dataset and regression reference. | [`f0b_3_pmg132_plant.md`](audit/assistant/f0b_3_pmg132_plant.md) |

| Task | Suite |
|---|---|
| D1 | 1430 passed, 9 xfailed |
| D2 | 1432 passed, 9 xfailed |
| D3 and closure | 1461 passed, 9 xfailed; smoke `--launch` 6/6 |

## 13. Simulator artifact S-21 in the training data of the dc_motor classifiers (AI-N27): mitigated for PC and RPi5, open for ESP32

> **Open; measured: AFFECTS (2026-11-03).** The three promoted dc_motor classifiers were trained on simulated data that contain a non-physical converter artifact. A pre-registered study (`docs/audit/assistant/ai_n27_impact.md`) shows they change their decision on 30–54 % of the windows of the affected regime (80–125 rad/s, high severity) when the physics is corrected, and on none of the control runs. Their recorded metrics stay as recorded and nothing has been demoted. **Option 2 (2026-11-04):** the grids were regenerated with the corrected converter and candidates retrained with the same recipe and split (`docs/audit/assistant/ai_n27_regen.md`, pre-registered). The PC and RPi5 candidates meet the pre-registered criterion. The ESP32 candidate does not: its outer_race recall collapses, mostly in int8. **Promoted on 2026-11-04 (Luciano):** the PC and RPi5 candidates (`*_5class_w2048/2026-11-04_run01`). AI-N27 is **mitigated** for PC and RPi5 and **open** for the ESP32, which keeps `esp32_dscnn/2026-10-18_run01` with the warning that it was trained on data with the artifact. The failed ESP32 candidate is a new finding, AI-N28 (int8 fragility); a repair round will be pre-registered. On corrected physics, the previously promoted RPi5 did not pass the D1 floor (macro-F1 0.870 < 0.88). The default plant (`dc_perm_ex_v1`) still uses GEM's converter, so the artifact can still appear in GUI simulations used as inference inputs. **New finding AI-N29 (open, High):** the new PC model is confidently wrong at 50 rad/s, below its training grid (2 of 30 answered windows right; the previous PC answered the same windows correctly); the new RPi5 is not affected.

- **Artifact (S-21, `docs/SIMULATION_STABILITY_REPORT.md`):** the one-quadrant converter model applies the full supply for a whole control period when the current is negative; the speed then rises above its reference instead of coasting. S-19 is this mechanism.
- **Training windows that contain it** (`docs/audit/assistant/s21_dataset_scan.md`, rebuilt exactly):

Training windows with the artifact in the runs promoted when AI-N27 was found (PC and RPi5 replaced on 2026-11-04):

| Model (registry entry) | Run | Training windows with the artifact | Classes |
|---|---|---|---|
| `dc_motor/esp32/classifier` | `esp32_dscnn/2026-10-18_run01` | 450 / 6319 (7.1 %) | inner_race 159, outer_race 291, normal 0 |
| `dc_motor/pc/classifier` | `pc_server_5class_w2048/2026-10-20_run01` | 790 / 4440 (17.8 %) | cage 391, ball 223, outer_race 105, inner_race 71, normal 0 |
| `dc_motor/rpi5/classifier` | `rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` | 790 / 4440 (17.8 %) | same as PC |

- **Shown in Model Registry:** each of the three runs carries AI-N27 in its `audit_findings.json`, listed as an audit finding next to the run (no test pins a hash of that file; checked 2026-11-02).
- **Finding:** AI-N27, `docs/AI_TECHNICAL_AUDIT.md` §13.1.

## 14. Rounds 2026-10-22 → 2026-11-09: portability, C2000, edge, v2 plants, assistant (E8, 2026-11-09)

This section catches up on work that was recorded only in its own evidence files. Each row cites the file and the command behind its numbers. **Everything is simulated**; the only board measurements are on the C2000, replaying simulated windows from flash.

### 14.1 Portability and platform (X1–X4, W1)

| Task | Result | Evidence |
|---|---|---|
| X1 (2026-10-26/27). OpenModelica Model Exchange FMUs on Windows, macOS and Linux | Compiled from source on each OS in CI. NRMS v_out / i_L = 3.2e-11 / 5.6e-11 on all three, the same as the Linux Docker build. Functional criteria 1–4 and 7 pass everywhere. Criterion 6 (p50 ≤ 0.5 ms per step) depends on the CPU: GO on macOS (0.381 ms) and the local Ryzen (0.388 ms), NO-GO on the shared Linux (0.840 ms) and Windows (0.719 ms) runners. Declared deviation: the CI exit uses criteria 1–4 and 7 | [`audit/plecs_om/x1_fmu_portability.md`](audit/plecs_om/x1_fmu_portability.md) (CI run 40442231526; `experiments/report_x1_fmu_portability.py`) |
| X1 (AI, 2026-10-25). Independent samples behind the 1.000 figures of the promoted 5-class classifier | The scenario seed changes vibration and sensor noise, not current, rpm or torque; some conditions have one distinct trajectory per class. A caveat sits next to every 1.000 figure | [`audit/ai/x1_eval_independence.md`](audit/ai/x1_eval_independence.md) (`experiments/report_x1_eval_independence.py`) |
| X2. WCAG 2.1 AA contrast, both themes | Every provenance label and text token meets 4.5:1 (essential UI 3:1); components take the active theme's tones; no colour changed. Two Plotly decorations still use dark tones | [`audit/expansion/x2_contrast.md`](audit/expansion/x2_contrast.md) (`pytest tests/test_contrast.py`, 20 tests) |
| X3. Worker life cycle on every OS | `pid_alive` on Windows through `ctypes`; `python -m driveflow.jobs.worker stop` stops every worker cleanly | [`audit/expansion/x3_worker_windows.md`](audit/expansion/x3_worker_windows.md) |
| X4. ESP32 3-class coverage | The promoted ESP32 classifier has 3 classes while PC and RPi5 have 5; a ball or cage fault goes to a known class (ball → outer_race, mean calibrated confidence 0.99) and no OOD method catches it (AI-N13). Shown in the Model Registry and recorded in its `audit_findings.json`. int8 calibration is labelled as simulated data | `tests/test_x4_esp32_class_coverage.py`, §7.1 |
| W1. Windows installer | Double-click installer (embedded Python, dependencies, SPADE, win64 FMUs). Built, installed silently and smoke-tested on a GitHub Windows runner; release `app-v4`. **Not tried on a real PC** | `.github/workflows/windows-installer.yml`, `packaging/windows/RELEASE_NOTES.md`, `tests/test_w1_windows_installer.py` |

### 14.2 C2000 export and Edge Deployment

- **C2000 export (2026-10-30).** `experiments/export_c2000.py` turns a run into float32 C with its window normalization (the C28x cannot run TFLite Micro). Measured on a LAUNCHXL-F28379D (CPU1, 200 MHz) against the pre-registered C1–C5 ([`audit/c2000/criteria.md`](audit/c2000/criteria.md), [`audit/c2000/report.md`](audit/c2000/report.md)):

  | Run | Worst latency | Max error vs. Keras | Pre-registered C5 (5 kHz window) | C5 at 10 kHz (**post hoc**) |
  |---|---|---|---|---|
  | `esp32_dscnn/2026-10-18_run01` | 68.5 ms | 1.2e-7 | pass (102.4 ms) | FAIL (51.2 ms, 1.34×) |
  | `esp32_tiny/2026-10-19_run01` | 19.7 ms | 4.8e-7 | pass (51.2 ms) | pass (25.6 ms, 0.77×) |
  | `rpi5_tcn/2026-10-20_run01` | 520.1 ms | 1.2e-6 | FAIL (51.2 ms) | FAIL (25.6 ms, 20.32×) |

  The pre-registered verdicts stand. The post-hoc column is [`audit/c2000/c5_post_hoc_10khz.md`](audit/c2000/c5_post_hoc_10khz.md) (`python experiments/report_c5_post_hoc_10khz.py`). The promoted `dc_motor/rpi5` classifier needs a 448 KB arena against 128 KB, so it gets no kit (AI-N25).
- **Edge Deployment page.** One tab per board with scope and limits, model download, C2000 export and inference code generated from the run (`ai/edge_examples.py`). What each example is checked for (`tests/test_edge_examples.py`): C2000 compiles and links against a real kit when TI's compiler is installed; RPi Python equals Keras on a PC; RPi C++ and ESP32 only parse and reproduce the training normalization. The sample period comes from the run's `fs_hz` (`tests/test_ai_n26_sample_rate.py`).

### 14.3 Simulator findings S-21 to S-23 and decision E3

| ID | Status | Evidence |
|---|---|---|
| S-21 (Major) | **Corrected in the v2 plants · open in the frozen v1 plants.** GEM's one-quadrant converter applies the full supply for a whole period when the current is negative; the speed rises instead of coasting. v1 shows a non-blocking warning (S21-3) | `SIMULATION_STABILITY_REPORT.md`; [`audit/assistant/s21_3_warning.md`](audit/assistant/s21_3_warning.md) |
| S-22 (Low) | Documented: an FMU stage-6 timing check (p50 ≤ 0.5 ms) failed twice with the machine loaded (p50 0.566 ms) and passes alone. Re-checked in E9 | `SIMULATION_STABILITY_REPORT.md` |
| S-23 (Medium) | **Corrected (2026-11-08).** The catalog gave GEM's stock-motor defaults for `dc_pmg132_v2`; a test now checks catalog = runner for every plant | `tests/test_plant_catalog_defaults_match_runner.py` |

- **Variant B (`dcm_exact`).** Closed-form converter in discontinuous conduction. Converges: worst case 3.0e-5 against ×10000 substeps, 28× cheaper than ×1000 ([`audit/assistant/dc_perm_ex_v2_convergence.md`](audit/assistant/dc_perm_ex_v2_convergence.md), `python experiments/report_dc_perm_ex_v2_convergence.py`). Data equivalence: classifier agreement 1.000 ([`dc_perm_ex_v2_equivalence.md`](audit/assistant/dc_perm_ex_v2_equivalence.md), `python experiments/report_dc_perm_ex_v2_equivalence.py`).
- **E3 (Luciano, 2026-11-08).** `dc_perm_ex_v2` and `dc_pmg132_v2` are the plants of the GUI, the assistant and the DC dataset scripts. The `Scenario` default stays `dc_perm_ex_v1`, and the v1 plants stay bit for bit as the GEM-faithful mode. DC Motor page wall time v2/v1 = 1.02–1.03. Changed (non-byte) tests are declared ([`audit/assistant/e3_v2_plants.md`](audit/assistant/e3_v2_plants.md)).

### 14.4 AI findings AI-N26 to AI-N31

| ID | Status | Evidence |
|---|---|---|
| AI-N26 (Medium) | **Mitigated (software) · board re-measurement pending in B4.** Data at 10 kHz, C2000 budget and examples assumed 5 kHz. Every run records `fs_hz` (2026-11-09) | [`audit/c2000/ai_n26_sample_rate.md`](audit/c2000/ai_n26_sample_rate.md), [`c5_post_hoc_10khz.md`](audit/c2000/c5_post_hoc_10khz.md); `AI_TECHNICAL_AUDIT.md` §12.3 |
| AI-N27 (High) | **Mitigated for PC and RPi5** (corrected-data runs promoted 2026-11-04) · **open for ESP32** | §13; [`audit/assistant/ai_n27_regen.md`](audit/assistant/ai_n27_regen.md) |
| AI-N28 (High) | **Open.** ESP32 int8 fragility. The pre-registered repair did not pass: outer_race 0.560 → 0.717, but normal 0.994 → 0.687 and abstention +19 points. The promoted ESP32 stays; to B4 | [`audit/assistant/esp32_repair.md`](audit/assistant/esp32_repair.md) (`python experiments/report_esp32_repair.py`) |
| AI-N29 (High) | **Open.** The PC classifier promoted on 2026-11-04 is confidently wrong at 50 rad/s, below its grid; to B4 | `AI_TECHNICAL_AUDIT.md` §13.3 |
| AI-N30 (High) | **Open.** The assistant does not meet its benchmarks (§14.5) | `AI_TECHNICAL_AUDIT.md` §13.4 |
| AI-N31 (Low) | **Fixed (2026-11-09).** The tuning card said "custom" for an unmodified motor (introduced by E3) | `tests/assistant/test_assistant_ai_n31.py`; `AI_TECHNICAL_AUDIT.md` §13.5 |

### 14.5 Assistant, phase 1: closed as experimental / internal demo

Local LLM `qwen3.5:9b` through Ollama. Each benchmark had its thresholds committed before its requests were written, with new requests each time (`python experiments/report_f1_11_benchmark.py --bench <f1_11|f1_12|f1_13>`):

| Criterion | Threshold | F1.11 | F1.12 | F1.13 |
|---|---|---|---|---|
| Launches without confirmation | 0 | 0 | 0 | 0 |
| Invented numbers | 0 | 0 | 0 | 0 |
| False statements | 0 | 5 | 4 | 13 (separate reviewer) ✗ |
| Correct plan | ≥ 85 % | 90.5 % | 93.7 % | 88.9 % |
| Asks when a required value is missing | ≥ 90 % | 60 % | 80 % | 93.3 % |
| ≤ 3 questions, numbers with a source | ≥ 90 % | 76 % | 75 % | 93.0 % |
| Citations | ≥ 90 % | 83 % | 100 % | 100 % |
| "Not documented" | ≥ 80 % | 0 % | 75 % | 100 % |
| Impossible requests with an alternative | ≥ 85 % | 100 % | 100 % | 100 % |
| p95 demo / off-lab | 60 / 30 s | 35.8 s | 34.4 s | 32.1 s (✓ / ✗) |

- F1.11 met 7 of 13 criteria, F1.12 met 9 of 13 and F1.13 met 11 of 13 ([`f1_11_benchmark.md`](audit/assistant/f1_11_benchmark.md), [`f1_12_benchmark.md`](audit/assistant/f1_12_benchmark.md), [`f1_13_benchmark.md`](audit/assistant/f1_13_benchmark.md)).
- The harness fixes between benches were designed while looking at the previous bench, and this is declared ([`ai_n30_harness_fixes.md`](audit/assistant/ai_n30_harness_fixes.md), [`ai_n30_f1_12_fixes.md`](audit/assistant/ai_n30_f1_12_fixes.md)).
- In F1.13, the scorer's default-PI list was missing the v2 plants. Fixing it moved the correct plan from 84.1 % to 88.9 %; the overall verdict did not change.
- F1.13's false-statement review was done by a separate agent and was stricter than the earlier ones, so its count is not directly comparable with F1.11 and F1.12.
- A descriptive 14B run on F1.11 did worse (plan 44 %, asks-if-missing 20 %, p95 134 s; [`f1_11_qwen3_14b.md`](audit/assistant/f1_11_qwen3_14b.md)).
- **Pre-committed closing rule applied:** phase 1 is closed, and the assistant stays "experimental / internal demo". There is no F1.14.
- **Next action recorded for AI-N30** (R2, not implemented): the alternatives offered for impossible requests should come from a catalog table and be shown in a card, since most F1.13 false statements are invented alternatives.
