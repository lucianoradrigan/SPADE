# Validation summary — SPADE (2026-10-18, second pass 2026-10-19, third pass 2026-10-20)

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
| Diagnosis classifiers | n/a | ✅ | ✅ **AI-N06/N07 fixed** with the grid-trained `pc_server/2026-10-18_run01` (3 classes, promoted 2026-10-18 to 2026-10-20): 1.000 on held-out speeds (115, 190, 250 rad/s) and severity ×0.6; 0.969 at 100 % noise. **Promoted since 2026-10-20: `pc_server_5class_w2048/2026-10-20_run01`** (5 classes): 1.000 on the same held-out conditions and at 100 % noise, 0.963 at 100 rad/s, 0.60 during start-up ([`classifier_evaluation_5class_w2048.md`](audit/ai/classifier_evaluation_5class_w2048.md)); calibrated, with abstention | AI-N12 / **AI-N13**: no OOD method catches ball faults (A1 negative); **AI-N14**: 0.58 at 50 rad/s, 0.68 at 150 % noise |
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
- **A2 (promoted):** PC and RPi5 run the 5-class models with 2048-sample windows (test accuracy 0.90; cage recognized at 100 rad/s). Ball and cage are synthetic-only. The ESP32 keeps its 3-class model.

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
   - **Open:** a noise-aware threshold, designed on validation data and evaluated on new held-out conditions, if you want to pursue it.

### 7.3 Open, not blocked on a decision

1. **Real data.** No real measurements exist in this project, so every model number above comes from the simulator. A real hold-out set is the only way to validate any diagnosis claim: another machine or bearing, at least two speeds, inner and outer race faults.
2. **AI-N14.** Measured on `pc_server/2026-10-18_run01` (3 classes): accuracy on answered windows 0.58 at 50 rad/s, below the training grid, and 0.68 at 150 % noise. On the promoted `pc_server_5class_w2048/2026-10-20_run01` (round 2026-10-22, [`promoted_pc_server_5class_w2048/ood_detector_custom_orders.md`](audit/ai/promoted_pc_server_5class_w2048/ood_detector_custom_orders.md)): 0.818 at 50 rad/s, but on only 41 % of the windows (the rest abstained or flagged unknown); 1.000 at 150 % noise; 54 windows each. The OOD filter in these numbers is not active in the GUI (AI-N17). Extending the training grid down in speed is still open.
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

**Waiting for your decision:** the `min_macro_f1` floor, required when the class set changes. No value is proposed in the GUI; 0.8 was used only in this report.

