# Validation summary — SPADE (2026-10-18, second pass 2026-10-19)

This page consolidates the technical state after the stability / AI / installation audit, which started from commit `b053de7`.

The **second pass** (2026-10-19) fixed the findings the first pass had left open: AI-N02, N06, N07, N08, N09, N10, N11, S-01 and S-05; AI-N12 is mitigated. §2–§4 and §6 describe the state after it. The detailed reports are:
- [Simulation stability](SIMULATION_STABILITY_REPORT.md), findings **S-01…S-08**;
- [AI technical audit](AI_TECHNICAL_AUDIT.md), findings **AI-N01…AI-N12**, which builds on the earlier [`AUDIT_ANSWERS.md`](../AUDIT_ANSWERS.md);
- installation guides for [Ubuntu](INSTALL_UBUNTU.md), [Windows](INSTALL_WINDOWS.md) and [macOS](INSTALL_MACOS.md);
- [demo scenarios](DEMO_VALIDATION_SCENARIOS.md).

## 1. Test runs

| Run | Command | Result |
|---|---|---|
| Baseline, before any change | `python -m pytest -q` | **540 passed**, 0 failed, 7 min 56 s ([log](audit/validation/baseline_pytest_output.txt)) |
| New battery only | `pytest tests/simulation tests/control tests/ai tests/compat` | all pass; 5 strict xfails (known open defects) ([final log](audit/validation/final_pytest_output.txt)) |
| Final, everything | `python -m pytest -rxXs` | **638 passed, 5 xfailed, 0 failed**, 9 min 41 s ([final log](audit/validation/final_pytest_output.txt)) |
| **Second pass, everything** | `python -m pytest -rxXs` | **663 passed, 0 xfailed, 0 failed**, 9 min 51 s. The 5 strict xfails of the first pass now pass and are ordinary tests ([log](audit/validation/second_pass_pytest_output.txt)) |
| Smoke test, second pass | `python scripts/smoke_test.py --launch` | 6/6 PASS, with the promoted classifier's calibrated output |
| Do the new tests detect the defects? | fixed files reverted to pre-fix versions | **25 of 43** targeted tests fail on the pre-fix code and pass after; the other 18 pin behavior that was already correct ([evidence](audit/validation/regression_tests_vs_prefix_code.txt)) |
| Stability battery | `python experiments/run_stability_battery.py` | executed, 731 s ([results](audit/stability/results.md)) |
| Classifier evaluation | `python experiments/report_classifier_evaluation.py` | executed, 98 s ([results](audit/ai/classifier_evaluation.md)) |

**Test organization.** The new tests are grouped by pytest markers: `simulation`, `control`, `ai`, `compat` and `slow`. For example, `pytest -m "control and not slow"`.

**Strict xfail.** It marks a test that asserts a property the platform does **not** have yet. Each one is tied to an open finding. If the defect gets fixed, the test reports XPASS and fails the run, so the marker cannot silently outlive the defect.

## 2. Status by module

| Module | Numerical | Reproducible | Dynamic behavior | Open issues |
|---|---|---|---|---|
| DC motor + PI | ✅ first-order convergence to RK45; Euler limit 26× above the production step; 60 s with no drift | ✅ bitwise | ✅ matches the linear design model; reaches the current-limited bound; robust to the 4 parameter drifts | S-04 (≈15 % small-signal overshoot, documented) |
| DC motor + MPC | ✅ | ✅ | ✅ stable; small steady-state offset 0.04–0.09 % (no integrator) | about 65–75× the PI's CPU cost |
| PMSM / FOC | ✅ **S-01 fixed**: default Euler with 3 substeps, stable up to ≈460 rad/s (1.47× nominal) | ✅ | ✅ Euler = RK45 within 0.5 % of i_s | — |
| VSC + DPC | ✅ bounded; divergence below R\* is physics | ✅ | ✅ **S-05 fixed**: DPC v4 tracks within 3.1 V from 4 Ω to 1 kΩ (v3: up to 111 V). Trade-off: 2.5 V at 8 Ω (v3 1.18 V) | nominal-point and 10 V-reference accuracy below v3 |
| Fault injection / signals | ✅ S-06 fixed | ✅ | n/a | — |
| Diagnosis classifiers | n/a | ✅ | ✅ **AI-N06/N07 fixed**: grid-trained model, 1.000 on held-out speeds (115, 190, 250 rad/s) and severity ×0.6; 0.969 at 100 % noise; calibrated, with abstention | AI-N12: ball faults (no class) confidently called outer_race |
| Forecasters | n/a | ✅ | ✅ **AI-N11 fixed**: condition-split test, RMSE 0.073 vs persistence 1.056 | RPi5 GRU export needs Flex ops |
| Monitoring / alerts | — | ✅ | ✅ AI-N02/N03/N04/N08 fixed: telemetry-time evaluation, off-delay; anomaly agent AUC 1.0 and 0.057 s per call (AI-N09/N10) | thresholds measured on simulated runs only |
| Installation | — | — | ✅ Ubuntu 26.04 executed (uv + Python 3.11, venv/pip + Python 3.12): smoke 6/6 | Windows, macOS, Ubuntu 22.04/24.04 **not executed** (CI jobs pushed) |

## 3. Critical and high issues

| ID | Severity | Status |
|---|---|---|
| AI-N06 — the promoted classifier never detected inner_race | Critical | **Fixed and tested.** `pc_server/2026-10-18_run01` promoted, edge models re-distilled and re-exported. |
| AI-N03 — VSC alarm missed the unstable bands R < 1 and 3.0–3.37 Ω | High | **Fixed and tested** |
| AI-N04 — NaN telemetry silently resolved alerts | High | **Fixed and tested** |
| AI-N07 — classifiers did not generalize to other speeds or severities | High | **Fixed and tested** on held-out simulated conditions (speed × severity × noise × plant-drift grid, rpm level kept). |
| AI-N12 — uncalibrated confidence, no abstention | High | **Mitigated**: temperature scaling and an abstention threshold, fitted on validation. Not solved for unseen fault types (ball). |
| S-01 — Euler integration of the PMSM at high speed | High (latent) | **Fixed and tested** |
| macOS Intel — an unpinned install breaks | High (installation) | **Mitigated**: `constraints/macos-intel.txt`. The pinned set was checked by emulation on Linux, not on a Mac. |

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


## 7. Cross-check against reference implementations (2026-10-20, TODO_CLAUDE_CODE.md)

**What these checks are.** They compare SPADE with the implementations it was ported from:
- DPC4PowerElectronics (MATLAB/Simulink);
- gym-electric-motor.

They are not checks against measured data. They show whether the port is faithful; they do not show that the models represent a real machine.

| Task | What was checked | Result | Evidence |
|---|---|---|---|
| P0 | Baseline on Python 3.12 before the block | 655 passed (not slow), smoke 6/6, CI green on `5427819` | [`p0_baseline.txt`](audit/validation/p0_baseline.txt) |
| V0 | ADF/BDF vs. the ZOH of a physical L-C filter | exact to float32 (L = 5.02 mH, R_f = 2 Ω, C = 15 µF, Ts = 100 µs); "LCL" → L-C | `tests/simulation/test_vsc_params.py` |
| V1 | Parameters in `CCS_MPC.slx` / `DPCvalidation.slx` vs. SPADE | L, R_f, C, Ts, the reference and ADF/BDF match. New findings: S-09 (no converter voltage limit in SPADE; upstream V_dc = 200 V and a ±134 V clamp), S-10 (the data-collection MPC is 5-step), S-11, S-12 | [`dpc4pe_parameters.md`](audit/upstream/dpc4pe_parameters.md) |

## 6. Pending actions

1. **Real data.** No real measurements exist in this project, so every model number above is from the simulator. A real hold-out set is the only way to validate any diagnosis claim: another machine or bearing, at least two speeds, inner and outer race faults.
2. **Unseen fault types.** The classifier has no ball or cage class. Ball faults are called outer_race with confidence 0.99, and the abstention threshold does not catch them. Either add those classes (their severity calibration is a documented placeholder) or add an out-of-distribution detector.
3. **DPC accuracy at the nominal point.** v4 trades nominal accuracy (2.5 V vs 1.18 V at 8 Ω, 4.2 V at a 10 V reference) for range. One extra round at a lower learning rate made it worse (`dpc_r_range_finetune_v5.json`). A larger network or a loss weighted toward small references are the next options.
4. **RPi5 GRU forecaster** still needs Flex ops on TFLite. An unrolled GRU or a TCN would remove them.
5. **Installation.** Confirm that the Windows and macOS CI jobs pass. Run the guides once by hand on Windows 11, an Apple-Silicon Mac, an Intel Mac and Ubuntu 22.04.
6. **Hardware.** No `.tflite` model has been run on a Raspberry Pi 5 or an ESP32.
