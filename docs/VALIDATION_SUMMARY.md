# Validation summary — SPADE (2026-10-18)

This page consolidates the technical state after the stability / AI / installation audit. It starts from commit `b053de7`. The detailed reports are:
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
| Do the new tests detect the defects? | fixed files reverted to pre-fix versions | **25 of 43** targeted tests fail on the pre-fix code and pass after; the other 18 pin behavior that was already correct ([evidence](audit/validation/regression_tests_vs_prefix_code.txt)) |
| Stability battery | `python experiments/run_stability_battery.py` | executed, 731 s ([results](audit/stability/results.md)) |
| Classifier evaluation | `python experiments/report_classifier_evaluation.py` | executed, 98 s ([results](audit/ai/classifier_evaluation.md)) |

**Test organization.** The new tests are grouped by pytest markers: `simulation`, `control`, `ai`, `compat` and `slow`. For example, `pytest -m "control and not slow"`.

**Strict xfail.** It marks a test that asserts a property the platform does **not** have yet. Each one is tied to an open finding. If the defect gets fixed, the test reports XPASS and fails the run, so the marker cannot silently outlive the defect.

## 2. Status by module

| Module | Numerical | Reproducible | Dynamic behavior | Open issues |
|---|---|---|---|---|
| DC motor + PI | ✅ first-order convergence to RK45; Euler limit 26× above the production step; 60 s with no drift | ✅ bitwise | ✅ matches the linear design model (rise/settling within 5 %, overshoot within 0.5 pt, disturbance dip within 1.5 %); reaches the current-limited physical bound; robust to the 4 parameter drifts | S-04 (≈15 % small-signal overshoot, documented) |
| DC motor + MPC | ✅ (same plant) | ✅ | ✅ stable; small steady-state offset 0.04–0.09 % (no integrator) | about 65–75× the PI's CPU cost |
| PMSM / FOC | ⚠️ **S-01**: Euler unstable above 265.8 rad/s (< nominal 314) | ✅ | ✅ standstill current steps: Euler = RK45 within 0.12 % of i_s | S-01 latent (not reached by current flows) |
| VSC + DPC | ✅ bounded over the GUI range; physical divergence at R < R\* simulated correctly | ✅ | ⚠️ **S-05**: tracking good only near R = 8 Ω | S-05 |
| Fault injection / signals | ✅ S-06 fixed (no more NaN in the state) | ✅ | n/a | the seed varies only the vibration (Q15 / AI-N07) |
| Diagnosis classifiers | n/a | ✅ | ❌ **AI-N06**: promoted model's inner_race recall = 0; ❌ **AI-N07**: candidate fails at 100 rad/s and at half severity, with confidence ≈ 1 | AI-N06, AI-N07, AI-N12 |
| Forecasters | n/a | ✅ | ⚠️ **AI-N11**: test set is a near-duplicate of the training set | AI-N11 |
| Monitoring / alerts | — | ✅ | ✅ detection / acknowledgement / resolution are separate; AI-N03 and AI-N04 fixed | AI-N02, AI-N08, AI-N09, AI-N10 |
| Installation | — | — | ✅ Ubuntu 26.04 executed (uv + Python 3.11, venv/pip + Python 3.12): smoke 6/6 | Windows, macOS, Ubuntu 22.04/24.04 **not executed** |

## 3. Critical and high issues

| ID | Severity | Status |
|---|---|---|
| AI-N06 — the promoted classifier never detects inner_race | Critical | **Open.** Promote the candidate `2026-10-13_run01` through the gate and re-distil the edge models. This is the researcher's decision, because it changes the production model. |
| AI-N03 — VSC alarm missed the unstable bands R < 1 and 3.0–3.37 Ω | High | **Fixed and tested** |
| AI-N04 — NaN telemetry silently resolved alerts | High | **Fixed and tested** |
| AI-N07 — classifiers do not generalize to other speeds or severities | High | **Open.** Needs a new data design (speed/severity grid, order-tracked inputs). |
| AI-N12 — uncalibrated confidence, no abstention for unknown conditions | High | **Open.** The UI already labels confidence as uncalibrated. |
| S-01 — Euler integration of the PMSM at high speed | High (latent) | **Open**, documented. The API now accepts a solver argument. |
| macOS Intel — an unpinned install breaks (TF ≤ 2.16 plus NumPy 2 / pyarrow 26) | High (installation) | **Mitigated**: `constraints/macos-intel.txt`. The pinned set was checked by emulation on Linux; it has not been run on a Mac. |

## 4. Corrections implemented

| Change | Files |
|---|---|
| GUI inference uses the training normalization (AI-N01) | `models/common/windowing.py` (`normalize_window_like_training`), `viz/adapters/ai_adapter.py` |
| VSC alarm threshold = analytic instability boundary (AI-N03) | `monitoring/rules/vsc_dpc.yaml`, help texts |
| Non-finite telemetry is *indeterminate*, never a resolution (AI-N04) | `monitoring/agents/agent_gateway.py` |
| DC agent uses the real current channel `current_r` (AI-N05) | `agents/dc_motor_agent.py`, `configs/agents.yaml` |
| `Scenario` rejects invalid numeric inputs (S-03) | `datagen/scenario.py` |
| Corrected the PI overshoot claim in the docstring (S-04) | `control/classical/pi_controller.py` |
| Finite unit limit for fault-phase states (S-06) | `datagen/fault_injection.py` |
| Optional ODE solver for the PMSM builder; default unchanged (S-01) | `control/classical/pmsm_foc.py` |
| Intel-Mac dependency constraints | `constraints/macos-intel.txt` |

New code that supports the tests and reports:
- `src/driveflow/validation/`: benches, step metrics and Euler bounds;
- `src/driveflow/ai/evaluation.py`: metrics;
- `experiments/run_stability_battery.py` and `experiments/report_classifier_evaluation.py`;
- `scripts/smoke_test.py`;
- `tests/{simulation,control,ai,compat}/`.

No physical model, trained model or registry entry was changed.

## 5. Installation verification

| OS | Executed here? | Evidence | Status |
|---|---|---|---|
| Ubuntu 26.04 (x86_64), Python 3.11 via uv | **yes**, fresh copy, documented commands | [log](audit/install/ubuntu_26.04_fresh_install_log.txt): smoke 6/6 including the real server health check | **Verified** |
| Ubuntu 26.04, Python 3.12 via venv + pip | **yes** | same log, 6/6 | **Verified** |
| Ubuntu 22.04 / 24.04 | no | resolution OK; CI job `ubuntu-latest` | Pending |
| Windows 10/11 | no | resolution OK for 3.11/3.12 ([summary](audit/install/resolution_summary.txt)); CI job `windows-latest` | Pending |
| macOS Apple Silicon | no | resolution OK (TF 2.21); CI job `macos-latest` | Pending |
| macOS Intel | no | unpinned resolution **broken**; constrained set emulated on Linux: [smoke 6/6; non-slow suite 631 passed, 3 failed](audit/install/macos_intel_emulated_on_linux.txt). The 3 failures are specific to that set: a TFLite size heuristic, and a Transfer Learning `AppTest` flow under Streamlit 1.60, which is capped there by protobuf | Pending (constraints shipped) |

**CI (`.github/workflows/`).** Nothing has run on GitHub yet: the workflows only start once they are pushed.

| Workflow | Trigger | What it runs |
|---|---|---|
| `tests.yml` | each push / PR | `pytest -m "not slow"` on Ubuntu (3.11, 3.12), `pip check`, and the smoke matrix on Ubuntu / Windows / macOS (3.12) with `smoke_test.py --launch` and `pytest -m compat` |
| `validation-scheduled.yml` | weekly + manual | the slow tests, the stability battery and the classifier evaluation; results uploaded as artifacts |

## 6. Pending actions

1. **Researcher decision.**
   - Promote `pc_server/2026-10-13_run01` (AI-N06), then re-distil and re-export the ESP32 / RPi5 models.
   - Choose the hysteresis semantics for GUI monitoring (AI-N02) and whether rules get an off-delay (AI-N08).
2. **Data design** (AI-N07, AI-N11, AI-N12):
   - vary speed, load and severity per run and split by condition;
   - add speed-invariant inputs;
   - calibrate (temperature scaling) and add an "unknown" abstention;
   - give the forecasters varied R / v_ref.
3. **Controller.** Retrain the DPC over a range of R (S-05). Optionally add setpoint weighting to the PI (S-04) and integral action or a disturbance observer to the MPC.
4. **Agents.** Normalize features before DTW, use a banded or compiled DTW, and pass the operating point into the hypotheses (AI-N09, AI-N10).
5. **Installation.** Push the workflows and check that the Windows and macOS smoke jobs pass. Run the guides once by hand on Windows 11, an Apple-Silicon Mac, an Intel Mac and Ubuntu 22.04.
6. **Real data.** A real hold-out set (other machine or bearing, at least two speeds) is the only way to validate any diagnosis claim beyond the simulator.
