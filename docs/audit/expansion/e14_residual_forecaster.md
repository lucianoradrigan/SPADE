# E14: dc_motor forecast-residual anomaly detector

Round 2026-10-21, task T2 (decision D2). Branch `feat/E14-residual-forecaster` from `dev` (a6bcb56). Machine: Linux x86-64, Python 3.12.13, TensorFlow 2.21 / Keras 3.15 (CPU).

Evaluation report: [`docs/audit/ai/dc_motor_residual_anomaly.md`](../ai/dc_motor_residual_anomaly.md) / `.json`. Findings: AI-N18, AI-N19, AI-N20 in `docs/AI_TECHNICAL_AUDIT.md` §10.

## What was built

- **`ai/residual.py`:**
  - `raw_windows`: physical segments placed exactly as in `build_direct_forecast_windows`.
  - `forecast`: normalises like training and maps the forecast back to physical units.
  - `channel_rmse`, `fit_threshold`, `score`, `is_anomalous`.
  - `fit_threshold` raises if any window is not labelled `normal`.
- **Training pipeline (backward compatible):**
  - Optional `ForecasterConfig` fields `channels`, `target_channels`, `label_filter` and `split: three_way`. Existing configs load unchanged.
  - `train_model.py`, in the three-way path, records `val_groups`/`train_groups`, per-channel test and persistence RMSE, `input_window`/`horizon`, and fits `metrics.json["residual_detector"]` on the healthy validation runs.
- **Configs and runs:** `configs/regressors/dc_motor_residual_{pc,pc_novv,rpi5,esp32}.yaml` and their runs `…/2026-10-21_run01`. The edge runs were distilled from the PC run (`--distill --teacher-run`) and exported to TFLite (rpi5 float16, esp32 int8). Each run has its `audit_findings.json`.
- **`experiments/report_residual_anomaly.py`:**
  - all 6 criteria, both variants plus the edge runs;
  - the fault-frequency table and the horizon rule;
  - start-up and reference steps;
  - the agent comparison;
  - TFLite parity.
- **GUI:**
  - Monitoring: "Forecast residual" as the 4th consensus detector. `emitted` is always False, so it never alerts by itself.
  - AI Diagnostics: a "Forecast residual" panel for dc_motor, with forecast and measured signals overlaid, the score against the threshold, the AI SUGGESTION badge, and the text "behaviour differs from learned healthy dynamics — not a fault class".
  - Model Registry: shows the detector block.
  - With no promoted dc_motor regressor every view says **Residual detector not available**. The runs are not promoted (D2).

## Results (fixed criteria)

| Criterion | Target | PC with voltage_v (GUI) | PC without voltage_v |
|---|---|---|---|
| 1. Forecast / persistence RMSE per channel | ≤ 0.8 | PASS* (current_r 0.72–0.73, rpm 0.22–0.56) | PASS* (rpm 0.23–0.65) |
| 2. AUROC outer / inner, ×1 | ≥ 0.9 | 1.000 / 1.000 | 1.000 / 1.000 |
| 3. False alarms (noise 100 %, drift, unseen speeds) | ≤ 1 % | 0 % in every condition | 0 % |
| 4. Ball / cage / 1.3× order | report | AUROC 1.000; detection 51 / 59 / 65 % | 75 / 46 / 69 % |
| 5. Start-up / reference steps | report | 1.4 % / 8.3 % | 4.9 % / 5.6 % |
| 6. Edge parity | parity, cost, 0 Flex | RPi5: 100 % decision agreement, 0.14 ms/window, no Flex. ESP32: 100 %, 0.02 ms, no Flex. **But both distilled detectors fail criterion 1 for rpm and flag ≤ 6 % of faults (AI-N20): not exported to the edge.** | — |

\* **Declared deviation.** Healthy steady-state current_r and rpm are constant in the simulator, so the persistence ratio on those windows is undefined. PASS means every evaluable reading is ≤ 0.8: whole held-out runs, and the train-time test split. This rule was added after seeing the undefined ratios; no faulted window was involved.

**Variant choice.** `with_voltage_v`, chosen by criteria 1 and 3 only. The AUROC was 1.000 for both and could not influence the choice. Note: the other variant flags more faults at its own threshold.

**Fault frequency.** The highest is 196.1 Hz (inner race at 250 rad/s), so the half period is 25.5 samples. Horizon: 16 samples. No decimation.

## Pending decisions

1. Promote `dc_motor_residual_pc/2026-10-21_run01` (activates the GUI panels). Not done.
2. AI-N18: refit the threshold on steady-state validation windows (a regime filter), which also concerns the 8.3 % false alarms on reference steps.
3. Which variant goes to the GUI, now that detection at the threshold is known.

## Tests

| File | Tests |
|---|---|
| `tests/ai/test_residual.py` | 8, one `slow` |
| `tests/test_residual_gui.py` | 6 |
| `tests/test_findings_e14.py` | 4 (2 strict xfail) |

Smoke test: `python scripts/smoke_test.py --launch` → 6/6 PASS. Full suite: **1023 passed, 4 xfailed (strict: AI-N15, AI-N16, AI-N18, AI-N20), 0 failed** (735 s) — `e14_branch_pytest.txt`.
