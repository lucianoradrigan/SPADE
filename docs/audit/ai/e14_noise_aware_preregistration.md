# E14 — noise-aware residual threshold: pre-registration (round 2026-10-22, T6)

**Status of this file.** It was written and committed **before** any σ̂ was computed and before any
score of the new design was looked at. The commit that adds it is the pre-registration. No point
below is changed after results are seen; deviations, if any, go into the results report, marked as
such.

**Decision D2** (user, 2026-10-22): one last attempt with a noise-dependent threshold fixed in
advance. If it does not meet the criteria, E14 closes as a negative result with no further attempts.
PC tier only; no promotion and no GUI or Monitoring integration in either outcome.

## 1. What is kept

- **Detector:** the E14 forecaster weights, score definition and per-channel scale, unchanged.
  - The score is the mean over the target channels of RMSE(physical residual over the horizon),
    divided by the median healthy-validation RMSE (`ai/residual.py`).
- **Steady-state gate:** gate_s = 0.4107 s, the 411 ms gate of run02 (`refit_residual_gate.py`), kept as is.
- **Variant:** **`without_voltage_v`** = `configs/regressors/dc_motor_residual_pc_novv/2026-10-21_run02` (inputs and targets current_r, rpm; context 256, horizon 16).
  - **Why this variant:** the E14 rule (user decision of 2026-10-21) selects by criteria 1 and 3, tie → without voltage_v.
  - In run02 both variants passed criterion 1 and both failed criterion 3, which is a tie, so the rule gives `without_voltage_v`.
- **Test data:** the held-out test conditions and caches of E14, unchanged.
  - `data/eval_cache_5class.parquet`: conditions unseen_speed_115_rad_s, unseen_speed_190_rad_s, extrapolated_speed_250_rad_s, noise_100pct, plant_drift_psi-10pct_J+50pct; healthy, outer_race and inner_race at severity ×1, plus ×0.6.
  - No new test condition is added.

## 2. Noise estimator σ̂ (per window, per channel)

- **Input:** the whole physical segment of the window (context + horizon = 272 samples at 10 kHz), per channel.
- **High-pass cutoff f_c = 1000 Hz**, derived from the fault code, not from data:
  - the electrical fault path (`datagen/fault_injection.py`, `BearingFaultLoad`) injects a torque ripple made of the fault order's fundamental and 3 overtones (`_HARMONIC_WEIGHTS = (1.0, 0.35, 0.15, 0.07)`), with no load-zone modulation on this path;
  - the highest-order fault is inner race (order 4.93 × shaft speed, KAt 6203 geometry);
  - the highest speed evaluated in E14 is 250 rad/s, so BPFI = 196.1 Hz and its 4th harmonic is **784.5 Hz**;
  - f_c is the next multiple of 250 Hz above 784.5 Hz × 1.2, i.e. 1000 Hz.
- **Filter:** Butterworth, order 8, zero-phase (`scipy.signal.butter(..., output="sos")` + `sosfiltfilt`).
- **σ̂ = 1.4826 · median(|x_hp − median(x_hp)|) / sqrt(1 − f_c / 5000)**: the robust (MAD) standard deviation of the high-passed signal, rescaled to the full-band standard deviation of white noise.
- **Conditioning variable:** σ̂ of **current_r**.
  - The simulator adds measurement noise only to current_r and the accelerometers (`datagen/runner._apply_white_noise`), so rpm carries no noise.
  - σ̂ of rpm is computed and reported, but not used.
- **Ripple-inflation check, before any calibration:**
  - **Data:** healthy windows only, plus synthetic ripple. No faulted simulation is used.
  - **Windows:** every gated healthy calibration window (§3) with electrical noise ≥ 10 %, where σ̂ is defined away from zero.
  - **Synthetic ripple, added to current_r:** amplitude A · Σ_k w_k cos(2π k f_F t + φ), with w = _HARMONIC_WEIGHTS and f_F = order × (window mean rpm)/60.
    - The check is done for outer race (3.07) and inner race (4.93).
    - A = 2 × 8.0 N·m / ψ_e = **96.97 A**: the largest grid torque ripple (severity × 2) converted to current at the nominal ψ_e = 0.165 Wb, an upper bound on the closed-loop current ripple.
    - φ is drawn uniformly per window, with seed 0.
  - **Measure:** the median over windows of σ̂(with ripple) / σ̂(without) − 1, per fault.
  - **Rule:** if it exceeds **10 %** for either fault, the estimator is not usable, and E14 closes as a negative result (§6) without calibration or test evaluation.

## 3. Calibration set (healthy validation only)

The 21 healthy validation runs recorded in run02 (`val_groups`) reach 91.9 % electrical noise, while the training grid reaches 99.6 %. They therefore do not cover the training grid's noise range. They are complemented with **21 new healthy runs** made with the training-grid recipe (`experiments/generate_diagnosis_grid_dataset.py`):

| Parameter | Value |
|---|---|
| Speeds | the 7 grid speeds (80, 100, 125, 150, 175, 200, 220 rad/s) |
| Electrical noise | **50 %, 75 %, 100 %** (one run per speed and level) |
| Mechanical noise | same level as the electrical noise |
| Plant drift | ψ_e × U(0.9, 1.1), J × U(0.8, 1.5), drawn with seed 22 |
| Seeds | 5001–5021: disjoint from training (1001+), the E14 evaluation (200–203) and every report seed |
| Duration | 2.5 s, as in the grid |

- **Windows:** only those passing the gate (starting ≥ 0.4107 s after the run start). Cap 60 per run, as in run02's fit, for 2520 windows in total.
- **No test-condition run** is in this set: the test speeds 115/190/250 rad/s are not grid speeds, and the test noise and drift runs use seeds 200–203.

## 4. Threshold τ(σ̂)

- **Bins:** **K = 8** equal-count bins of σ̂(current_r) over the 2520 calibration windows. Edges are the 0, 1/8, …, 1 quantiles; duplicate edges, from the noise-free windows at σ̂ ≈ 0, are merged.
- **Threshold per bin:** the **99th percentile** of the score of the calibration windows in that bin.
- **Monotonicity:** τ is made non-decreasing in σ̂ (cumulative maximum over ascending bins).
- **Lookup:** a window takes the τ of the bin containing its σ̂. σ̂ below the first edge uses the first bin.
- **Out of range:** a window with σ̂ above the largest calibration σ̂ gets the answer **"out of calibrated noise range"** instead of a verdict. It is neither an alarm nor a pass.

## 5. Single evaluation on the E14 test (criteria unchanged)

- **Decision statistic:** r = score / τ(σ̂); alarm when r > 1. AUROC is computed on r over in-range windows.
- **Criterion 2** (as in E14, `report_residual_anomaly.py`): AUROC ≥ 0.9 for outer_race and inner_race at severity × 1, pooled over the five held-out conditions, against the pooled healthy in-range windows of those conditions. The per-condition AUROC is reported without a target.
- **Criterion 3:** false alarms ≤ 1 % of healthy in-range windows in **each** of: noise_100pct, plant_drift_psi-10pct_J+50pct, unseen_speed_115_rad_s, unseen_speed_190_rad_s, extrapolated_speed_250_rad_s.
- **Reported, per condition:** the share of windows "out of calibrated noise range", healthy and faulted.
- **Added guard, declared here before the evaluation:** a condition where more than 10 % of the healthy windows are out of range **fails** criterion 3.
  - **Why:** a detector that abstains on most windows of a condition is not usable there, and this rule prevents a pass by abstention.
- **Also reported, without a target:** criterion 1 (forecast vs. persistence, unchanged because the forecaster is unchanged), detection rates at severity ×1 and ×0.6, and the start-up / reference-step false alarms as in E14.
- **Single evaluation:** the evaluation is run once. No parameter (f_c, K, quantile, calibration set, guard) is tuned on its result.

## 6. Cut rule

- **Both criteria 2 and 3 pass:** the detector is NOT promoted and NOT integrated into AI Diagnostics or the Monitoring consensus. The evidence is prepared, and the user is asked.
- **Either criterion fails, or the inflation check of §2 fails:** E14 closes as a negative result:
  - a finding is registered (next free AI-N ID) in `docs/AI_TECHNICAL_AUDIT.md`;
  - `audit_findings.json` of run02 (both variants) records it;
  - `docs/VALIDATION_SUMMARY.md` is updated;
  - the detector is not integrated or promoted;
  - **no further attempts.**
- **Edge tiers** (RPi5 / ESP32 distillates) stay discarded (AI-N20) and are not evaluated.

## 7. Artifacts

- **Script:** `experiments/report_e14_noise_aware.py`, written after this commit. It implements §2–§5 exactly.
- **Results:** `docs/audit/ai/e14_noise_aware_results.{md,json}`.
- **New calibration runs:** simulated by the script and cached under `data/` (not committed).

## Amendment before any computation (second commit, same day)

The first version of §5 required criterion 2 per condition. That is stricter than E14's criterion,
which is pooled over the held-out conditions, while D2 asks for the original criteria. It was
corrected to the pooled form **before any σ̂, score or result of this design was computed**. No
other point changed.
