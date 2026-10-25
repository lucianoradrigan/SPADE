# R2 — Sensitivity of fault diagnosis to the DC plant's discretization (S-16): pre-registration

**Status of this file.**
- Committed **before** any diagnosis metric is computed.
- The design, the metrics and the decision rule come from the user's round document (R2.2, 2026-10-25). They are not changed after results are seen. Any deviation is declared in the report.
- What was measured before this file is R2.1 only: the plant's convergence (`r2_convergence.md`, `r2_rk45_convergence.json`), not diagnosis.

**Question.** Does the native plant's discretization error (one forward-Euler step per τ = 100 µs; 2.6 % NRMS of i_a in B3) change what the promoted classifiers see and decide? All their training data came from that plant.

## Versions compared

| Version | Plant integration (Scenario) | Everything else |
|---|---|---|
| **NAT** | production: one Euler step per τ (`dc_plant_solver` and `dc_plant_substeps` unset) | identical |
| **REF** | RK45, rtol = atol = 1e-9 (`dc_plant_solver="rk45"`). User decision of 2026-10-25 (option a). Converged: within 6.4e-6 of RK45 at 1e-11 in every R2.1 scenario | identical |

**The same in both:**
- PI controller at τ = 100 µs, with the action (and the converter branch) held for τ;
- the same Scenario seeds, so the same vibration synthesis seed and the same noise RNG.

## Grid (60 runs per version)

- **Speeds:** 100, 150 and 200 rad/s, all inside the training grid.
- **Severities:** ×1.0 and ×0.5 of the calibrated electrical (8.0) and mechanical (`CALIBRATED_MECHANICAL_SEVERITY`) severities. `normal` runs carry no severity: their two "severity" cells are distinct runs with different seeds.
- **Classes:** normal, outer_race, inner_race, ball, cage.
- **Seeds:** 2 per cell. Scenario seed = 5000 + cell index, so every run has its own seed.
- **Duration:** 2.5 s, as in `experiments/generate_diagnosis_grid_dataset.py`, the generator of the promoted 5-class dataset.
- **Noise:** the generator's rule. With probability 1/3 a run is noise-free; otherwise the electrical and mechanical noise are drawn independently from U(0, 100) %. The draws come from `numpy.random.default_rng(25)`, in grid order, and are the same for NAT and REF.
- **Plant drift:** none (nominal motor). Declared simplification: the generator also draws ψ_e and J drift; they are left out to isolate the discretization.
- **Processing:** the training pipeline's own `build_classification_windows`, with each model's input window, channels and level scaling. The windows do not overlap, so window k of a run in NAT and window k of the same run in REF cover the same samples (paired windows).

**Conditions, reported separately:**
- **start-up transient:** windows entirely in t < 0.5 s (`SETTLE_S` of the classifier evaluation report, where the promoted PC model scores 0.60);
- **steady state:** windows entirely in t ≥ 0.5 s.

## Classifiers (the promoted ones, as in `configs/registry.yaml`)

| Tier | Run | Inference | Classes | Window |
|---|---|---|---|---|
| PC | `pc_server_5class_w2048/2026-10-20_run01` | Keras weights | 5 | 2048 |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` | `model.tflite` (float16), tf.lite.Interpreter | 5 | 2048 |
| ESP32 | `esp32_dscnn/2026-10-18_run01` | `model.tflite` (int8), tf.lite.Interpreter | 3 (no ball or cage) | 512 |

Probabilities are calibrated with each run's temperature (`ai.calibration.apply_calibration`). The abstention threshold is each run's `abstain_below`.

## Metrics

1. **Signal NRMS of NAT vs REF**, per channel (`current_r`, `rpm`, `torque_nm`, `acc_x/y/z`), start-up and steady state separately.
   - NRMS = RMS(NAT − REF) / range of REF over the run.
   - The median and the maximum over runs are reported.
2. **Spectral amplitude of `current_r`** in the steady state (t ≥ 0.5 s, Hann window, one FFT per run). Bands: BPFO, BPFI, BSF and FTF and their first 2 harmonics (k = 1, 2, 3), each ±2 Hz. The fault frequencies come from REF's mean `rpm` and the bearing geometry (`bearing_frequencies.fault_frequencies_hz`).
   - Relative difference = |A_NAT − A_REF| / A_REF, using the peak amplitude in the band.
   - **Decision bands:** for faulted runs, the bands of the run's own fault type (fundamental + 2 harmonics).
   - Every other band is reported, not decided on.
   - The vibration envelope is not used by any classifier pipeline (windows are z-scored raw channels), so it is not computed.
3. **Top-1 agreement NAT vs REF**, window by window, for each classifier and condition, over all windows of all five classes. For the ESP32, ball and cage windows are included in the agreement.
4. **Accuracy against the label, and macro-F1**, in NAT and in REF, for each classifier and condition. For the ESP32, only windows whose label is one of its 3 classes.
5. **Calibrated confidence and abstention:** mean absolute difference of the top-1 calibrated confidence (paired windows), and the abstention rate in NAT and in REF.

## Decision rule

Applied per classifier, separately for start-up and steady state.

| Result | Condition |
|---|---|
| **S-16 harmless for diagnosis** | agreement ≥ 0.98, **and** \|Δ accuracy\| ≤ 1 point (0.01), **and** spectral difference < 5 % in every decision band (the spectral criterion is common to the three classifiers: worst decision band over all faulted runs, steady state) |
| **S-16 affects diagnosis** | any of the conditions fails |

**Overall verdict:** "harmless" only if every classifier passes in both regimes.

**What a failure triggers:**
- New finding **S-17 → renumbered:** S-17 is already taken by the Euler time offset (R2.1), so the next free IDs are used. **S-18** if the effect is in the signals or spectra. **AI-N24** if the signals pass but a model's decisions change (the effect is the model's sensitivity).
- The result of every classifier and regime is reported, even though the verdict is one.

**What a failure does not trigger:** no dataset is regenerated, nothing is retrained and no plant default changes (round §6). The options go to the user.

## Regression test

One `slow` test pins the measured result: the agreement and |Δ accuracy| of the PC model in steady state, at 150 rad/s, with 1 seed, against the measured values with a declared tolerance of ±0.02.

## Artifacts

- Script: `experiments/report_s16_discretization_sensitivity.py sensitivity` (grid arguments, `--cache`, `--out`).
- Results: `r2_s16_sensitivity.{md,json}`.
- Figures (SVG): i_a NAT vs REF at start-up and in steady state; the `current_r` spectrum with the fault-band markers.
