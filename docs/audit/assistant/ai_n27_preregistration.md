# AI-N27 / S21-2 — pre-registration: converged converter model and impact on the promoted dc_motor classifiers

**Status of this file.**
- Committed **before** the convergence of S21-2 and before any classifier metric are measured. Its criteria are not changed after results are seen (`CLAUDE.md`); any deviation is declared in the report.
- Design and decision rule: Luciano's decisions of 2026-11-02 (option 1 of AI-N27, `DECISIONES_S21_F1_2026-11-02.md` S21-2, and his message of the same day). Choices left to Claude Code are marked **[CC]** and fixed here.
- **Measured before this file (declared):** the S21-1 scan (`s21_dataset_scan.md`), and informal checks while writing `sim/dc_dcm.py`: on `dc_pmg132_v1` at 120 rad/s without load, `"dcm"` ends at 120.000 rad/s with ×10, ×100 and ×1000 substeps (GEM: 179.4 rad/s); `"dcm"` with 1 substep equals the default plant within 2e-10 on three artifact-free cases; explicit `"gem"` is byte-identical to the default. No convergence NRMS and no classifier output was computed.

## 1. The converter model (S21-2)

- `Scenario.dc_converter_model`: `None` / `"gem"` = the production plant, unchanged; `"dcm"` = `sim/dc_dcm.DcmDcMotorSystem`: the converter re-evaluated in each of `dc_plant_substeps` Euler substeps (each at its own start time), discontinuous conduction (the current stays at 0 while the diode blocks, `u ≤ e`), the zero crossing located inside the substep. Default substeps `runner.DCM_DEFAULT_SUBSTEPS` = 100.
- **NRMS** (as R2, `docs/audit/plecs_om/r2_convergence.md`): RMS of the difference over the run / range of the finer solution over the run.

### 1.1 Convergence criterion (from the decision)

`"dcm"` with ×10, ×100 and ×1000 substeps; **converged if NRMS(×100 vs ×1000) < 1e-4 for ω and for i_a in every case below.** If it fails, S21-2 is reported as not converged and the impact study (§2) is **not** run.

Cases [CC] (2.5 s unless stated; stock motor unless stated):
1. `dc_pmg132_v1`, 120 rad/s, no extra load, 1.0 s (the S-21 case);
2. stock motor with the 0b-D1 friction (motor_parameter_overrides = stock values, dc_nominal_overrides = stock ratings), 120 rad/s, 1.0 s;
3. outer_race, electrical severity 16 N·m (×2.0), 80 rad/s, 2.5 s (an S-19-type grid case);
4. cage, ×2.0, 80 rad/s, 2.5 s;
5. control: healthy, 220 rad/s, 1.0 s;
6. control: healthy, 120 rad/s, load 5 N·m, 1.0 s.

### 1.2 Control tolerance (declared now) [CC]

Where GEM's current never goes negative (cases 5 and 6), the two converter models must give the same plant:
- **`"dcm"` ×N vs `"gem"` with `dc_plant_substeps` = N** (same integration; N = 1 and N = 100): NRMS ≤ **1e-9** for ω and i_a. (Not 0: GEM computes the step as `(t + τ) − t`, which is not exactly τ in floating point.)
- Reported, not decided on: `"dcm"` ×100 vs the default plant (1 Euler step) — this includes S-16's discretization error (2.6 % NRMS of i_a in B3).

## 2. Impact on the promoted classifiers (AI-N27)

**Question.** Do the predictions of the three promoted dc_motor classifiers change when the S-21 artifact is removed from the signals they see?

### 2.1 Versions compared [CC]

| Version | Plant | Role |
|---|---|---|
| **GEM100** | `dc_converter_model="gem"`, `dc_plant_substeps=100` | primary: has the artifact (GEM's converter, per period) |
| **DCM100** | `dc_converter_model="dcm"`, `dc_plant_substeps=100` | primary: without it (converged, §1.1) |
| **STORED** | the acquisition as stored in the dataset (production plant, 1 Euler step) | secondary only |

The **decision uses GEM100 vs DCM100**: the same integration, only the converter model differs, so the difference is AI-N27 alone. STORED vs DCM100 is reported as the total difference with respect to what the models were trained on; it also contains S-16 / S-18 (already a separate finding) and does not decide.

### 2.2 Set

- **Affected:** every acquisition with the converter-branch signature in S21-1 (`s21_dataset_scan.json`): 21 of `diagnosis_grid_dataset.parquet` and 103 of `diagnosis_grid_dataset_5class.parquet` → **124 runs** (within the 100–200 of the decision).
- **Control [CC]:** 30 unaffected acquisitions: 15 of the 3-class grid (5 per class) and 15 of the 5-class grid (3 per class), drawn without replacement with `numpy.random.default_rng(2711)` in each grid's acquisition order.
- **Same configuration:** each run is rebuilt with `experiments/generate_diagnosis_grid_dataset.py`'s own job list (`build_jobs`, seeds 3, duration 2.5 s, `--seed 7`, the grid's fault list) and `_simulate`, so the same Scenario seed, severities, ψ_e / J drift and sensor-noise percentages; only the plant options of §2.1 are added.
- **Reconstruction check (precondition):** for 6 acquisitions (the first 3 affected of each grid) the rebuilt default plant must equal the stored parquet **exactly** in `current_r`, `rpm`, `voltage_v`, `torque_nm`, `acc_x`. If not, the study stops and is reported.

### 2.3 Windows

- Each model's own pipeline: `build_classification_windows` with its input window, channels and level scaling, all non-overlapping windows of each run (no cap); calibration `ai.calibration.apply_calibration`, abstention below the run's `abstain_below`. RPi5 and ESP32 through their `model.tflite`, PC through its Keras weights (as R2).
- **Paired windows:** window k of run r in one version and window k of the same run in the other cover the same samples (same instant of the same run). Their counts must be equal (assertion).
- Reported by segment (start-up t < 0.5 s, steady state t ≥ 0.5 s, as R2) and over all windows; **the rule uses all windows**.

### 2.4 Models

| Tier | Run | Inference | Classes |
|---|---|---|---|
| PC | `pc_server_5class_w2048/2026-10-20_run01` | Keras | 5 |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` | TFLite float16 | 5 |
| ESP32 | `esp32_dscnn/2026-10-18_run01` | TFLite int8 | 3 (inner_race, normal, outer_race) |

Every model sees the whole set (both grids). For the ESP32, ball and cage windows enter the agreement but not accuracy or macro-F1 (AI-N13).

### 2.5 Metrics (per model; per label class and overall; affected and control sets separately)

1. Top-1 agreement GEM100 vs DCM100 (window by window).
2. Accuracy against the label, GEM100 and DCM100, and Δ accuracy = DCM100 − GEM100 (percentage points).
3. Macro-F1, GEM100 and DCM100.
4. Mean |Δ calibrated top-1 confidence| (paired), and abstention rate in each version.
5. Secondary: the same four with STORED in place of GEM100.

### 2.6 Decision rule (from the decision)

On the **affected** set, for each model and **each label class with affected windows that the model knows** (ESP32: inner_race and outer_race; PC and RPi5: ball, cage, inner_race, outer_race):

- **innocuous** if top-1 agreement ≥ **0.98** AND |Δ accuracy| ≤ **1 percentage point**, in every such class;
- otherwise **affects**.

The overall verdict is "innocuous" only if all three models are innocuous. The control set is reported (expected agreement ≈ 1) and does not decide. After the result Claude Code stops and brings options 2 and 3 of AI-N27 with their costs.

## 3. Not done (decision S21-4)

No full grid regenerated, no model retrained, no default changed; the regenerated runs stay in a scratch directory and are not committed (only the metrics).
