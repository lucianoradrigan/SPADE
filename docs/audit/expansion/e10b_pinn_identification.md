# E10b: Parameter Identification (PINN) tab in VSC / DPC

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E10b. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §8.3.
Branch: `feat/E10b-pinn-identification` from `dev` (0342573, E10a merged). Date: 2026-10-21. Machine: Linux x86-64, Python 3.12.13, TensorFlow 2.21 / Keras 3.15 (CPU), Streamlit 1.66.0.

- **Origin:** Fund. `5_PIML/PINN/pinn_ode.ipynb`, an inverse-problem PINN in PyTorch. This is a reimplementation in Keras: no code was copied and nothing from Fund. was downloaded.
- **Depends on V0:** `sim/vsc_params.discretize_zoh` and the identified L = 5.02 mH, R_f = 2 Ω, C = 15 µF.
- **Closure report:** [`docs/audit/ai/pinn_identification.md`](../ai/pinn_identification.md) / `.json`.

## 1. What was built

### 1.1 `src/driveflow/ai/pinn/identification.py`

**Model.** It identifies L, R_f and C of the plant's own model:
- equations: L di_f/dt = v_o − v_c − R_f i_f and C dv_c/dt = i_f − v_c/R, on both axes;
- the inputs are held over each 100 µs step, exactly as `sim/vsc_system.py` steps the plant, so the residual enforces the plant's physics and not a continuous approximation of it;
- R is an operating condition and is **given**: with i_load unmeasured, C and R appear only as 1/C and 1/(RC).

**Inputs.** `from_dataframe(df, R, start_s, length_s)` takes a window of `vc_*`, `i_f_*` and `v_o_*` (the datagen columns). The row before the window supplies the initial state. Missing columns, a non-finite window, too short a window, or one that starts at the first sample raise a readable `ValueError`.

**Two estimators:**
- `pinn`: an MLP N(t) → [i_f, v_c] (3×32, tanh). The loss is the data term plus the residual at 2 collocation points inside every step. The load current uses the *network's* v_c at the start of each step, not the noisy measurement. The parameters are trainable log-offsets from nominal and are **initialised at nominal**, which is an informative start. 3000 Adam steps, about 2.3 s per fit on this CPU.
- `least_squares`: a reference without a network. It is nonlinear least squares of the one-step prediction error with A_d, B_d = `discretize_zoh(L, R_f, C)`.
  - A linear regression on the four discrete regressors was tried first and is unusable: i_load = v_c/R is collinear with v_c, and the minimum-norm solution gave R_f 4.4 times the true value on noise-free data.

**Uncertainty.** The declared method, the same for both estimators, is a **moving-block bootstrap with a 5–95 % percentile interval**:
- blocks of 20 samples are resampled and the estimator is refitted;
- for the PINN each refit also uses a new network seed, so it is a bootstrap ensemble;
- defaults: 5 refits for the PINN, 50 for least squares.

**Noise.** `add_noise` adds Gaussian noise to i_f and v_c, as a % of each signal's RMS. v_o is the controller's command and is taken as exact.

### 1.2 GUI (spec §8.3)

- Files: `viz/adapters/identification_adapter.py` → `viz/views/vsc_parameter_identification.py`, as a new "Parameter Identification" tab of `Simulation / VSC`.
- Fixed text above everything: **"Parameter estimate — not a validated fault diagnosis."** It also says that nominal values are not ground truth for a real converter.
- **Data sources:**
  - Simulated run: a closed-loop DPC run with a chosen capacitor fraction and measurement noise, so the true value is known and shown;
  - Uploaded CSV;
  - PLECS: shown as "Not available in this version".
- **Controls:** the given load R, window start and length, the estimator, and the number of bootstrap refits.
- **Results:**
  - estimate / nominal with its interval and the simulated true value (chart);
  - a table with nominal, estimate, 5–95 % interval, Δ vs nominal and, for simulations, the true value and error;
  - the method and refit count, the window, and the time taken;
  - "Validation of this estimator": the report's numbers at the closest noise level;
  - "Longitudinal trend of C: not available", because it needs comparable windows from several runs;
  - a provenance table. The run is recorded in the run store (E0).
- I checked the tab in Chrome in dark and light mode. Simulated nominal run, 1 % noise, PINN: L −1.3 %, R_f −0.2 %, C −2.9 %, in 15 s.

## 2. Recovery error (summary of `docs/audit/ai/pinn_identification.md`)

- **Cases:** 10 simulated closed-loop cases: nominal; C ×0.8 / ×0.6 / ×0.5 / ×0.4; L ×0.8 / ×1.2; R_f ×0.5 / ×1.5; and a combined C ×0.7 · L ×1.1 · R_f ×1.2.
- **Window and noise:** a 20 ms window after the transient; noise of 0 / 1 / 2 / 5 % with 3 realisations each.
- **Simulator check:** the report's simulator equals `datagen/runner.py` exactly (max |Δ| = 0).

Mean absolute error of the point estimate:

| Noise | PINN L | PINN R_f | PINN C | LS L | LS R_f | LS C |
|---|---|---|---|---|---|---|
| 0 % | 1.3 % | 0.15 % | 1.4 % | 0.00 % | 0.00 % | 0.00 % |
| 1 % | 1.5 % | 0.22 % | 3.0 % | 4.1 % | 0.37 % | 9.8 % |
| 2 % | 1.6 % | 0.32 % | 4.5 % | 8.1 % | 1.1 % | 27.7 % |
| 5 % | 1.8 % | 0.61 % | 9.1 % | 18.0 % | 5.5 % | 60.2 % |

What the numbers say:

1. **Least squares is exact without noise and collapses with noise.** It is an errors-in-variables problem: C is off by 60 % on average at 5 % noise, and by up to 166 %.
2. **The PINN is much more robust to noise, but biased.**
   - Its C is systematically **low**, and the bias grows with noise and with degradation: −1 % nominal at 0 %, and −15 % at C ×0.5 with 5 % noise.
   - L is about 1.5 % low in every case.
   - A degraded capacitor is still clearly separated: C ×0.6 reads −4 % at 1 % noise, i.e. 0.58 of nominal.
   - More steps or more collocation points did not reduce the noise-free bias (tried 6000 steps, 4 points per step, width 64, lr 1e-3). The likely cause is that a smooth network cannot follow the kinks the held inputs put into di/dt.
3. **The bootstrap interval is NOT calibrated.**
   - PINN coverage of the true value: C 0 % / 52 % / 70 % / 78 % at 0 / 1 / 2 / 5 % noise; L 0–63 %; R_f 78–96 %.
   - Least squares: C 0–15 % with noise.
   - The interval measures resampling and optimiser spread, not the bias. The GUI shows these coverages next to the estimate; it does not present the interval as a 90 % interval.
4. **C ×0.4 cannot be identified:** the trained DPC loop diverges on that plant (first non-finite sample at 33.7 ms). C ×0.45 and ×0.5 are still stable. This also says that the DPC network does not tolerate a capacitor degraded below ~0.45.

## 3. Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| `ai/pinn/identification.py` estimates L, C, R_f from windows of v_c, i_f, v_o with a declared uncertainty method | module; `tests/ai/test_pinn_identification.py` (10 tests: window alignment, rejections, least squares exact to 1e-6 on noise-free data, PINN recovering C ×0.6 within 5 %, noise, bootstrap) | PASS |
| Validated on simulations with known parameters, including degraded C, with the recovery error reported | `experiments/report_pinn_identification.py` → `docs/audit/ai/pinn_identification.{md,json}`: error per parameter and per noise level, interval coverage, cases including C ×0.8 / ×0.6 / ×0.5 (C ×0.4 recorded as not identifiable) | PASS (report written; results as in §2) |
| GUI with the fixed text "Parameter estimate — not a validated fault diagnosis." | `tests/test_vsc_identification_page.py` (5 tests: the tab and text before any run; least squares on a simulated C ×0.6 shows 9.000 µF against a true 9 µF; PLECS not available; missing-column rejection; validation-summary lookup) | PASS |

`file_uploader` cannot be driven by AppTest, so the upload path is tested at the adapter (`run_identification` with a file that lacks columns), not through the widget.

## 4. Tests

| File | Tests |
|---|---|
| `tests/ai/test_pinn_identification.py` | 10 |
| `tests/test_vsc_identification_page.py` | 5 |

Full suite on the branch: **905 passed, 0 failed** (641 s; 890 after E10a + 15 new) — `e10b_branch_pytest.txt`.

## 5. Limits

- Everything is simulated, and the estimators assume exactly the model that generated the data. There is no ESR (the A4 degradation model has one, and it would bias R_f and C), and no switching ripple.
- The PINN's C bias and the uncalibrated interval are measured, not fixed. Do not read a PINN C below nominal as degradation without accounting for the bias at that noise level (table above).
- One operating point (R = 8.0064 Ω, |v_ref| = 50 V, 50 Hz) and one window length. Other loads and references were not evaluated.
- No PLECS data. The longitudinal trend of C needs repeated runs of the same converter and is not implemented.
