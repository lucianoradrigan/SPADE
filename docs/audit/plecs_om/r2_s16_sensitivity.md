# R2 — Sensitivity of fault diagnosis to the DC plant's discretization (S-16)

Pre-registration: `docs/audit/plecs_om/r2_preregistration.md` (committed before any diagnosis metric). Command: `python experiments/report_s16_discretization_sensitivity.py sensitivity --workers 10 --cache <scratch>/r2_sims.parquet`. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13.

NAT = Euler x1 (production); REF = RK45 rtol=atol=1e-9. Grid: speeds [100.0, 150.0, 200.0] rad/s × severities [1.0, 0.5] × 5 classes × 2 seeds = 60 runs per version, 2.5 s each.

## Convergence of the reference (R2.1)

See `r2_convergence.md`: Euler substeps do not converge in a load-step scenario (S-17); by the user's decision (option a) the reference is RK45 (1e-9), within 6.4e-6 NRMS of RK45 at 1e-11 in every scenario (`r2_rk45_convergence.json`).

## Verdict: **S-16 affects diagnosis**

| Classifier | Regime | Harmless? | Failed conditions |
|---|---|---|---|
| PC | startup | NO | agreement 0.917 < 0.98; |Δ accuracy| 0.025 > 0.01; spectral difference 0.165 ≥ 0.05 |
| PC | steady_state | NO | spectral difference 0.165 ≥ 0.05 |
| RPi5 | startup | NO | agreement 0.925 < 0.98; |Δ accuracy| 0.025 > 0.01; spectral difference 0.165 ≥ 0.05 |
| RPi5 | steady_state | NO | spectral difference 0.165 ≥ 0.05 |
| ESP32 | startup | NO | agreement 0.969 < 0.98; spectral difference 0.165 ≥ 0.05 |
| ESP32 | steady_state | NO | agreement 0.977 < 0.98; spectral difference 0.165 ≥ 0.05 |

## 1. Signal NRMS, NAT vs REF (RMS of the difference / REF range over the run)

| Channel | Start-up median / max | Steady state median / max |
|---|---|---|
| current_r | 2.32e-03 / 3.86e-01 | 2.87e-03 / 4.72e-01 |
| rpm | 3.44e-04 / 2.37e-03 | 1.31e-04 / 5.08e-04 |
| torque_nm | 3.05e-03 / 3.86e-01 | 3.67e-03 / 4.72e-01 |
| acc_x | 8.84e-03 / 3.15e-02 | 1.14e-02 / 3.84e-02 |
| acc_y | 8.75e-03 / 3.14e-02 | 1.13e-02 / 3.81e-02 |
| acc_z | 8.44e-03 / 3.13e-02 | 1.13e-02 / 3.83e-02 |

## 2. current_r spectrum at the fault bands (steady state, ±2 Hz, fundamental and 2 harmonics)

Decision bands (each faulted run's own fault): worst relative difference **16.458%**, median 0.637% (limit 5 %). All bands of all runs, median of the per-run worst: 5.550% (reported, not decided on).

| Fault | Speed | Severity | Noisy | Own-band worst rel. diff |
|---|---|---|---|---|
| outer_race | 100 | ×1 | yes | 10.781% |
| outer_race | 100 | ×1 | no | 0.645% |
| inner_race | 100 | ×1 | yes | 2.199% |
| inner_race | 100 | ×1 | yes | 7.733% |
| ball | 100 | ×1 | no | 2.446% |
| ball | 100 | ×1 | no | 2.446% |
| cage | 100 | ×1 | yes | 1.555% |
| cage | 100 | ×1 | no | 0.433% |
| outer_race | 100 | ×0.5 | no | 0.651% |
| outer_race | 100 | ×0.5 | yes | 0.873% |
| inner_race | 100 | ×0.5 | yes | 5.094% |
| inner_race | 100 | ×0.5 | yes | 4.793% |
| ball | 100 | ×0.5 | yes | 1.244% |
| ball | 100 | ×0.5 | yes | 0.733% |
| cage | 100 | ×0.5 | yes | 0.393% |
| cage | 100 | ×0.5 | yes | 0.435% |
| outer_race | 150 | ×1 | no | 0.691% |
| outer_race | 150 | ×1 | yes | 2.242% |
| inner_race | 150 | ×1 | yes | 16.458% |
| inner_race | 150 | ×1 | yes | 8.277% |
| ball | 150 | ×1 | yes | 3.748% |
| ball | 150 | ×1 | no | 0.649% |
| cage | 150 | ×1 | no | 0.558% |
| cage | 150 | ×1 | no | 0.558% |
| outer_race | 150 | ×0.5 | no | 0.677% |
| outer_race | 150 | ×0.5 | yes | 1.433% |
| inner_race | 150 | ×0.5 | yes | 2.571% |
| inner_race | 150 | ×0.5 | yes | 3.935% |
| ball | 150 | ×0.5 | yes | 1.555% |
| ball | 150 | ×0.5 | yes | 2.408% |
| cage | 150 | ×0.5 | yes | 0.582% |
| cage | 150 | ×0.5 | yes | 0.725% |
| outer_race | 200 | ×1 | yes | 5.709% |
| outer_race | 200 | ×1 | no | 0.897% |
| inner_race | 200 | ×1 | yes | 11.620% |
| inner_race | 200 | ×1 | no | 1.742% |
| ball | 200 | ×1 | yes | 1.143% |
| ball | 200 | ×1 | yes | 0.682% |
| cage | 200 | ×1 | no | 0.625% |
| cage | 200 | ×1 | no | 0.625% |
| outer_race | 200 | ×0.5 | yes | 5.392% |
| outer_race | 200 | ×0.5 | yes | 1.188% |
| inner_race | 200 | ×0.5 | yes | 8.525% |
| inner_race | 200 | ×0.5 | yes | 2.623% |
| ball | 200 | ×0.5 | yes | 0.583% |
| ball | 200 | ×0.5 | yes | 8.854% |
| cage | 200 | ×0.5 | no | 0.625% |
| cage | 200 | ×0.5 | yes | 0.511% |

## 3–5. Classifiers (paired windows)

| Classifier | Regime | Windows | Agreement | Acc. NAT → REF | Δ acc. | Macro-F1 NAT → REF | Mean |Δ conf.| | Abstention NAT → REF |
|---|---|---|---|---|---|---|---|---|
| PC | startup | 120 (120 with a known label) | 0.917 | 0.575 → 0.600 | +0.025 | 0.582 → 0.611 | 0.0187 | 0.558 → 0.550 |
| PC | steady_state | 540 (540 with a known label) | 0.987 | 0.989 → 0.994 | +0.006 | 0.989 → 0.994 | 0.0017 | 0.035 → 0.033 |
| RPi5 | startup | 120 (120 with a known label) | 0.925 | 0.508 → 0.533 | +0.025 | 0.509 → 0.528 | 0.0222 | 0.575 → 0.542 |
| RPi5 | steady_state | 540 (540 with a known label) | 1.000 | 1.000 → 1.000 | +0.000 | 1.000 → 1.000 | 0.0046 | 0.048 → 0.044 |
| ESP32 | startup | 540 (324 with a known label) | 0.969 | 0.701 → 0.710 | +0.009 | 0.701 → 0.711 | 0.0217 | 0.372 → 0.376 |
| ESP32 | steady_state | 2340 (1404 with a known label) | 0.977 | 1.000 → 1.000 | +0.000 | 1.000 → 1.000 | 0.0130 | 0.173 → 0.176 |

ESP32: 3 classes (no ball/cage); accuracy and macro-F1 over its own classes, agreement over all windows.

Figures (`r2_figures/`, cell 22): `current_spectrum.svg`, `ia_startup.svg`, `ia_steady_state.svg`.

Nothing was regenerated or retrained and no default changed (round §6).
## Post-hoc description (not pre-registered; does not change the verdict)

Added after the results, to locate the effect. **The pre-registered verdict above stands: S-16 affects diagnosis (finding S-18).**

**Where the spectral criterion fails.**
- The decision-band worst case, 16.5 %, is cell 24 (inner_race, 150 rad/s, ×1.0, noisy run), at the **3rd harmonic**.
- Every decision band above 5 % is in a **noisy** run, at the 2nd or 3rd harmonic, where the fault line is close to the noise floor.
- On the 15 noise-free faulted runs, the worst is 2.4 % (median 0.65 %). On the fundamentals of all runs, the worst is 1.3 %.

**Largest signal differences.**
- current_r NRMS reaches 0.46 in cell 9 (cage, 100 rad/s, ×1.0, noise-free), then 0.21 (cage, noisy) and 0.07 (ball).
- In those runs the fault ripple drives i_a between about −110 and +211 A. The one-quadrant converter keeps switching branch (supply voltage while i_a < 0).
- The single Euler step places each branch change up to one τ off, which is the S-16 mechanism. The means (rpm, i_a) agree to < 0.1 %.

**Where the classifiers disagree.**
- **Start-up:** all three models fall below 0.98 agreement (PC 0.917, RPi5 0.925, ESP32 0.969). PC and RPi5 are 2.5 points **more** accurate on REF.
- **Steady state, PC:** passes its own two conditions (0.987, Δ +0.006).
- **Steady state, RPi5:** passes its own two conditions (1.000, Δ 0).
- **Steady state, ESP32 (0.977):** its 1404 known-label windows agree to 1.000. Every disagreement is on ball/cage windows, classes it does not have (AI-N13).
- Command for the ESP32 split: `_windows_paired` / `_predict` of the script on the cached simulations; the numbers are in the user report.
