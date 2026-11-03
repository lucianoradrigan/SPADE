# S21-1 — dataset scan for the one-quadrant converter effect (S-21)

**Simulated data; read only** (no plant code run or changed). Decision S21-1 of 2026-11-02.

- **Command:** `python experiments/report_s21_dataset_scan.py`. Raw values: [`s21_dataset_scan.json`](s21_dataset_scan.json).
- **Signature of the converter branch (noise-free):** full supply (voltage_v ≥ 64.9 V) while the speed is already above `grid_speed`. Validated on noise-free simulations: it coincides sample by sample with `current_r` < 0 (dc_pmg132_v1 at 120 rad/s: 2626 vs 2627; stock motor, outer race at high severity, 80 rad/s: 2750 vs 2750, final speed 128.6 rad/s; healthy stock runs at 120 and 220 rad/s: 0 and 0).
- **`current_r` < 0 is NOT used as evidence** in the grids: they add sensor noise to `current_r` after the run (`grid_electrical_noise_pct`, up to ≈ 95 % of the run's std), and every healthy acquisition with a negative current has noise ≥ 31 %. Those counts are kept in the JSON for reference.
- Speed: mean over the second half more than 2% above `grid_speed` (S-19's symptom).

## Datasets

| Dataset | Acquisitions | With the converter branch | Speed > +2 % | Branch without > +2 % | > +2 % without branch | S-19 listed (with branch) |
|---|---|---|---|---|---|---|
| `data/diagnosis_grid_dataset.parquet` | 315 | 21 (6.67 %) | 21 (6.67 %) | 0 | 0 | 21 (21) |
| `data/diagnosis_grid_dataset_rk45.parquet` | 315 | 21 (6.67 %) | 21 (6.67 %) | 0 | 0 | 21 (21) |
| `data/diagnosis_grid_dataset_5class.parquet` | 525 | 103 (19.62 %) | 85 (16.19 %) | 18 | 0 | 78 (78) |
| `data/diagnosis_grid_dataset_5class_rk45.parquet` | 525 | 102 (19.43 %) | 84 (16.0 %) | 18 | 0 | 78 (77) |
| `data/diagnosis_dataset.parquet` | 15 | 0 (0.0 %) | 0 (0.0 %) | 0 | 0 | — |
| `data/diagnosis_dataset_c1_5s.parquet` | 30 | 0 (0.0 %) | 0 (0.0 %) | 0 | 0 | — |

## Training windows of the promoted dc_motor classifiers

| Model | Dataset | Train windows (rebuilt / recorded) | In acquisitions with the branch | Windows that contain the branch | In acquisitions > +2 % speed |
|---|---|---|---|---|---|
| `dc_motor/esp32/classifier` | `data/diagnosis_grid_dataset.parquet` | 6319 / 6319 | 493 (7.8 %) | 450 (7.12 %) | 493 (7.8 %) |
| `dc_motor/pc/classifier` | `data/diagnosis_grid_dataset_5class.parquet` | 4440 / 4440 | 828 (18.65 %) | 790 (17.79 %) | 684 (15.41 %) |
| `dc_motor/rpi5/classifier` | `data/diagnosis_grid_dataset_5class.parquet` | 4440 / 4440 | 828 (18.65 %) | 790 (17.79 %) | 684 (15.41 %) |

Per class (train windows that contain the branch / train windows):

- `dc_motor/esp32/classifier`: inner_race 159/2117, normal 0/2101, outer_race 291/2101
- `dc_motor/pc/classifier`: ball 223/888, cage 391/888, inner_race 71/888, normal 0/888, outer_race 105/888
- `dc_motor/rpi5/classifier`: ball 223/888, cage 391/888, inner_race 71/888, normal 0/888, outer_race 105/888

## Breakdown (Euler grids)

**`data/diagnosis_grid_dataset_5class.parquet` by label**

| label | Acquisitions | With the branch | Speed > +2 % | Max speed error |
|---|---|---|---|---|
| ball | 105 | 26 | 25 | 71.4 % |
| cage | 105 | 54 | 37 | 76.0 % |
| inner_race | 105 | 8 | 8 | 71.3 % |
| normal | 105 | 0 | 0 | 0.0 % |
| outer_race | 105 | 15 | 15 | 74.8 % |

**`data/diagnosis_grid_dataset_5class.parquet` by speed**

| speed | Acquisitions | With the branch | Speed > +2 % | Max speed error |
|---|---|---|---|---|
| 80.0 | 75 | 39 | 39 | 76.0 % |
| 100.0 | 75 | 29 | 29 | 44.7 % |
| 125.0 | 75 | 14 | 12 | 13.1 % |
| 150.0 | 75 | 6 | 5 | 3.5 % |
| 175.0 | 75 | 6 | 0 | 1.7 % |
| 200.0 | 75 | 6 | 0 | 0.5 % |
| 220.0 | 75 | 3 | 0 | 0.0 % |

**`data/diagnosis_grid_dataset_5class.parquet` by severity**

| severity | Acquisitions | With the branch | Speed > +2 % | Max speed error |
|---|---|---|---|---|
| 0.0 | 105 | 0 | 0 | 0.0 % |
| 0.5 | 84 | 0 | 0 | 0.0 % |
| 0.75 | 84 | 9 | 9 | 74.8 % |
| 1.0 | 84 | 18 | 17 | 71.4 % |
| 1.5 | 84 | 35 | 28 | 71.5 % |
| 2.0 | 84 | 41 | 31 | 76.0 % |

**`data/diagnosis_grid_dataset.parquet` by label**

| label | Acquisitions | With the branch | Speed > +2 % | Max speed error |
|---|---|---|---|---|
| inner_race | 105 | 6 | 6 | 72.0 % |
| normal | 105 | 0 | 0 | 0.0 % |
| outer_race | 105 | 15 | 15 | 74.9 % |

**`data/diagnosis_grid_dataset.parquet` by speed**

| speed | Acquisitions | With the branch | Speed > +2 % | Max speed error |
|---|---|---|---|---|
| 80.0 | 45 | 15 | 15 | 74.9 % |
| 100.0 | 45 | 6 | 6 | 38.9 % |
| 125.0 | 45 | 0 | 0 | 0.0 % |
| 150.0 | 45 | 0 | 0 | 0.0 % |
| 175.0 | 45 | 0 | 0 | 0.0 % |
| 200.0 | 45 | 0 | 0 | 0.0 % |
| 220.0 | 45 | 0 | 0 | 0.0 % |

**`data/diagnosis_grid_dataset.parquet` by severity**

| severity | Acquisitions | With the branch | Speed > +2 % | Max speed error |
|---|---|---|---|---|
| 0.0 | 105 | 0 | 0 | 0.0 % |
| 0.5 | 42 | 0 | 0 | 0.0 % |
| 0.75 | 42 | 0 | 0 | 0.0 % |
| 1.0 | 42 | 3 | 3 | 74.9 % |
| 1.5 | 42 | 9 | 9 | 72.0 % |
| 2.0 | 42 | 9 | 9 | 71.4 % |

## Not executed

- How much the branch changes the classifiers' input signals and their accuracy: no ejecutado (needs the converged reference of S21-2; no dataset regenerated, no model retrained, S21-4).
- Non-DC datasets (PMSM, VSC): not applicable (no one-quadrant converter).
