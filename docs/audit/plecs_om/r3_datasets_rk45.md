# R3 — Classifier training datasets regenerated with the RK45 reference

Command: `python experiments/report_r3_datasets_rk45.py`. Generation: `python experiments/generate_diagnosis_grid_dataset.py --dc-plant-solver rk45 [--faults outer_race,inner_race,ball,cage] --out data/<name>_rk45.parquet --workers 11` (same grid, seeds, noise and plant-drift draws as the originals: `--seeds 3 --duration-s 2.5 --seed 7`).

Decision: user, 2026-10-25, after R2 (S-16 affects diagnosis, S-18): regenerate the datasets with the reference. **Not done:** retraining, promotion, registry or default changes. The Euler datasets are kept unchanged; the promoted models' `run_metadata.json` still point to them.

**Reproduction check before generating** (`r3_euler_reproduction.json`): the generator in Euler mode reproduces 6 sampled runs of the stored datasets with current_r, rpm, torque_nm and voltage_v bit-identical; acc_x/y/z differ by ≤ 4.7e-15 (floating-point rounding). So the RK45 datasets differ from the originals only by the plant integration.

## 5-class (PC, RPi5)

| | Path | Rows | SHA-256 |
|---|---|---|---|
| Euler (original) | `data/diagnosis_grid_dataset_5class.parquet` | 13125000 | `31eb5a0e2f92a0e8…` |
| RK45 (new) | `data/diagnosis_grid_dataset_5class_rk45.parquet` | 13125000 | `b246a943b918d3f5…` |

525 paired runs (ball: 105, cage: 105, inner_race: 105, normal: 105, outer_race: 105). NRMS RK45 vs Euler per run (RMS difference / RK45 range of the run):

| Label | current_r start-up median / max | current_r steady median / max | rpm steady median / max | acc_x steady median / max |
|---|---|---|---|---|
| ball | 5.38e-03 / 3.97e-01 | 7.52e-03 / 4.88e-01 | 3.14e-04 / 2.57e-02 | 1.03e-02 / 1.68e-02 |
| cage | 2.88e-03 / 3.94e-01 | 1.26e-02 / 5.04e-01 | 3.57e-04 / 9.16e-02 | 9.88e-03 / 2.92e-02 |
| inner_race | 5.02e-03 / 4.67e-01 | 7.41e-03 / 5.93e-01 | 3.57e-04 / 2.83e-01 | 1.56e-02 / 2.72e-02 |
| normal | 4.90e-04 / 8.80e-04 | 2.27e-06 / 1.45e-05 | 2.50e-14 / 2.61e-09 | 3.46e-16 / 4.02e-16 |
| outer_race | 5.13e-03 / 4.09e-01 | 7.90e-03 / 4.90e-01 | 3.15e-04 / 8.91e-02 | 3.22e-02 / 4.15e-02 |
| all | 3.18e-03 / 4.67e-01 | 4.89e-03 / 5.93e-01 | 2.22e-04 / 2.83e-01 | 1.14e-02 / 4.15e-02 |

Largest steady-state current differences: `grid_inner_race_w100_s2.00_seed1138` 0.593; `grid_cage_w125_s1.00_seed1195` 0.504; `grid_outer_race_w100_s2.00_seed1137` 0.490; `grid_ball_w100_s1.50_seed1129` 0.488; `grid_cage_w100_s0.75_seed1100` 0.481.

## 3-class (ESP32)

| | Path | Rows | SHA-256 |
|---|---|---|---|
| Euler (original) | `data/diagnosis_grid_dataset.parquet` | 7875000 | `428cbb91268e8a46…` |
| RK45 (new) | `data/diagnosis_grid_dataset_rk45.parquet` | 7875000 | `e7c255ff2053ebd8…` |

315 paired runs (inner_race: 105, normal: 105, outer_race: 105). NRMS RK45 vs Euler per run (RMS difference / RK45 range of the run):

| Label | current_r start-up median / max | current_r steady median / max | rpm steady median / max | acc_x steady median / max |
|---|---|---|---|---|
| inner_race | 5.30e-03 / 2.76e-01 | 6.98e-03 / 3.64e-01 | 3.63e-04 / 6.68e-03 | 1.51e-02 / 2.28e-02 |
| normal | 5.20e-04 / 9.40e-04 | 2.48e-06 / 1.45e-05 | 1.20e-14 / 4.41e-09 | 3.42e-16 / 4.24e-16 |
| outer_race | 5.47e-03 / 3.92e-01 | 7.39e-03 / 4.94e-01 | 3.37e-04 / 5.15e-03 | 3.24e-02 / 4.42e-02 |
| all | 2.84e-03 / 3.92e-01 | 3.66e-03 / 4.94e-01 | 1.81e-04 / 6.68e-03 | 1.51e-02 / 4.42e-02 |

Largest steady-state current differences: `grid_outer_race_w100_s1.50_seed1080` 0.494; `grid_outer_race_w100_s1.50_seed1077` 0.473; `grid_outer_race_w80_s1.00_seed1026` 0.454; `grid_outer_race_w80_s1.00_seed1020` 0.451; `grid_inner_race_w80_s2.00_seed1045` 0.364.

**Next step (needs the user's decision):** retraining the classifiers on these datasets (and a promotion decision) is not part of this task.
## Operating point (mean speed, t ≥ 0.5 s) — `r3_speed_tracking.json`

Computed with an inline check over both parquet files per dataset; the numbers are stored in the JSON.

- **Euler and RK45 reach a different mean speed** (difference > 2 % of the reference) in **4 of 525 runs** of the 5-class dataset and in none of the 3-class dataset.
  - In one run the difference is large: `grid_inner_race_w100_s2.00_seed1138` settles at 1229 rpm with Euler and at 955 rpm, the reference, with RK45.
  - The other three are cage and outer_race runs at 80 rad/s, with a 3–8 % difference.
  - In these runs the training label stays right, but the operating point the models learned from changes.
- **Independent of the integration, finding S-19:** 78 of 525 runs (5-class) and 21 of 315 (3-class) do not hold their reference speed within 5 %, in both versions. They are high-severity fault runs at 80–125 rad/s.
  - The one-quadrant converter cannot brake against the fault ripple, so the mean speed rises above the reference, by up to +72 %.
  - Their `grid_speed` column is therefore the reference, not the speed the run actually had.
