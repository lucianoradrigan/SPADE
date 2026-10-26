# R4 — Classifiers retrained on the RK45 datasets (candidates)

Command: `python experiments/report_r4_retrained_rk45.py --r2-cache <scratch>/r2_sims.parquet --candidates PC=configs/classifiers/pc_server_5class_w2048/2026-10-25_run01 RPi5=configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-25_run01 ESP32=configs/classifiers/esp32_dscnn/2026-10-25_run01`. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13.

Decision: user, 2026-10-25 (after R2 → S-18 and R3). Same recipe as the promoted runs (`train_model.py --epochs 30 --batch-size 32 --seed 42 --keep-rpm-level --cap-per-class 3000`; RPi5 and ESP32 distilled, alpha 0.5, from the new PC teacher of the same dataset; RPi5 float16 and ESP32 int8 TFLite exported as before). **Candidates only: nothing promoted, `configs/registry.yaml` unchanged.**

## 1. Recorded test split (each run's own grouped test split, from its own dataset)

| Tier | Run | Dataset | Test accuracy | Test macro-F1 | n_test | Temperature / abstain below |
|---|---|---|---|---|---|---|
| PC | `pc_server_5class_w2048/2026-10-20_run01` (promoted, Euler) | `diagnosis_grid_dataset_5class.parquet` | 0.903 | 0.903 | 600 | 0.754 / 0.525 |
| PC | `pc_server_5class_w2048/2026-10-25_run01` (candidate, RK45) | `diagnosis_grid_dataset_5class_rk45.parquet` | 0.890 | 0.890 | 600 | 1.078 / 0.493 |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` (promoted, Euler) | `diagnosis_grid_dataset_5class.parquet` | 0.902 | 0.902 | 600 | 0.674 / 0.508 |
| RPi5 | `rpi5_resnet1d_se_5class_w2048/2026-10-25_run01` (candidate, RK45) | `diagnosis_grid_dataset_5class_rk45.parquet` | 0.870 | 0.871 | 600 | 0.820 / 0.513 |
| ESP32 | `esp32_dscnn/2026-10-18_run01` (promoted, Euler) | `diagnosis_grid_dataset.parquet` | 0.901 | 0.902 | 870 | 0.905 / 0.734 |
| ESP32 | `esp32_dscnn/2026-10-25_run01` (candidate, RK45) | `diagnosis_grid_dataset_rk45.parquet` | 0.905 | 0.905 | 870 | 1.002 / 0.738 |

## 2. R2 simulations (unseen seeds; NAT = Euler plant, REF = RK45 plant)

| Tier | Model | Start-up acc. NAT / REF | Start-up macro-F1 NAT / REF | Steady acc. NAT / REF | Steady macro-F1 NAT / REF | Steady abstention NAT / REF |
|---|---|---|---|---|---|---|
| PC | promoted (Euler-trained) | 0.575 / 0.600 | 0.582 / 0.611 | 0.989 / 0.994 | 0.989 / 0.994 | 0.035 / 0.033 |
| PC | candidate (RK45-trained) | 0.533 / 0.575 | 0.536 / 0.576 | 0.985 / 0.985 | 0.985 / 0.985 | 0.033 / 0.033 |
| RPi5 | promoted (Euler-trained) | 0.508 / 0.533 | 0.509 / 0.528 | 1.000 / 1.000 | 1.000 / 1.000 | 0.048 / 0.044 |
| RPi5 | candidate (RK45-trained) | 0.525 / 0.517 | 0.529 / 0.520 | 0.987 / 0.987 | 0.987 / 0.987 | 0.031 / 0.030 |
| ESP32 | promoted (Euler-trained) | 0.701 / 0.710 | 0.701 / 0.711 | 1.000 / 1.000 | 1.000 / 1.000 | 0.173 / 0.176 |
| ESP32 | candidate (RK45-trained) | 0.710 / 0.710 | 0.706 / 0.706 | 1.000 / 1.000 | 1.000 / 1.000 | 0.197 / 0.198 |

ESP32: accuracy and macro-F1 over its own 3 classes.

## 3. PC models on the classifier-evaluation conditions (cached runs, Euler plant)

| Condition | Promoted accuracy / macro-F1 | Candidate accuracy / macro-F1 |
|---|---|---|
| nominal_steady_state | 1.000 / 1.000 | 1.000 / 1.000 |
| startup_transient | 0.600 / 0.633 | 0.500 / 0.455 |
| speed_100_rad_s | 0.963 / 0.962 | 0.963 / 0.962 |
| speed_200_rad_s | 1.000 / 1.000 | 1.000 / 1.000 |
| severity_x0.5 | 1.000 / 1.000 | 1.000 / 1.000 |
| severity_x2 | 1.000 / 1.000 | 1.000 / 1.000 |
| noise_20pct | 1.000 / 1.000 | 1.000 / 1.000 |
| noise_100pct | 1.000 / 1.000 | 1.000 / 1.000 |
| plant_drift_psi-10pct_J+50pct | 1.000 / 1.000 | 1.000 / 1.000 |
| unseen_speed_115_rad_s | 1.000 / 1.000 | 1.000 / 1.000 |
| unseen_speed_190_rad_s | 1.000 / 1.000 | 1.000 / 1.000 |
| extrapolated_speed_250_rad_s | 1.000 / 1.000 | 1.000 / 1.000 |
| unseen_severity_x0.6 | 1.000 / 1.000 | 1.000 / 1.000 |

**Not done:** promotion, registry or default changes (they need the user's decision).
## Reading

**Retraining on the RK45 datasets does not improve diagnosis**, on either plant. This is a negative result.

| Model | Steady state | Start-up |
|---|---|---|
| PC candidate | equal to the promoted model within a point (0.985 vs 0.989–0.994 on the R2 runs; 1.000 on every steady evaluation condition) | worse: 0.533 / 0.575 vs 0.575 / 0.600 on the R2 runs, 0.500 vs 0.600 on the evaluation report's start-up condition |
| RPi5 candidate | 0.987 vs 1.000 | about equal |
| ESP32 candidate | equal | equal; slightly more abstention (0.197 vs 0.173) |

**Each model was trained once (seed 42).**
- The differences of 1–3 points are not separable from seed-to-seed variance, which this study did not measure. Earlier rounds found the dc_motor training seed-sensitive.
- Even so, no tier is better with the RK45 data. The promoted models were trained on the Euler data, and they also score higher on the RK45 plant (REF columns).
- The discretization effect of S-16/S-18 is real in the signals, but it does not make the Euler-trained models worse on the more accurate plant.

**Recommendation (decision for the user):** keep the promoted models. If the RK45 data should become the training standard, compare several seeds per dataset before any promotion.
