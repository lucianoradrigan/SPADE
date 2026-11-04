# AI-N27 option 2 — candidates trained on the `dcm` ×1000 grids vs the promoted classifiers

Pre-registration: [`ai_n27_regen_preregistration.md`](ai_n27_regen_preregistration.md) (committed before any data existed). Command: `python experiments/report_ai_n27_regen.py --candidates PC3=configs/classifiers/pc_server/2026-11-04_run01 ESP32=configs/classifiers/esp32_dscnn/2026-11-04_run01 PC5=configs/classifiers/pc_server_5class_w2048/2026-11-04_run01 RPI5=configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01`. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13.

**Nothing is promoted.** Meeting the criterion is not a promotion; the decision is Luciano's.

## Verdict per tier (pre-registered rule §6)

| Tier | Artifact | Verdict | Failing checks |
|---|---|---|---|
| PC | keras | **MEETS THE CRITERION** | — |
| RPi5 | tflite | **MEETS THE CRITERION** | — |
| ESP32 | tflite | **DOES NOT MEET THE CRITERION** | gate_D1_macro_f1 (0.72); gate_D1_min_recall (0.38); E1_delta_macro_f1 (-18.06); E1_worst_delta_recall (-51.03, outer_race); E2a_delta_macro_f1 (-27.95); E2a_worst_delta_recall (-65.14, outer_race); E2a_delta_abstention (-8.66); E3_delta_macro_f1 (-31.08); E3_worst_delta_recall (-78.27, outer_race) |

Δ = candidate − promoted, in points. Gate D1 on E1: candidate macro-F1 ≥ 0.88 and recall ≥ 0.70 per class. On E1, E2a, E3: Δ macro-F1 ≥ −1, Δ recall ≥ −5 per class, |Δ abstention| ≤ 5.

## PC

Promoted `configs/classifiers/pc_server_5class_w2048/2026-10-20_run01` vs candidate `configs/classifiers/pc_server_5class_w2048/2026-11-04_run01` (keras; candidate dataset `data/diagnosis_grid_dataset_5class_dcm.parquet`).

| Set | Acq. | Windows | Macro-F1 prom. → cand. (Δ, 95 % CI) | Accuracy prom. → cand. | Abstention prom. → cand. | Worst Δ recall (class) |
|---|---|---|---|---|---|---|
| **E1** | 50 | 600 | 0.891 → 0.930 (+3.9, [-0.5, +9.1]) | 0.892 → 0.930 | 0.098 → 0.115 | -3.3 (outer_race) |
| E2b | 13 | 156 | 0.683 → 0.695 (+1.2, [-8.9, +11.6]) | 0.795 → 0.904 | 0.090 → 0.135 | -16.7 (outer_race) |
| **E2a** | 90 | 1080 | 0.888 → 0.929 (+4.1, [+1.0, +7.7]) | 0.890 → 0.928 | 0.094 → 0.120 | -4.2 (outer_race) |
| E2a_startup | 90 | 180 | 0.627 → 0.630 (+0.3, [-4.8, +5.5]) | 0.622 → 0.628 | 0.450 → 0.633 | -19.4 (inner_race) |
| E2a_steady_state | 90 | 810 | 0.931 → 0.989 (+5.8, [+2.4, +9.7]) | 0.935 → 0.989 | 0.021 → 0.017 | -3.7 (outer_race) |
| E3_nominal_steady_state | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_startup_transient | 20 | 40 | 0.633 → 0.574 (-5.9, [-14.2, +0.0]) | 0.600 → 0.550 | 0.600 → 0.600 | -25.0 (normal) |
| E3_speed_100_rad_s | 20 | 80 | 0.893 → 1.000 (+10.7, [+6.7, +16.7]) | 0.900 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_speed_200_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_severity_x0.5 | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_severity_x2 | 20 | 80 | 0.723 → 1.000 (+27.7, [+21.5, +37.0]) | 0.800 → 1.000 | 0.138 → 0.000 | +0.0 (ball) |
| E3_noise_20pct | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_noise_100pct | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_plant_drift_psi-10pct_J+50pct | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_unseen_speed_115_rad_s | 20 | 80 | 0.949 → 1.000 (+5.1, [+2.9, +9.5]) | 0.950 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_unseen_speed_190_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_extrapolated_speed_250_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_unseen_severity_x0.6 | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| **E3** | 260 | 1000 | 0.955 → 0.982 (+2.7, [+1.1, +4.7]) | 0.956 → 0.982 | 0.035 → 0.024 | -1.0 (normal) |

Per-class recall (decision sets):

| Set | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| E1 | 0.950 → 0.958 | 0.758 → 0.950 | 0.925 → 0.917 | 0.925 → 0.958 | 0.900 → 0.867 |
| E2a | 0.935 → 0.968 | 0.699 → 0.944 | 0.931 → 0.898 | 0.935 → 0.921 | 0.949 → 0.907 |
| E3 | 1.000 → 1.000 | 0.840 → 0.980 | 0.980 → 0.980 | 0.980 → 0.970 | 0.980 → 0.980 |
| E2b | 0.917 → 0.917 | 0.719 → 0.958 | 1.000 → 1.000 | – → – | 0.889 → 0.722 |

## RPi5

Promoted `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01` vs candidate `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01` (tflite; candidate dataset `data/diagnosis_grid_dataset_5class_dcm.parquet`).

| Set | Acq. | Windows | Macro-F1 prom. → cand. (Δ, 95 % CI) | Accuracy prom. → cand. | Abstention prom. → cand. | Worst Δ recall (class) |
|---|---|---|---|---|---|---|
| **E1** | 50 | 600 | 0.870 → 0.925 (+5.5, [+1.6, +10.7]) | 0.870 → 0.925 | 0.098 → 0.115 | -2.5 (outer_race) |
| E2b | 13 | 156 | 0.580 → 0.754 (+17.4, [+2.6, +24.3]) | 0.737 → 0.929 | 0.109 → 0.128 | -5.6 (outer_race) |
| **E2a** | 90 | 1080 | 0.851 → 0.924 (+7.3, [+4.3, +10.6]) | 0.851 → 0.924 | 0.090 → 0.122 | -0.9 (outer_race) |
| E2a_startup | 90 | 180 | 0.535 → 0.594 (+5.9, [+0.8, +11.1]) | 0.544 → 0.594 | 0.428 → 0.600 | -5.6 (outer_race) |
| E2a_steady_state | 90 | 810 | 0.918 → 0.990 (+7.2, [+4.2, +11.0]) | 0.919 → 0.990 | 0.031 → 0.023 | +0.0 (inner_race) |
| E3_nominal_steady_state | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_startup_transient | 20 | 40 | 0.467 → 0.467 (+0.0, [+0.0, +0.0]) | 0.500 → 0.500 | 0.500 → 0.600 | +0.0 (ball) |
| E3_speed_100_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_speed_200_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.050 → 0.000 | +0.0 (ball) |
| E3_severity_x0.5 | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_severity_x2 | 20 | 80 | 0.723 → 1.000 (+27.7, [+21.5, +37.1]) | 0.800 → 1.000 | 0.138 → 0.000 | +0.0 (ball) |
| E3_noise_20pct | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_noise_100pct | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_plant_drift_psi-10pct_J+50pct | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_unseen_speed_115_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_unseen_speed_190_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_extrapolated_speed_250_rad_s | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| E3_unseen_severity_x0.6 | 20 | 80 | 1.000 → 1.000 (+0.0, [+0.0, +0.0]) | 1.000 → 1.000 | 0.000 → 0.000 | +0.0 (ball) |
| **E3** | 260 | 1000 | 0.964 → 0.980 (+1.6, [+0.4, +3.4]) | 0.964 → 0.980 | 0.035 → 0.024 | +0.0 (ball) |

Per-class recall (decision sets):

| Set | ball | cage | inner_race | normal | outer_race |
|---|---|---|---|---|---|
| E1 | 0.908 → 0.925 | 0.733 → 0.950 | 0.900 → 0.900 | 0.900 → 0.967 | 0.908 → 0.883 |
| E2a | 0.759 → 0.921 | 0.745 → 0.926 | 0.880 → 0.898 | 0.921 → 0.935 | 0.949 → 0.940 |
| E3 | 1.000 → 1.000 | 0.900 → 0.980 | 0.980 → 0.980 | 0.960 → 0.960 | 0.980 → 0.980 |
| E2b | 0.583 → 0.917 | 0.677 → 0.958 | 0.917 → 1.000 | – → – | 0.889 → 0.833 |

Keras (descriptive, not deciding): E1 macro-F1 0.870 → 0.925; E2a macro-F1 0.851 → 0.924; E3 macro-F1 0.964 → 0.980.

## ESP32

Promoted `configs/classifiers/esp32_dscnn/2026-10-18_run01` vs candidate `configs/classifiers/esp32_dscnn/2026-11-04_run01` (tflite; candidate dataset `data/diagnosis_grid_dataset_dcm.parquet`).

| Set | Acq. | Windows | Macro-F1 prom. → cand. (Δ, 95 % CI) | Accuracy prom. → cand. | Abstention prom. → cand. | Worst Δ recall (class) |
|---|---|---|---|---|---|---|
| **E1** | 30 | 870 | 0.899 → 0.718 (-18.1, [-28.3, -8.2]) | 0.899 → 0.744 | 0.166 → 0.151 | -51.0 (outer_race) |
| E2b | 0 | 0 | – | – | – | – |
| **E2a** | 45 | 2160 | 0.843 → 0.563 (-27.9, [-35.3, -19.1]) | 0.843 → 0.669 | 0.240 → 0.153 | -65.1 (outer_race) |
| E2a_startup | 45 | 405 | 0.705 → 0.516 (-18.9, [-27.0, -9.1]) | 0.701 → 0.615 | 0.447 → 0.200 | -64.4 (outer_race) |
| E2a_steady_state | 45 | 1755 | 0.875 → 0.583 (-29.2, [-36.5, -20.4]) | 0.877 → 0.688 | 0.195 → 0.148 | -63.4 (outer_race) |
| E3_nominal_steady_state | 12 | 228 | 1.000 → 0.627 (-37.3, [-50.0, -29.0]) | 1.000 → 0.702 | 0.000 → 0.018 | -89.5 (outer_race) |
| E3_startup_transient | 12 | 108 | 0.732 → 0.406 (-32.6, [-40.5, -18.0]) | 0.731 → 0.519 | 0.324 → 0.065 | -69.4 (outer_race) |
| E3_speed_100_rad_s | 12 | 228 | 1.000 → 0.635 (-36.5, [-49.4, -27.4]) | 1.000 → 0.706 | 0.004 → 0.048 | -88.2 (outer_race) |
| E3_speed_200_rad_s | 12 | 228 | 1.000 → 0.742 (-25.8, [-38.5, -16.7]) | 1.000 → 0.772 | 0.000 → 0.276 | -68.4 (outer_race) |
| E3_severity_x0.5 | 12 | 228 | 1.000 → 0.556 (-44.4, [-57.1, -35.6]) | 1.000 → 0.667 | 0.000 → 0.000 | -100.0 (outer_race) |
| E3_severity_x2 | 12 | 228 | 1.000 → 0.789 (-21.1, [-33.3, -14.9]) | 1.000 → 0.807 | 0.000 → 0.061 | -57.9 (outer_race) |
| E3_noise_20pct | 12 | 228 | 1.000 → 0.610 (-39.0, [-51.5, -29.3]) | 1.000 → 0.693 | 0.000 → 0.022 | -92.1 (outer_race) |
| E3_noise_100pct | 12 | 228 | 1.000 → 0.556 (-44.4, [-57.1, -35.6]) | 1.000 → 0.667 | 0.329 → 0.035 | -100.0 (outer_race) |
| E3_plant_drift_psi-10pct_J+50pct | 12 | 228 | 1.000 → 0.619 (-38.1, [-50.1, -29.0]) | 1.000 → 0.697 | 0.000 → 0.035 | -90.8 (outer_race) |
| E3_unseen_speed_115_rad_s | 12 | 228 | 1.000 → 0.575 (-42.5, [-55.6, -32.4]) | 1.000 → 0.675 | 0.000 → 0.022 | -97.4 (outer_race) |
| E3_unseen_speed_190_rad_s | 12 | 228 | 1.000 → 0.838 (-16.2, [-27.0, -10.8]) | 1.000 → 0.846 | 0.000 → 0.083 | -46.1 (outer_race) |
| E3_extrapolated_speed_250_rad_s | 12 | 228 | 1.000 → 0.956 (-4.4, [-10.5, -1.2]) | 1.000 → 0.956 | 0.000 → 0.136 | -13.2 (outer_race) |
| E3_unseen_severity_x0.6 | 12 | 228 | 1.000 → 0.556 (-44.4, [-57.1, -35.6]) | 1.000 → 0.667 | 0.000 → 0.009 | -100.0 (outer_race) |
| **E3** | 156 | 2844 | 0.990 → 0.679 (-31.1, [-35.8, -25.8]) | 0.990 → 0.730 | 0.039 → 0.062 | -78.3 (outer_race) |

Per-class recall (decision sets):

| Set | inner_race | normal | outer_race |
|---|---|---|---|
| E1 | 0.879 → 0.859 | 0.924 → 0.990 | 0.893 → 0.383 |
| E2a | 0.879 → 0.963 | 0.951 → 0.999 | 0.699 → 0.047 |
| E3 | 0.987 → 0.983 | 0.994 → 1.000 | 0.988 → 0.206 |

Keras (descriptive, not deciding): E1 macro-F1 0.900 → 0.875; E2a macro-F1 0.840 → 0.709; E3 macro-F1 0.989 → 0.961.

Unseen fault types (ball, cage; descriptive): abstention 0.434 → 0.500.

Teacher (PC 3-class, no tier, no criterion): `configs/classifiers/pc_server/2026-11-04_run01`, recorded test accuracy 0.921, macro-F1 0.921 (n = 870).

## Post-hoc analysis (descriptive; does not change the verdict)

ESP32 candidate, outer_race recall, promoted → candidate:

| Set | Keras | int8 TFLite (deployed) |
|---|---|---|
| E1 | 0.862 → 0.762 | 0.893 → 0.383 |
| E2a | 0.664 → 0.314 | 0.699 → 0.047 |
| E3 | 0.983 → 0.899 | 0.988 → 0.206 |

The failure has two parts:
1. The distilled Keras model is already weaker on outer_race than the promoted one.
2. The int8 quantization (calibrated on 1000 windows of the dcm grid, without `--calibration-noise-free`, as pre-registered) makes it much worse.

Following §7, nothing was retried. Possible next steps, each needing a new pre-registration and Luciano's decision:
- int8 calibration on noise-free runs (`--calibration-noise-free`, A2 follow-up);
- a different distillation alpha;
- training the ESP32 without distillation.

