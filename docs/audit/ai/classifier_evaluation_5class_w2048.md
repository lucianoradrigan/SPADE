### `configs/classifiers/pc_server_5class_w2048/2026-10-20_run01`

Recorded at training time: test accuracy 0.903, macro-F1 0.903 (n_test = 600 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 0.754, abstain below 0.525 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| startup_transient | 40 | 0.600 | 0.600 | 0.633 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.500 | 0.000 | 0.060 | 0.77 / 0.28 | 0.60 / 1.000 / 1.00 |
| speed_100_rad_s | 80 | 0.963 | 0.963 | 0.962 | 0.81 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.075 | 0.90 / 0.45 | 0.20 / 1.000 / 1.00 |
| speed_200_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| severity_x2 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| noise_20pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 | 1.00 / - | 0.00 / 1.000 / - |
| noise_100pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 | 1.00 / - | 0.00 / 1.000 / - |
| plant_drift_psi-10pct_J+50pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_speed_190_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.006 | 0.99 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 16 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 1.000 | 1.00 | 0.0 |
| cage | 16 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 1.000 | 1.00 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[16, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 16, 0], [0, 0, 0, 0, 16]]`

### `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01`

Recorded at training time: test accuracy 0.902, macro-F1 0.902 (n_test = 600 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 0.674, abstain below 0.508 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.006 | 0.99 / - | 0.00 / 1.000 / - |
| startup_transient | 40 | 0.500 | 0.500 | 0.467 | 1.00 | 0.50 | 0.50 | 0.00 | 0.50 | 1.000 | 0.000 | 0.143 | 0.86 / 0.43 | 0.50 / 0.800 / 0.80 |
| speed_100_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.128 | 0.87 / - | 0.20 / 1.000 / - |
| speed_200_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.072 | 0.93 / - | 0.09 / 1.000 / - |
| severity_x0.5 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.004 | 1.00 / - | 0.00 / 1.000 / - |
| severity_x2 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.030 | 0.97 / - | 0.00 / 1.000 / - |
| noise_20pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| noise_100pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 1.00 / - | 0.00 / 1.000 / - |
| plant_drift_psi-10pct_J+50pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.032 | 0.97 / - | 0.00 / 1.000 / - |
| unseen_speed_190_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.046 | 0.95 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.001 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.004 | 1.00 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 16 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.991 | 1.00 | 0.0 |
| cage | 16 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.992 | 1.00 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[16, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 16, 0], [0, 0, 0, 0, 16]]`

### `configs/classifiers/esp32_dscnn_5class_w2048/2026-10-20_run01`

Recorded at training time: test accuracy 0.790, macro-F1 0.792 (n_test = 600 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 0.795, abstain below 0.366 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.137 | 0.86 / - | 0.00 / 1.000 / - |
| startup_transient | 40 | 0.500 | 0.500 | 0.461 | 0.50 | 0.50 | 1.00 | 0.00 | 0.50 | 1.000 | 0.000 | 0.180 | 0.74 / 0.38 | 0.50 / 0.800 / 0.80 |
| speed_100_rad_s | 80 | 0.963 | 0.963 | 0.962 | 1.00 | 0.81 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.307 | 0.67 / 0.36 | 0.29 / 1.000 / 1.00 |
| speed_200_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.080 | 0.92 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.136 | 0.86 / - | 0.00 / 1.000 / - |
| severity_x2 | 80 | 0.900 | 0.900 | 0.893 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.101 | 0.88 / 0.60 | 0.00 / 0.900 / 0.00 |
| noise_20pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.116 | 0.88 / - | 0.00 / 1.000 / - |
| noise_100pct | 80 | 0.863 | 0.863 | 0.844 | 0.31 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.080 | 0.92 / 0.52 | 0.00 / 0.863 / 0.00 |
| plant_drift_psi-10pct_J+50pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.134 | 0.87 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 80 | 0.750 | 0.750 | 0.705 | 0.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.231 | 0.79 / 0.52 | 0.12 / 0.771 / 0.20 |
| unseen_speed_190_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.085 | 0.91 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.022 | 0.98 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.139 | 0.86 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 16 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.624 | 0.00 | 0.0 |
| cage | 16 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.983 | 1.00 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[16, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 16, 0], [0, 0, 0, 0, 16]]`

### `configs/classifiers/pc_server_5class/2026-10-20_run01`

Recorded at training time: test accuracy 0.821, macro-F1 0.823 (n_test = 1450 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 1.515, abstain below 0.308 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.061 | 0.94 / - | 0.00 / 1.000 / - |
| startup_transient | 180 | 0.661 | 0.661 | 0.677 | 0.67 | 0.78 | 0.61 | 0.58 | 0.67 | 0.417 | 0.042 | 0.059 | 0.81 / 0.31 | 0.42 / 0.923 / 0.87 |
| speed_100_rad_s | 380 | 0.803 | 0.803 | 0.735 | 1.00 | 0.01 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.179 | 0.83 / 0.44 | 0.04 / 0.819 / 0.12 |
| speed_200_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.040 | 0.96 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.065 | 0.94 / - | 0.00 / 1.000 / - |
| severity_x2 | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.016 | 0.98 / - | 0.00 / 1.000 / - |
| noise_20pct | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.041 | 0.96 / - | 0.00 / 1.000 / - |
| noise_100pct | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.021 | 0.98 / - | 0.00 / 1.000 / - |
| plant_drift_psi-10pct_J+50pct | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.060 | 0.94 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 380 | 0.811 | 0.811 | 0.756 | 1.00 | 0.05 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.080 | 0.99 / 0.43 | 0.01 / 0.821 / 0.07 |
| unseen_speed_190_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.041 | 0.96 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.008 | 0.99 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.065 | 0.93 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.998 | 1.00 | 0.0 |
| cage | 76 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.753 | 0.33 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[76, 0, 0, 0, 0], [0, 76, 0, 0, 0], [0, 0, 76, 0, 0], [0, 0, 0, 76, 0], [0, 0, 0, 0, 76]]`
