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

### `configs/classifiers/rpi5_resnet1d_se_5class/2026-10-20_run01`

Recorded at training time: test accuracy 0.809, macro-F1 0.811 (n_test = 1450 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 1.076, abstain below 0.380 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.108 | 0.89 / - | 0.00 / 1.000 / - |
| startup_transient | 180 | 0.644 | 0.644 | 0.656 | 0.67 | 0.67 | 0.56 | 0.78 | 0.56 | 0.222 | 0.208 | 0.102 | 0.80 / 0.46 | 0.24 / 0.750 / 0.47 |
| speed_100_rad_s | 380 | 0.829 | 0.829 | 0.791 | 1.00 | 0.14 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.259 | 0.73 / 0.44 | 0.17 / 0.825 / 0.15 |
| speed_200_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.031 | 0.97 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.111 | 0.89 / - | 0.00 / 1.000 / - |
| severity_x2 | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.079 | 0.92 / - | 0.00 / 1.000 / - |
| noise_20pct | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.079 | 0.92 / - | 0.00 / 1.000 / - |
| noise_100pct | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.081 | 0.92 / - | 0.00 / 1.000 / - |
| plant_drift_psi-10pct_J+50pct | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.110 | 0.89 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 380 | 0.832 | 0.832 | 0.795 | 1.00 | 0.16 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.164 | 0.89 / 0.43 | 0.03 / 0.859 / 0.19 |
| unseen_speed_190_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.046 | 0.95 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.014 | 0.99 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.110 | 0.89 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.984 | 1.00 | 0.0 |
| cage | 76 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.775 | 0.03 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[76, 0, 0, 0, 0], [0, 76, 0, 0, 0], [0, 0, 76, 0, 0], [0, 0, 0, 76, 0], [0, 0, 0, 0, 76]]`

### `configs/classifiers/esp32_dscnn_5class/2026-10-20_run01`

Recorded at training time: test accuracy 0.741, macro-F1 0.738 (n_test = 1450 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 0.998, abstain below 0.377 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 380 | 0.932 | 0.932 | 0.930 | 0.99 | 0.67 | 1.00 | 1.00 | 1.00 | 0.000 | 0.082 | 0.126 | 0.85 / 0.45 | 0.02 / 0.944 / 0.19 |
| startup_transient | 180 | 0.622 | 0.622 | 0.642 | 0.67 | 0.44 | 0.56 | 0.89 | 0.56 | 0.111 | 0.361 | 0.135 | 0.78 / 0.50 | 0.13 / 0.692 / 0.29 |
| speed_100_rad_s | 380 | 0.787 | 0.787 | 0.717 | 0.93 | 0.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.147 | 0.72 / 0.35 | 0.37 / 0.966 / 0.90 |
| speed_200_rad_s | 380 | 0.997 | 0.997 | 0.997 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.000 | 0.003 | 0.108 | 0.89 / 0.41 | 0.00 / 0.997 / 0.00 |
| severity_x0.5 | 380 | 0.945 | 0.945 | 0.944 | 1.00 | 0.72 | 1.00 | 1.00 | 1.00 | 0.000 | 0.069 | 0.126 | 0.85 / 0.44 | 0.03 / 0.957 / 0.24 |
| severity_x2 | 380 | 0.968 | 0.968 | 0.968 | 0.89 | 0.95 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.145 | 0.83 / 0.62 | 0.03 / 0.973 / 0.17 |
| noise_20pct | 380 | 0.900 | 0.900 | 0.894 | 0.97 | 0.53 | 1.00 | 1.00 | 1.00 | 0.000 | 0.118 | 0.138 | 0.86 / 0.43 | 0.02 / 0.919 / 0.21 |
| noise_100pct | 380 | 0.811 | 0.811 | 0.799 | 0.57 | 0.49 | 1.00 | 1.00 | 1.00 | 0.000 | 0.099 | 0.095 | 0.84 / 0.50 | 0.03 / 0.832 / 0.14 |
| plant_drift_psi-10pct_J+50pct | 380 | 0.937 | 0.937 | 0.935 | 1.00 | 0.68 | 1.00 | 1.00 | 1.00 | 0.000 | 0.079 | 0.123 | 0.85 / 0.47 | 0.02 / 0.947 / 0.17 |
| unseen_speed_115_rad_s | 380 | 0.789 | 0.789 | 0.719 | 0.95 | 0.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.203 | 0.82 / 0.35 | 0.19 / 0.977 / 0.91 |
| unseen_speed_190_rad_s | 380 | 0.984 | 0.984 | 0.984 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 0.000 | 0.020 | 0.104 | 0.89 / 0.43 | 0.00 / 0.984 / 0.00 |
| extrapolated_speed_250_rad_s | 380 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.080 | 0.92 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 380 | 0.939 | 0.939 | 0.938 | 1.00 | 0.70 | 1.00 | 1.00 | 1.00 | 0.000 | 0.076 | 0.118 | 0.85 / 0.43 | 0.03 / 0.959 / 0.35 |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | ball 0.99, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.01 | 0.763 | 0.00 | 0.0 |
| cage | 76 | ball 0.00, cage 0.67, inner_race 0.00, normal 0.33, outer_race 0.00 | 0.613 | 0.09 | 0.09 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[75, 0, 0, 0, 1], [0, 51, 0, 25, 0], [0, 0, 76, 0, 0], [0, 0, 0, 76, 0], [0, 0, 0, 0, 76]]`

### `configs/classifiers/pc_server/2026-10-18_run01`

Recorded at training time: test accuracy 0.914, macro-F1 0.914 (n_test = 870 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 1.005, abstain below 0.594 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - | 0.00 / 1.000 / - |
| startup_transient | 108 | 0.815 | 0.815 | 0.810 | 0.78 | 1.00 | 0.67 | 0.000 | 0.139 | 0.152 | 0.81 / 0.56 | 0.33 / 0.889 / 0.60 |
| speed_100_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.040 | 0.96 / - | 0.00 / 1.000 / - |
| speed_200_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - | 0.00 / 1.000 / - |
| severity_x2 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - | 0.00 / 1.000 / - |
| noise_20pct | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.055 | 0.94 / - | 0.00 / 1.000 / - |
| noise_100pct | 228 | 0.969 | 0.969 | 0.969 | 1.00 | 0.91 | 1.00 | 0.092 | 0.000 | 0.111 | 0.87 / 0.53 | 0.16 / 0.990 / 0.71 |
| plant_drift_psi-10pct_J+50pct | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.060 | 0.94 / - | 0.00 / 1.000 / - |
| unseen_speed_190_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.000 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | inner_race 0.00, normal 0.00, outer_race 1.00 | 0.993 | 1.00 | 0.0 |
| cage | 76 | inner_race 0.28, normal 0.72, outer_race 0.00 | 0.591 | 0.00 | 0.54 |

Confusion matrix, nominal steady state (rows = true ['inner_race', 'normal', 'outer_race'], columns = predicted): `[[76, 0, 0], [0, 76, 0], [0, 0, 76]]`
