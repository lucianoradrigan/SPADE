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

### `configs/classifiers/pc_server/2026-10-18_run02`

Recorded at training time: test accuracy 0.838, macro-F1 0.834 (n_test = 870 windows).

Preprocessing: rpm level z-scored per window. Calibration: temperature T = 1.099, abstain below 0.544 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 228 | 0.939 | 0.939 | 0.938 | 1.00 | 1.00 | 0.82 | 0.000 | 0.000 | 0.025 | 0.95 / 0.62 | 0.03 / 0.950 / 0.21 |
| startup_transient | 108 | 0.676 | 0.676 | 0.651 | 0.67 | 1.00 | 0.36 | 0.000 | 0.264 | 0.070 | 0.84 / 0.49 | 0.38 / 0.881 / 0.77 |
| speed_100_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.082 | 0.92 / - | 0.00 / 1.000 / - |
| speed_200_rad_s | 228 | 0.750 | 0.750 | 0.709 | 1.00 | 1.00 | 0.25 | 0.000 | 0.000 | 0.180 | 0.98 / 0.77 | 0.02 / 0.753 / 0.04 |
| severity_x0.5 | 228 | 0.728 | 0.728 | 0.674 | 1.00 | 1.00 | 0.18 | 0.000 | 0.000 | 0.209 | 0.95 / 0.76 | 0.01 / 0.726 / 0.00 |
| severity_x2 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.001 | 1.00 / - | 0.00 / 1.000 / - |
| noise_20pct | 228 | 0.838 | 0.838 | 0.827 | 1.00 | 1.00 | 0.51 | 0.000 | 0.000 | 0.100 | 0.96 / 0.70 | 0.03 / 0.842 / 0.05 |
| noise_100pct | 228 | 0.789 | 0.789 | 0.770 | 0.97 | 1.00 | 0.39 | 0.000 | 0.000 | 0.114 | 0.96 / 0.59 | 0.05 / 0.815 / 0.17 |
| plant_drift_psi-10pct_J+50pct | 228 | 0.943 | 0.943 | 0.943 | 1.00 | 1.00 | 0.83 | 0.000 | 0.000 | 0.027 | 0.95 / 0.58 | 0.04 / 0.968 / 0.46 |
| unseen_speed_115_rad_s | 228 | 0.991 | 0.991 | 0.991 | 1.00 | 1.00 | 0.97 | 0.000 | 0.000 | 0.054 | 0.94 / 0.73 | 0.00 / 0.991 / 0.00 |
| unseen_speed_190_rad_s | 228 | 0.829 | 0.829 | 0.817 | 1.00 | 1.00 | 0.49 | 0.000 | 0.000 | 0.104 | 0.98 / 0.72 | 0.02 / 0.839 / 0.08 |
| extrapolated_speed_250_rad_s | 228 | 0.684 | 0.684 | 0.593 | 1.00 | 1.00 | 0.05 | 0.000 | 0.000 | 0.301 | 1.00 / 0.96 | 0.00 / 0.684 / 0.00 |
| unseen_severity_x0.6 | 228 | 0.746 | 0.746 | 0.702 | 1.00 | 1.00 | 0.24 | 0.000 | 0.000 | 0.177 | 0.98 / 0.73 | 0.02 / 0.754 / 0.05 |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | inner_race 0.12, normal 0.00, outer_race 0.88 | 0.708 | 0.04 | 0.11 |
| cage | 76 | inner_race 0.00, normal 0.37, outer_race 0.63 | 0.856 | 0.59 | 0.09 |

Confusion matrix, nominal steady state (rows = true ['inner_race', 'normal', 'outer_race'], columns = predicted): `[[76, 0, 0], [0, 76, 0], [14, 0, 62]]`

### `configs/classifiers/pc_server/2026-10-13_run01`

Recorded at training time: test accuracy 0.960, macro-F1 0.960 (n_test = 150 windows).

Preprocessing: rpm level z-scored per window. Uncalibrated softmax.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - | - |
| startup_transient | 108 | 0.778 | 0.778 | 0.778 | 0.67 | 1.00 | 0.67 | 0.000 | 0.278 | 0.057 | 0.84 / 0.40 | - |
| speed_100_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.324 | 0.99 / 1.00 | - |
| speed_200_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | - |
| severity_x0.5 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.500 | 0.331 | 1.00 / 1.00 | - |
| severity_x2 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.023 | 0.98 / - | - |
| noise_20pct | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.027 | 0.97 / - | - |
| noise_100pct | 228 | 0.561 | 0.561 | 0.472 | 1.00 | 0.68 | 0.00 | 0.316 | 0.000 | 0.390 | 0.82 / 0.89 | - |
| plant_drift_psi-10pct_J+50pct | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.019 | 0.98 / - | - |
| unseen_speed_115_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.331 | 1.00 / 1.00 | - |
| unseen_speed_190_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.001 | 1.00 / - | - |
| extrapolated_speed_250_rad_s | 228 | 0.333 | 0.333 | 0.222 | 0.00 | 1.00 | 0.00 | 0.000 | 0.500 | 0.596 | 1.00 / 0.89 | - |
| unseen_severity_x0.6 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.019 | 0.98 / - | - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | inner_race 0.00, normal 0.00, outer_race 1.00 | 0.990 | 1.00 | - |
| cage | 76 | inner_race 0.05, normal 0.32, outer_race 0.63 | 0.807 | 0.38 | - |

Confusion matrix, nominal steady state (rows = true ['inner_race', 'normal', 'outer_race'], columns = predicted): `[[76, 0, 0], [0, 76, 0], [0, 0, 76]]`

### `configs/classifiers/pc_server/2026-09-01_run01`

Recorded at training time: test accuracy 0.895, macro-F1 0.894 (n_test = 38 windows).

Preprocessing: rpm level z-scored per window. Uncalibrated softmax.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.167 | 0.52 / 0.45 | - |
| startup_transient | 108 | 0.500 | 0.500 | 0.415 | 0.00 | 0.61 | 0.89 | 0.389 | 0.069 | 0.110 | 0.48 / 0.40 | - |
| speed_100_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.164 | 0.52 / 0.46 | - |
| speed_200_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.178 | 0.52 / 0.43 | - |
| severity_x0.5 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.500 | 0.210 | 0.52 / 0.51 | - |
| severity_x2 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.166 | 0.53 / 0.45 | - |
| noise_20pct | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.221 | 0.46 / 0.43 | - |
| noise_100pct | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.469 | 0.41 / 0.40 | - |
| plant_drift_psi-10pct_J+50pct | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.167 | 0.53 / 0.45 | - |
| unseen_speed_115_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.163 | 0.53 / 0.46 | - |
| unseen_speed_190_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.174 | 0.52 / 0.43 | - |
| extrapolated_speed_250_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.500 | 0.159 | 0.52 / 0.48 | - |
| unseen_severity_x0.6 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.168 | 0.52 / 0.45 | - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 76 | inner_race 0.00, normal 0.00, outer_race 1.00 | 0.460 | 0.00 | - |
| cage | 76 | inner_race 0.00, normal 0.16, outer_race 0.84 | 0.427 | 0.00 | - |

Confusion matrix, nominal steady state (rows = true ['inner_race', 'normal', 'outer_race'], columns = predicted): `[[0, 0, 76], [0, 76, 0], [0, 0, 76]]`
