### `configs/classifiers/pc_server/2026-09-01_run01`

Recorded at training time: test accuracy 0.895, macro-F1 0.894 (n_test = 38 windows).

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect |
|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.167 | 0.52 / 0.45 |
| startup_transient | 108 | 0.500 | 0.500 | 0.415 | 0.00 | 0.61 | 0.89 | 0.389 | 0.069 | 0.110 | 0.48 / 0.40 |
| speed_100_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.164 | 0.52 / 0.46 |
| speed_200_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.178 | 0.52 / 0.43 |
| severity_x0.5 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.500 | 0.210 | 0.52 / 0.51 |
| severity_x2 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.166 | 0.53 / 0.45 |
| noise_20pct | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.221 | 0.46 / 0.43 |
| noise_100pct | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.469 | 0.41 / 0.40 |
| plant_drift_psi-10pct_J+50pct | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.167 | 0.53 / 0.45 |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 |
|---|---|---|---|---|
| ball | 76 | inner_race 0.00, normal 0.00, outer_race 1.00 | 0.460 | 0.00 |
| cage | 76 | inner_race 0.00, normal 0.16, outer_race 0.84 | 0.427 | 0.00 |

Confusion matrix, nominal steady state (rows = true ['inner_race', 'normal', 'outer_race'], columns = predicted): `[[0, 0, 76], [0, 76, 0], [0, 0, 76]]`

### `configs/classifiers/pc_server/2026-10-13_run01`

Recorded at training time: test accuracy 0.960, macro-F1 0.960 (n_test = 150 windows).

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect |
|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.026 | 0.97 / - |
| startup_transient | 108 | 0.778 | 0.778 | 0.778 | 0.67 | 1.00 | 0.67 | 0.000 | 0.278 | 0.057 | 0.84 / 0.40 |
| speed_100_rad_s | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.324 | 0.99 / 1.00 |
| speed_200_rad_s | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - |
| severity_x0.5 | 228 | 0.667 | 0.667 | 0.556 | 0.00 | 1.00 | 1.00 | 0.000 | 0.500 | 0.331 | 1.00 / 1.00 |
| severity_x2 | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.023 | 0.98 / - |
| noise_20pct | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.027 | 0.97 / - |
| noise_100pct | 228 | 0.561 | 0.561 | 0.472 | 1.00 | 0.68 | 0.00 | 0.316 | 0.000 | 0.390 | 0.82 / 0.89 |
| plant_drift_psi-10pct_J+50pct | 228 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.019 | 0.98 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 |
|---|---|---|---|---|
| ball | 76 | inner_race 0.00, normal 0.00, outer_race 1.00 | 0.990 | 1.00 |
| cage | 76 | inner_race 0.05, normal 0.32, outer_race 0.63 | 0.807 | 0.38 |

Confusion matrix, nominal steady state (rows = true ['inner_race', 'normal', 'outer_race'], columns = predicted): `[[76, 0, 0], [0, 76, 0], [0, 0, 76]]`
