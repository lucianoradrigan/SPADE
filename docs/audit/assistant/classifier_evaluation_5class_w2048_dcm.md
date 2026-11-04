### `configs/classifiers/pc_server_5class_w2048/2026-11-04_run01`

Recorded at training time: test accuracy 0.930, macro-F1 0.930 (n_test = 600 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 1.062, abstain below 0.828 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.017 | 0.98 / - | 0.00 / 1.000 / - |
| startup_transient | 40 | 0.550 | 0.550 | 0.574 | 1.00 | 0.50 | 0.50 | 0.25 | 0.50 | 0.750 | 0.000 | 0.026 | 0.81 / 0.29 | 0.60 / 1.000 / 1.00 |
| speed_100_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.001 | 1.00 / - | 0.00 / 1.000 / - |
| speed_200_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.006 | 0.99 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.017 | 0.98 / - | 0.00 / 1.000 / - |
| severity_x2 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.020 | 0.98 / - | 0.00 / 1.000 / - |
| noise_20pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| noise_100pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.004 | 1.00 / - | 0.00 / 1.000 / - |
| plant_drift_psi-10pct_J+50pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.016 | 0.98 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.008 | 0.99 / - | 0.00 / 1.000 / - |
| unseen_speed_190_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.008 | 0.99 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.007 | 0.99 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.017 | 0.98 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 16 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 1.000 | 1.00 | 0.0 |
| cage | 16 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.999 | 1.00 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[16, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 16, 0], [0, 0, 0, 0, 16]]`

### `configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01`

Recorded at training time: test accuracy 0.925, macro-F1 0.925 (n_test = 600 windows).

Preprocessing: rpm level kept ({'rpm': [1432.4, 477.5]}). Calibration: temperature T = 0.828, abstain below 0.768 (fitted on validation). Metrics below use calibrated probabilities.

| Condition | n | Accuracy | Balanced acc. | Macro-F1 | Recall ball | Recall cage | Recall inner_race | Recall normal | Recall outer_race | False-alarm rate | Missed-detection rate | ECE | Conf. correct / incorrect | Abstained / acc. answered / errors withheld |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_steady_state | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.004 | 1.00 / - | 0.00 / 1.000 / - |
| startup_transient | 40 | 0.500 | 0.500 | 0.467 | 1.00 | 0.50 | 0.50 | 0.00 | 0.50 | 1.000 | 0.000 | 0.133 | 0.86 / 0.38 | 0.60 / 1.000 / 1.00 |
| speed_100_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.018 | 0.98 / - | 0.00 / 1.000 / - |
| speed_200_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.025 | 0.97 / - | 0.00 / 1.000 / - |
| severity_x0.5 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| severity_x2 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.006 | 0.99 / - | 0.00 / 1.000 / - |
| noise_20pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.002 | 1.00 / - | 0.00 / 1.000 / - |
| noise_100pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.005 | 0.99 / - | 0.00 / 1.000 / - |
| plant_drift_psi-10pct_J+50pct | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_speed_115_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.006 | 0.99 / - | 0.00 / 1.000 / - |
| unseen_speed_190_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.017 | 0.98 / - | 0.00 / 1.000 / - |
| extrapolated_speed_250_rad_s | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |
| unseen_severity_x0.6 | 80 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | 0.003 | 1.00 / - | 0.00 / 1.000 / - |

Unseen fault types (no class in the model's vocabulary):

| Fault | n | Predicted as | Mean confidence | Share conf. > 0.9 | Abstained |
|---|---|---|---|---|---|
| ball | 16 | ball 1.00, cage 0.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.995 | 1.00 | 0.0 |
| cage | 16 | ball 0.00, cage 1.00, inner_race 0.00, normal 0.00, outer_race 0.00 | 0.999 | 1.00 | 0.0 |

Confusion matrix, nominal steady state (rows = true ['ball', 'cage', 'inner_race', 'normal', 'outer_race'], columns = predicted): `[[16, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 16, 0], [0, 0, 0, 0, 16]]`
