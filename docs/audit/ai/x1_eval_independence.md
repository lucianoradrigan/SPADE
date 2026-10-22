# X1 — independent samples behind the evaluations of the promoted 5-class classifier

Command: `python experiments/report_x1_eval_independence.py`. Counts only; nothing was re-measured. Raw values: [`x1_eval_independence.json`](x1_eval_independence.json).

The scenario seed changes the vibration channels and the sensor-noise realisation, **not** the simulated current, rpm or torque. A column with 1 distinct trajectory means every seed of that condition and class gave the same signal.

| Cache | Condition | Class | Seeds | Windows in the report (all classes) | Distinct current_r | Distinct rpm | Distinct torque |
|---|---|---|---|---|---|---|---|
| classifier_evaluation_5class_w2048 | extrapolated_speed_250_rad_s | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | extrapolated_speed_250_rad_s | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | extrapolated_speed_250_rad_s | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | extrapolated_speed_250_rad_s | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | extrapolated_speed_250_rad_s | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_100pct | ball | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_100pct | cage | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_100pct | inner_race | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_100pct | normal | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_100pct | outer_race | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_20pct | ball | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_20pct | cage | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_20pct | inner_race | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_20pct | normal | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | noise_20pct | outer_race | 4 | 80 | 4 | 1 | 1 |
| classifier_evaluation_5class_w2048 | nominal_steady_state | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | nominal_steady_state | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | nominal_steady_state | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | nominal_steady_state | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | nominal_steady_state | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | ood_unseen_fault_type | ball | 4 | — | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | ood_unseen_fault_type | cage | 4 | — | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | plant_drift_psi-10pct_J+50pct | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | plant_drift_psi-10pct_J+50pct | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | plant_drift_psi-10pct_J+50pct | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | plant_drift_psi-10pct_J+50pct | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | plant_drift_psi-10pct_J+50pct | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x0.5 | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x0.5 | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x0.5 | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x0.5 | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x0.5 | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x2 | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x2 | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x2 | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x2 | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | severity_x2 | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_100_rad_s | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_100_rad_s | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_100_rad_s | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_100_rad_s | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_100_rad_s | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_200_rad_s | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_200_rad_s | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_200_rad_s | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_200_rad_s | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | speed_200_rad_s | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_severity_x0.6 | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_severity_x0.6 | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_severity_x0.6 | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_severity_x0.6 | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_severity_x0.6 | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_115_rad_s | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_115_rad_s | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_115_rad_s | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_115_rad_s | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_115_rad_s | outer_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_190_rad_s | ball | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_190_rad_s | cage | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_190_rad_s | inner_race | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_190_rad_s | normal | 4 | 80 | 1 | 1 | 1 |
| classifier_evaluation_5class_w2048 | unseen_speed_190_rad_s | outer_race | 4 | 80 | 1 | 1 | 1 |
| AI-N14 benign shifts | benign_noise_150pct | inner_race | 2 | — | 2 | 1 | 1 |
| AI-N14 benign shifts | benign_noise_150pct | normal | 2 | — | 2 | 1 | 1 |
| AI-N14 benign shifts | benign_noise_150pct | outer_race | 2 | — | 2 | 1 | 1 |
| AI-N14 benign shifts | benign_speed_300_rad_s | inner_race | 2 | — | 1 | 1 | 1 |
| AI-N14 benign shifts | benign_speed_300_rad_s | normal | 2 | — | 1 | 1 | 1 |
| AI-N14 benign shifts | benign_speed_300_rad_s | outer_race | 2 | — | 1 | 1 | 1 |
| AI-N14 benign shifts | benign_speed_50_rad_s | inner_race | 2 | — | 1 | 1 | 1 |
| AI-N14 benign shifts | benign_speed_50_rad_s | normal | 2 | — | 1 | 1 | 1 |
| AI-N14 benign shifts | benign_speed_50_rad_s | outer_race | 2 | — | 1 | 1 | 1 |
