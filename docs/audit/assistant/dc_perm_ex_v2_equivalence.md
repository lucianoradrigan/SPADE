# dc_perm_ex_v2 step 1 — dataset equivalence of variant B (pre-registration §6)

**Simulated.** Command: `python experiments/report_dc_perm_ex_v2_equivalence.py`. Raw: `dc_perm_ex_v2_equivalence.json`. **Verdict: MET.**

(a) Per run, NRMS(B vs stored dcm ×1000) ≤ 0.001 for current_r, rpm, torque_nm: **met**.

| Run | NRMS current_r | NRMS rpm | NRMS torque_nm | Pass |
|---|---|---|---|---|
| grid_outer_race_w80_s1.00_seed1020 | 1.2e-05 | 9.4e-07 | 1.2e-05 | yes |
| grid_outer_race_w80_s1.50_seed1029 | 1.4e-05 | 1.5e-06 | 2.5e-05 | yes |
| grid_outer_race_w80_s2.00_seed1038 | 4.7e-05 | 3.4e-06 | 4.7e-05 | yes |
| grid_outer_race_w100_s1.50_seed1074 | 1.1e-05 | 1.3e-06 | 1.7e-05 | yes |
| grid_outer_race_w100_s2.00_seed1083 | 2.8e-05 | 3.3e-06 | 6.6e-05 | yes |
| grid_normal_w80_s0.00_seed1001 | 1.6e-07 | 1.2e-07 | 3.3e-07 | yes |
| grid_normal_w125_s0.00_seed1091 | 1.6e-07 | 8.5e-08 | 3.3e-07 | yes |
| grid_normal_w150_s0.00_seed1136 | 2.8e-07 | 7.3e-08 | 3.1e-07 | yes |
| grid_normal_w200_s0.00_seed1226 | 1.8e-07 | 6.8e-08 | 5.0e-07 | yes |
| grid_normal_w220_s0.00_seed1271 | 3.0e-07 | 6.4e-08 | 5.4e-07 | yes |

(b) Top-1 agreement ≥ 0.99 of the promoted classifiers, stored ×1000 vs B: **met**.

| Model | Artifact | Windows | Agreement | Accuracy stored / B (descriptive) |
|---|---|---|---|---|
| PC (`configs/classifiers/pc_server_5class_w2048/2026-11-04_run01`) | keras | 120 | 1.0000 | 0.917 / 0.917 |
| RPi5 (`configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01`) | tflite | 120 | 1.0000 | 0.958 / 0.958 |

**Declared:** the rule "the first fault acquisition of each affected cell" picked outer_race in all five affected cells, because of the class order of `build_jobs`. inner_race is not among the 10 runs. The 120 windows are all the windows of the 10 runs (known classes, no cap).

**Consequence:** the models trained on dcm ×1000 data stay valid for data simulated with variant B. Step 2 (the `dc_perm_ex_v2` plant) and decision E3 (v2 as the default of new simulations) can rest on this evidence; E3 is confirmed by Luciano.
