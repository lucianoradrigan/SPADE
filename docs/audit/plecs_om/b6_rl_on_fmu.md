# B6 — RL Lab and Tuning Studio on SPADE/OmBuck-v0

Command: `python experiments/report_b6_rl_on_fmu.py `  
Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13. Total wall time 295.6 s.

Environments found by capability (no name list): RL `SPADE/Buck-v0`, `SPADE/BuckSurrogate-v0`, `SPADE/OmBuck-v0`, `SPADE/Vsc-v0`; tuning systems `dc_motor_pi`, `buck_pi:SPADE/Buck-v0`, `buck_pi:SPADE/BuckSurrogate-v0`, `buck_pi:SPADE/OmBuck-v0`.

## 1. DDPG on SPADE/OmBuck-v0 vs the buck PI (rl_training job, run `run-20261025T162826-8e7e178f`, completed, origin SIMULATED · OPENMODELICA, 289.2 s)

Grid: the job's buck grid without load step (the FMU's load R is fixed per episode): 24 V / 6 Ω; 12 V / 12 Ω; 30 V / 8 Ω; 18 V / 20 Ω.

| Controller | Mean episode reward | Median tracking RMSE, 2nd half (V) | Terminated |
|---|---|---|---|
| RL (DDPG) | -7.02 | 2.980 | 0/4 |
| PI (BuckVoltageController) | -5.01 | 0.000 | 0/4 |

## 2. Transfer between implementations (same policy, same grid)

| Controller | Plant | Mean episode reward | Median tracking RMSE, 2nd half (V) | Terminated |
|---|---|---|---|---|
| RL (DDPG) | `SPADE/OmBuck-v0` | -7.02 | 2.980 | 0/4 |
| PI (BuckVoltageController) | `SPADE/OmBuck-v0` | -5.01 | 0.000 | 0/4 |
| RL (DDPG) | `SPADE/Buck-v0` | -7.02 | 2.980 | 0/4 |
| PI (BuckVoltageController) | `SPADE/Buck-v0` | -5.01 | 0.000 | 0/4 |

Largest difference of the episode reward between the two implementations, per controller: RL (DDPG): 8.54e-05 absolute (6.3e-06 relative); PI (BuckVoltageController): 1.78e-04 absolute (9.6e-05 relative). The policy transfers because the two plants agree to ~1e-5 (B3 criterion 4, B4).

## 3. PSO tuning of the buck PI on SPADE/OmBuck-v0 (controller_tuning job, run `run-20261025T163317-23832c9d`, completed, 24 evaluations, 4.7 s)

Objective: J = mean tracking RMSE (% of v_ref, whole 20 ms episode from rest) + mean overshoot (% of v_ref). Lower is better. Tuning grid ['24V_6ohm', '12V_12ohm']; evaluation grid ['30V_8ohm', '18V_20ohm', '36V_4ohm', '8V_3ohm'] (distinct).

| | Current-loop bandwidth (Hz) | Voltage-loop bandwidth (Hz) |
|---|---|---|
| Baseline | 5000 | 1000 |
| Candidate | 6339 | 919 |

| Evaluation grid | Baseline J | Candidate J | Baseline mean RMSE % / overshoot % | Candidate mean RMSE % / overshoot % |
|---|---|---|---|---|
| SPADE/OmBuck-v0 (tuned on) | 16.146 | 15.197 | 8.693 / 7.453 | 8.682 / 6.515 |
| SPADE/Buck-v0 (native) | 16.146 | 15.197 | 8.693 / 7.453 | 8.682 / 6.515 |

Tuning-grid J on SPADE/OmBuck-v0: baseline 14.633 → candidate 14.664.

The PSO candidate is slightly worse than the baseline on the tuning grid it was optimized on (24 evaluations did not sample the baseline point) and better on the evaluation grid, on both implementations; reported as measured.

**Status:** the policy and the tuning candidate are candidates only; neither is used as a default (decisions D1/D4 of the round 2026-10-21 stay in force). SPADE/OmBuck-v0 replicates the native buck, so this demonstrates the FMU integration end to end, not a different plant.
