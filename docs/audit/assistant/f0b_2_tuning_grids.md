# 0b D2 — tuning grids scaled to the base motor's ratings

**Simulated evidence; the tuned controller is a candidate. No default changed.** User decision of 2026-11-02.

- **Command:** `python experiments/report_0b_2_tuning_grids.py`. Raw values: [`f0b_2_tuning_grids.json`](f0b_2_tuning_grids.json).
- **Rule** (`optim/objectives.grids_for`):
  - speeds, speed steps and the speed-sensor noise × ω_rated / 300 rad/s;
  - load steps × T_rated / 16 N·m (300 rad/s and 16 N·m are the stock ratings);
  - the stock motor uses the module's grids unchanged (same objects);
  - a changed motor without rated speed and torque cannot be tuned (`MotorSpecError`, `INVALID_SPEC` in the job).
- **Stock motor, same numbers as before F0.6:** TUNING_GRID yes, EVALUATION_GRID yes, EVALUATION_GRID_NOISY yes.

## Grids of the 220 V motor (rated 157.1 rad/s, 28.65 N·m)

| Grid | Condition | Kind | Speed (rad/s) | Step (rad/s) | Load (N·m) | Drift |
|---|---|---|---|---|---|---|
| tuning | `s04_step_2rad_s_at_150_scaled` | step | 78.5 | 1.05 | 0.00 | no |
| tuning | `load_10Nm_at_150_scaled` | load | 78.5 | 0.00 | 17.91 | no |
| evaluation | `step_5rad_s_at_120_scaled` | step | 62.8 | 2.62 | 0.00 | no |
| evaluation | `saturating_step_30rad_s_at_100_scaled` | step | 52.4 | 15.71 | 0.00 | no |
| evaluation | `drift_step_2rad_s_at_150_scaled` | step | 78.5 | 1.05 | 0.00 | yes |
| evaluation | `drift_load_15Nm_at_200_scaled` | load | 104.7 | 0.00 | 26.86 | yes |

The fastest condition is now 104.7 rad/s (2/3 of 157 = the stock grid's 200 of 300 rad/s), within the 220 V motor's range; the noisy grid uses 0.314 rad/s of sensor noise.

## Short PSO on the scaled grids (8 evaluations, objective J, seed 0)

| Controller | Bandwidth (rad/s) | Damping | b | J tuning grid | J evaluation grid | Every evaluation run finite |
|---|---|---|---|---|---|---|
| baseline | 80.0 | 0.900 | 1.000 | 17.13 | 25.50 | True |
| candidate | 112.8 | 1.038 | 0.681 | 4.85 | 10.14 | True |

Baseline load-step conditions on the evaluation grid: `drift_load_15Nm_at_200_scaled` dip 5.68 rad/s, final error 0.000 rad/s.
With the absolute grids (F0.6) the 200 rad/s load condition ended 22.6 rad/s off because it exceeded the supply; on the scaled grid every condition is inside the motor's range.

## Not executed

- A full-budget tuning on the 220 V motor: no ejecutado (machinery only).
