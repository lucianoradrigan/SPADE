# F0.6 — PI tuning on another DC motor

**Simulated evidence; the tuned controller is a candidate. No default changed.**

- **Command:** `python experiments/report_f0_6_tuning_motor.py`. Raw values: [`f0_6_tuning_motor.json`](f0_6_tuning_motor.json). Numbers before the change: [`f0_6_stock_before.json`](f0_6_stock_before.json).
- **What changed:**
  - `optim/objectives.py`: a `BaseMotor` (motor parameters, supply, limits, nominal values, as the Scenario fields);
  - the drift conditions are now relative factors (`DRIFT_FACTORS`: ψ_e × 0.9, j_rotor × 1.5) of the base motor;
  - `evaluate`, `simulate_condition`, `j_effort` and `objective_functions` take the motor;
  - `validation/benches.run_dc_closed_loop` designs the controller on the base motor and accepts supply, limits and nominal values;
  - `controller_tuning` accepts an optional `params.motor` block, validated by `Scenario`.
- **Rated current of J_effort:** 97 A for the stock motor; otherwise `dc_nominal_overrides['i']`. With changed motor parameters and no rated current, J_effort is refused before any simulation (`MotorSpecError`, `INVALID_SPEC` in the job).

## 1. Stock motor: before vs after

| Quantity | Before | After | Identical |
|---|---|---|---|
| J of BASELINE on TUNING_GRID | 25.20494963878825 | 25.20494963878825 | yes |
| J of BASELINE on EVALUATION_GRID | 27.829457688832704 | 27.829457688832704 | yes |
| J of BASELINE on EVALUATION_GRID_NOISY | 43.38879097692182 | 43.38879097692182 | yes |
| J_effort of BASELINE on the tuning grid | 4.0 | — | yes |

Every per-condition metric (overshoot, settling, dip, effort, peak current; CPU time excluded) was compared too: identical, as JSON equality.

## 2. Case B, 220 V motor: short PSO tuning (8 evaluations, objective J, seed 0)

| Controller | Bandwidth (rad/s) | Damping | b | J tuning grid | J evaluation grid | Max peak step current (A) | ≤ 25 A rated |
|---|---|---|---|---|---|---|---|
| baseline | 80.0 | 0.900 | 1.000 | 31.58 | 69.89 | 40.0 | False |
| candidate | 152.2 | 1.090 | 0.302 | 4.44 | 49.53 | 39.6 | False |

## Reading

- With the stock motor the tuning is unchanged, number for number.
- The tuning machinery runs on another motor; the candidate is a short-budget search, not a recommended controller.
- **Limit (not fixed, needs a decision):** the grids' speeds, steps, loads and the 0.6 rad/s noise are absolute values sized for the stock motor (up to 200 rad/s). For the 220 V motor the `drift_load_15Nm_at_200` condition asks for 200 rad/s, above its 157 rad/s rating (back-EMF 1.146 × 200 = 229 V > 220 V supply): its final error there is 22.6 rad/s, so that condition measures the supply limit, not the controller.

## Not executed

- Grids scaled to the base motor's ratings: no ejecutado (it would change the tuning definition; decision pending).
- A full-budget tuning on the 220 V motor: no ejecutado (only the machinery was asked for).
