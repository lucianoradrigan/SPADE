# 0b D1 — default load scaled to the motor

**Simulated evidence only.** User decision of 2026-11-02 (rule chosen after the proportional option was shown to keep the 94 % friction below).

- **Command:** `python experiments/report_0b_1_load_default.py`. Raw values: [`f0b_1_load_default.json`](f0b_1_load_default.json).
- **Rule** (`datagen/runner.dc_default_load_parameter`, also used by the tuning through `BaseMotor.bench_kwargs`):
  - stock motor: the stock load, a = 0.01 N·m, b = 0.05 N·m·s/rad, unchanged bit for bit;
  - another motor (`motor_parameter_overrides` set) whose rated torque and speed are known (`dc_nominal_overrides` 'torque' and 'omega'): a + b·ω_rated = 2% of the rated torque, half constant (a) and half viscous (b·ω); j_load stays 0.0025 kg·m²;
  - another motor without those ratings keeps the stock load (as before; no rating is invented).
- **Why not proportional:** the stock friction at the stock rated speed is 15.01 N·m, **94% of the 16 N·m rated torque**. Scaling it would give every other motor a friction close to its rated torque.

## Case B (220 V motor) at rated load torque, reference = rated speed

Friction at 157 rad/s: 7.86 N·m with the stock load, 0.573 N·m with the rule (a = 0.2865 N·m, b = 0.001824 N·m·s/rad).

| Controller | Final speed, stock friction (rad/s) | Final speed, scaled friction (rad/s) | Below rated | Steady current (A) | Steady voltage (V) |
|---|---|---|---|---|---|
| PI | 148.04 | 156.37 | 0.45 % | 25.5 | 220.0 |
| MPC | 148.04 | 156.37 | 0.45 % | 25.5 | 220.0 |

The motor now carries rated load within 0.5 % of rated speed. It does not reach it exactly: ψ_e·ω_rated + R_a·I = 180.0 + 40.8 V exceeds the 220 V supply by a little, because ψ_e = 1.146 was derived from P/(ω·I) for case B rather than from the back-EMF. That is the motor data, not the controller.

Effect on earlier evidence: `f0_2_dc_supply_limits.md` shows case B with the stock friction (its rated-load rows); those numbers describe the behaviour before this decision.
