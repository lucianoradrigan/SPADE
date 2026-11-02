# F0.2 — DC supply voltage, motor limits and nominal values; controller-design check

**Simulated evidence only.**

- **Command:** `python experiments/report_f0_2_dc_supply_limits.py`. Raw values: [`f0_2_dc_supply_limits.json`](f0_2_dc_supply_limits.json).
- **New `Scenario` fields (dc_perm_ex_v1 only):** `dc_supply_v` (IdealVoltageSupply, default 65 V), `dc_limit_overrides` and `dc_nominal_overrides` (keys `omega`, `torque`, `i`, `u`). All three `None` takes the original code path; `tests/test_assistant_f0_dc_supply_limits.py` shows the explicit stock values (65 V, stock limits and nominals) are bit-identical to the default with PI and MPC.

## 1. Does GEM overwrite an explicit current limit?

`ElectricMotor._update_limits` fills a limit only when it is missing or 0; `DcPermanentlyExcitedMotor._update_limits` proposes `i = u / r_a` for that case. Measured:

| Constructor arguments | i limit (A) |
|---|---|
| `stock` | 210 |
| `limit_values={'i': 150}` | 150 |
| `motor_parameter={'r_a': 1.6}` | 210 |
| `limit_values={'i': 0}` | 3750 |

- An explicit `i` is **kept** as given; nothing had to change to respect it.
- `u / r_a` is used only for `i = 0` (3750 A here).
- Changing `r_a` (or any motor parameter) without limits keeps the stock 210 A: GEM does not derive the limit from the new motor. A caller with another motor must pass `dc_limit_overrides`; the future assistant will have to fill it from the datasheet. Documented in the `Scenario` docstring.
- Limits only normalise the state and set the controllers' current cap (PI 0.8 × i, MPC 0.85 × i); the simulator does not clip states with them.

## 2. Controller-design check (`control/design_check.py`)

Thresholds, declared in the module before any run: τ_m/τ_a ≥ 10; current-loop / speed-loop bandwidth ≥ 10; τ_a ≥ 2 control periods (Euler stability); MPC horizon ≥ τ_a.

| Motor | τ_a (ms) | τ_m (ms) | τ_m/τ_a | Bandwidth ratio | PI | MPC horizon / τ_a | MPC |
|---|---|---|---|---|---|---|---|
| stock (PMG 132 / GEM) | 1.19 | 16.2 | 13.6 | 62.5 | **ok** | 1.68 | **ok** |
| case B, 220 V | 10 | 64 | 6.4 | 62.5 | **warning** | 0.20 | **warning** |
| constructed: stock with l_a × 2 | 2.38 | 16.2 | 6.8 | 62.5 | **warning** | — | — |

- The stock motor passes with little margin (τ_m/τ_a = 13.6), consistent with the PI module's "~100x" comment being optimistic: it is about 14×.
- **The MPC depends on the motor parameters**: its prediction model is built from r_a, l_a, psi_e and j_total (`control/mpc/controller.py`), with a fixed 20-step (2 ms) horizon sized for the stock τ_a.

## 3. Case B: 220 V motor, reference = rated speed 1500 rpm

| Controller | Load torque (N·m) | Final speed (rad/s) | Steady error (rad/s) | Overshoot | t to 95 % (s) | Peak / steady current (A) | Steady voltage (V) |
|---|---|---|---|---|---|---|---|
| PI | 0.00 | 157.08 | 0.000 | 0.31 % | 0.19 | 40.0 / 6.9 | 191.0 |
| MPC | 0.00 | 156.89 | 0.186 | 0.00 % | 0.178 | 42.6 / 6.9 | 190.8 |
| PI | 28.65 | 148.04 | 9.041 | 0.00 % | None | 40.0 / 31.5 | 220.0 (= supply) |
| MPC | 28.65 | 148.04 | 9.041 | 0.00 % | None | 42.6 / 31.5 | 220.0 (= supply) |

## Reading

- **Without extra load, both controllers reach rated speed** despite the `warning`: PI error 0.000 rad/s and 0.31 % overshoot; MPC 0.186 rad/s (its known offset, no integrator). The check is conservative here; it flags that the design assumptions do not hold, not that the run fails. So the instruction's S-20 condition (the 220 V motor not reaching its reference) does not occur.
- **At rated load torque (28.65 N·m) neither reaches rated speed** (PI 148.0, MPC 148.0 rad/s; voltage at the 220 V supply). Holding 1500 rpm would need about 231 V: the stock load's friction (a + b·ω = 7.9 N·m at this speed) adds to the rated torque, and back-EMF plus R_a·I exceeds the supply. This is the plant doing what its numbers say (the stock friction model was sized for the 60 V PMG 132, not for this motor), not a controller defect. Recorded as an observation for the assistant: with another motor, the default load friction is not neutral.

## Not executed

- Changing either controller: no ejecutado (out of scope; the check is diagnostic only).
- A friction model scaled to the motor: no ejecutado; it would change a default (§7 of the round instructions).
