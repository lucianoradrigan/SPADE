# F0.1 — configurable DC-motor load (`Scenario.load_torque_nm`)

**Simulated evidence only.** No measurement of a real motor is involved.

- **Command:** `python experiments/report_f0_1_load_torque.py`. Raw values: [`f0_1_load_torque.json`](f0_1_load_torque.json).
- **Setup:** stock `dc_perm_ex_v1` (GEM PMG 132 parameters), speed PI, healthy bearing, 3 s per run; steady state = last 0.2 s. Rated torque 16 N·m (GEM nominal). The PI caps its current reference at 0.8 × 210 A = 168 A.

## Is `PolynomialStaticLoad`'s `a` a passive load torque? (point 2)

Read in `sim/loads.py`: the static load is `T = sign(ω)·c·ω² + b·ω + a_s`, with `a_s = sign(ω)·a` for |ω| > ω_lim and `a_s = (j_total/τ_decay)·ω` below it, ω_lim = a·τ_decay/j_total, τ_decay = 1 ms. So:

- it always opposes the rotation and is zero at standstill, so it never drives the shaft (passive, Coulomb-type);
- the linear band is continuous at ω_lim (both branches give `a` there) and keeps the Euler step stable: its time constant is τ_decay = 1 ms = 10 control periods;
- the band grows with `a`, but stays small up to 100 % load: 0 N·m → 0.000 rad/s, 4 N·m → 0.146 rad/s, 8 N·m → 0.291 rad/s, 12 N·m → 0.437 rad/s, 16 N·m → 0.582 rad/s;
- `BearingFaultLoad` delegates `set_j_rotor` to the wrapped load, so the band is recomputed with the real inertia.

**Conclusion:** adequate for loads up to 100 % of rated torque; implemented as `a = 0.01 + load_torque_nm`. It is not an active load (a hoist or a regenerating load would need a different model), and it does not ramp within a run (A1).

## Load levels

| ω_ref (rad/s) | Load (N·m, % rated) | Final speed (rad/s) | Steady error (rad/s) | Steady mean current (A) | Current the load needs (A) | At the 168 A cap in steady state |
|---|---|---|---|---|---|---|
| 150 | 0 (0 %) | 150.00 | 0.000 | 45.5 | 45.5 | no |
| 150 | 4 (25 %) | 150.00 | 0.000 | 69.8 | 69.8 | no |
| 150 | 8 (50 %) | 150.00 | 0.000 | 94.0 | 94.0 | no |
| 150 | 12 (75 %) | 150.00 | 0.000 | 118.2 | 118.2 | no |
| 150 | 16 (100 %) | 150.00 | 0.000 | 142.5 | 142.5 | no |
| 300 | 0 (0 %) | 300.00 | 0.000 | 91.0 | 91.0 | no |
| 300 | 4 (25 %) | 300.00 | 0.000 | 115.2 | 115.2 | no |
| 300 | 8 (50 %) | 300.00 | 0.000 | 139.5 | 139.5 | no |
| 300 | 12 (75 %) | 300.00 | -0.000 | 163.7 | 163.7 | no |
| 300 | 16 (100 %) | 233.20 | 67.007 | 168.0 | 187.9 | **yes** |

## Reading

- Wherever the needed current is below the cap, the speed PI holds the reference and the steady current equals (a + b·ω + load)/ψ_e (also asserted in `tests/test_assistant_f0_load_torque.py`, 0.1 % tolerance).
- Every run saturates the current reference during the start-up from standstill; that is the PI's normal large-step behaviour, not the load.
- **100 % at 300 rad/s:** the load needs 188 A, above the 168 A cap. The motor cannot hold the reference: with the current at the cap the speed tends to (ψ_e·168 − a − load)/b = 234 rad/s (final speed 233.2 rad/s, still changing at 2.2 rad/s²). Cause: the stock load's viscous term b·ω already takes 15 N·m at this speed, so rated torque on top of it needs about twice the 97 A rated current. It is the current limit at work (210 A limit, PI margin 0.8), not a defect.

## Not executed

- A load ramp within a run (`load_profile`): no ejecutado, Phase 3 (A1).
- Active (overhauling) loads: no ejecutado, outside the model.
