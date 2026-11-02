# S-21 — non-physical speed rise when a one-quadrant DC drive is asked to brake

**Simulated evidence. Found in round Assistant, phase 1 (F1.6). Not fixed: needs Luciano's decision.**

- **Command:** `python experiments/report_s21_one_quadrant.py`. Raw values: [`s21_one_quadrant.json`](s21_one_quadrant.json).
- **Converter:** `sim/converters.ContOneQuadrantConverter._convert` returns duty 1 (full supply) when the current at the start of the period is negative; the converter output is held for the whole 0.1 ms period.

## Stock motor, speed PI at 120 rad/s, no extra load, three frictions (tuning bench `run_dc_closed_loop`)

| Friction | Peak speed | Final speed | Min current | Mean current after 0.2 s | Mean torque after 0.2 s | Mean u·i after 0.2 s | Periods at full voltage after 0.2 s |
|---|---|---|---|---|---|---|---|
| stock (a = 0.01, b = 0.05) | 120.4 rad/s | 120.0 rad/s | 34.4 A | 36.4 A | 6.01 N·m | 742 W | 0% |
| {'a': 0.16, 'b': 0.0005333333333333334, 'c': 0.0, 'j_load': 0.0025} | 196.8 rad/s | 196.8 rad/s | -163.1 A | 23.4 A | 3.86 N·m | 3522 W | 45% |
| {'a': 1.0, 'b': 0.01} | 120.8 rad/s | 120.0 rad/s | 9.1 A | 13.3 A | 2.20 N·m | 267 W | 0% |

With the stock friction (6 N·m at 120 rad/s) the start-up overshoot is braked by friction and the current never goes negative. With the 0b-D1 friction (a = 0.160 N·m, b = 0.000533 N·m·s/rad) the speed keeps rising after the overshoot: the current swings between about -160 A and positive values every period, and the periods held at full voltage (the converter's response to a negative current) leave a positive mean current, torque and electrical power while the controller asks for braking.

## `dc_pmg132_v1`, PI at 120 rad/s, no extra load: plant integration

| Integration | Peak speed | Final speed | Min current |
|---|---|---|---|
| Euler, one step per period (default) | 179.43 rad/s | 179.43 rad/s | -161.2 A |
| {'dc_plant_substeps': 10} | 179.40 rad/s | 179.40 rad/s | -156.4 A |
| {'dc_plant_substeps': 100} | 179.40 rad/s | 179.40 rad/s | -155.6 A |
| {'dc_plant_solver': 'rk45'} | 179.40 rad/s | 179.40 rad/s | -155.5 A |

The same result with 10 or 100 Euler substeps and with RK45: the plant integration converges, but the converter decision is still taken once per period, so this is **not** a converged reference for the switching behaviour (lesson G7 applies).

## `dc_pmg132_v1` at 120 rad/s with a little load

| Extra load | Peak speed | Final speed |
|---|---|---|
| 0 N·m | 179.4 rad/s | 179.4 rad/s |
| 1 N·m | 121.0 rad/s | 120.0 rad/s |
| 5 N·m | 120.5 rad/s | 120.0 rad/s |

## What it affects

- Any DC run where the speed PI asks for braking (overshoot, a lower reference, a load removed) and friction plus load are small: `dc_pmg132_v1` and every changed motor with known ratings (0b-D1 friction) at light load.
- Tuning on a changed motor (0b-D2): the evaluation numbers of such motors include this effect (F1.6, case B: the baseline "overshoot" of 2321 %).
- **S-19 is probably the same mechanism**, recorded on 2026-10-25 as physics ("the mean speed rises up to +72 %"): a one-quadrant drive cannot raise the speed above the reference by braking. To be re-checked.
- The stock motor with the stock friction in the shipped grids: not affected while the current stays positive (its 94 % friction brakes the overshoot).

## Not executed

- A reference with the converter decision re-evaluated inside the period (event-based diode conduction / discontinuous conduction): no ejecutado. Needed to prove the hypothesis and to size the error.
- Re-measurement of S-19 and of the 0b-D1 / 0b-D3 evidence: no ejecutado (waits for the decision).
