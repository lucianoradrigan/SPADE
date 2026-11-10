# B4 preconditions (pre-registration §2)

**Simulated.** Command: `python experiments/b4_preconditions.py`. Raw: `b4_preconditions.json`.

## Q1: the generator reproduces the stored 5-class dcm grid with `dc_perm_ex_v2` — **pass**

NRMS ≤ 0.0001 per channel; noise-free runs: min `current_r` ≥ −1e-6 A.

| Run | Noise-free | NRMS current_r | NRMS rpm | NRMS torque_nm | min current_r (A) | Pass |
|---|---|---|---|---|---|---|
| `grid_normal_w80_s0.00_seed1001` | False | 1.65e-07 | 1.17e-07 | 3.35e-07 | -63 | True |
| `grid_inner_race_w150_s1.00_seed1263` | True | 2.81e-05 | 9.99e-07 | 2.81e-05 | 33.8 | True |
| `grid_cage_w220_s2.00_seed1525` | False | 3.33e-05 | 1.54e-06 | 7.43e-05 | -102 | True |

## Q3: held-out seeds do not overlap — **pass**

| Set | First | Last |
|---|---|---|
| b4_grid_5class | 80001 | 80675 |
| b4_grid_3class | 80001 | 80405 |
| offgrid | 85001 | 85060 |
| e4_5class | 90001 | 90060 |
| e4_3class | 90001 | 90036 |
| e5 | 95001 | 95030 |
| earlier_grids | 1001 | 1675 |
| evaluation | 200 | 203 |

Overlapping seeds between a held-out set and any other set: e4_3class~b4_grid_5class 0, e4_3class~b4_grid_3class 0, e4_3class~offgrid 0, e4_3class~earlier_grids 0, e4_3class~evaluation 0, e4_5class~b4_grid_5class 0, e4_5class~b4_grid_3class 0, e4_5class~offgrid 0, e4_5class~earlier_grids 0, e4_5class~evaluation 0, e5~b4_grid_5class 0, e5~b4_grid_3class 0, e5~offgrid 0, e5~earlier_grids 0, e5~evaluation 0

## Q2

Checked on the windows of each trained candidate (`report_b4.py`).
