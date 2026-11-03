# S21-2 — convergence of the discontinuous-conduction converter model (`dc_converter_model="dcm"`)

**Simulated. Pre-registered** in [`ai_n27_preregistration.md`](ai_n27_preregistration.md) §1, committed before this ran.

- **Command:** `python experiments/report_s21_2_dcm_convergence.py`. Raw values: [`s21_2_dcm_convergence.json`](s21_2_dcm_convergence.json).
- **Criterion:** NRMS(×100 vs ×1000) < 0.0001 for ω and i_a in every case → **NOT CONVERGED**.
- **Control:** "dcm" ×N vs "gem" with N substeps (N = 1, 100), NRMS ≤ 1e-09, where GEM's current is never negative → **PASS**.

## Convergence

| Case | NRMS ×10 vs ×1000 (ω / i_a) | NRMS ×100 vs ×1000 (ω / i_a) | Converged | Final speed GEM → DCM ×1000 (ref) | Min current GEM → DCM | GEM vs DCM ×1000 (ω / i_a) |
|---|---|---|---|---|---|---|
| 1_pmg132_120_no_load | 1.6e-05 / 3.0e-04 | 1.5e-06 / 2.8e-05 | yes | 179.43 → 120.00 (120) | -161.2 → 0.0 A | 4.05e-01 / 5.53e-01 |
| 2_stock_D1_friction_120 | 1.7e-05 / 3.4e-04 | 1.5e-06 / 3.3e-05 | yes | 196.84 → 120.00 (120) | -163.1 → 0.0 A | 4.88e-01 / 5.39e-01 |
| 3_outer_race_x2_80 | 4.9e-04 / 6.1e-03 | 4.0e-05 / 5.0e-04 | **no** | 126.01 → 77.79 (80) | -103.5 → 0.0 A | 5.44e-01 / 6.04e-01 |
| 4_cage_x2_80 | 2.8e-04 / 2.5e-03 | 1.9e-05 / 2.0e-04 | **no** | 145.57 → 83.63 (80) | -127.2 → 0.0 A | 4.74e-01 / 6.64e-01 |
| 5_control_healthy_220 | 7.5e-06 / 5.9e-05 | 6.8e-07 / 5.4e-06 | yes | 220.00 → 220.00 (220) | 66.3 → 66.3 A | 8.51e-05 / 6.35e-04 |
| 6_control_load5_120 | 8.2e-06 / 5.9e-05 | 7.4e-07 / 5.4e-06 | yes | 120.00 → 120.00 (120) | 66.3 → 66.2 A | 9.08e-05 / 6.13e-04 |

## Control (no artifact)

| Case | GEM negative-current samples | DCM ×1 vs GEM ×1 (ω / i_a) | DCM ×100 vs GEM ×100 (ω / i_a) | Pass | Info: DCM ×100 vs default plant (ω / i_a) |
|---|---|---|---|---|---|
| 5_control_healthy_220 | 0 | 7.5e-15 / 1.3e-13 | 8.6e-15 / 1.5e-13 | yes | 8.4e-05 / 6.3e-04 |
| 6_control_load5_120 | 0 | 3.1e-15 / 1.5e-14 | 2.8e-15 / 2.8e-14 | yes | 9.0e-05 / 6.1e-04 |

The last column is S-16's discretization error (one Euler step vs 100), not the converter model; it is reported, not decided on.

## Cost (CPU s of one run)

| Case | Default plant | DCM ×100 | DCM ×1000 |
|---|---|---|---|
| 1_pmg132_120_no_load | 6.8 | 18.0 | 144.4 |
| 2_stock_D1_friction_120 | 6.8 | 25.3 | 161.2 |
| 3_outer_race_x2_80 | 17.3 | 75.1 | 334.9 |
| 4_cage_x2_80 | 17.4 | 70.5 | 334.1 |
| 5_control_healthy_220 | 6.3 | 23.9 | 152.4 |
| 6_control_load5_120 | 6.7 | 26.4 | 160.0 |
