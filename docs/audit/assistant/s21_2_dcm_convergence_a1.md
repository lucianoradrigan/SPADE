# S21-2, amendment 1 — convergence of `"dcm"` between ×1000 and ×10000

**Simulated. Pre-registered** in [`ai_n27_preregistration_amendment1.md`](ai_n27_preregistration_amendment1.md), committed before this ran. The original criterion at ×100 vs ×1000 was not met in 2 of 6 cases ([`s21_2_dcm_convergence.md`](s21_2_dcm_convergence.md)); that result stands.

- **Command:** `python experiments/report_s21_2_dcm_convergence_a1.py`. Raw values: [`s21_2_dcm_convergence_a1.json`](s21_2_dcm_convergence_a1.json).
- **Criterion (threshold unchanged):** NRMS(×1000 vs ×10000) < 0.0001 for ω and i_a in every case → **CONVERGED**.

| Case | ×100 vs ×1000 (original; ω / i_a) | ×1000 vs ×10000 (ω / i_a) | Converged | Final speed ×10000 (ref) | CPU s ×10000 |
|---|---|---|---|---|---|
| 1_pmg132_120_no_load | 1.5e-06 / 2.8e-05 | 1.5e-07 / 2.8e-06 | yes | 120.000 (120) | 1399 |
| 2_stock_D1_friction_120 | 1.5e-06 / 3.3e-05 | 1.5e-07 / 3.3e-06 | yes | 120.000 (120) | 1341 |
| 3_outer_race_x2_80 | 4.0e-05 / 5.0e-04 | 4.1e-06 / 5.9e-05 | yes | 77.786 (80) | 3161 |
| 4_cage_x2_80 | 1.9e-05 / 2.0e-04 | 2.0e-06 / 2.3e-05 | yes | 83.634 (80) | 3177 |
| 5_control_healthy_220 | 6.8e-07 / 5.4e-06 | 6.8e-08 / 5.4e-07 | yes | 220.000 (220) | 1445 |
| 6_control_load5_120 | 7.4e-07 / 5.4e-06 | 7.4e-08 / 5.4e-07 | yes | 120.000 (120) | 1391 |

×1000 re-run deterministically: same final speed as in the original run in every case: yes.
