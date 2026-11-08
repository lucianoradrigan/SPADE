# dc_perm_ex_v2 step 1 — convergence of variant B (`dcm_exact`)

**Simulated. Pre-registered** in `dc_perm_ex_v2_convergence_preregistration.md` and its amendment 1, both committed before any code of B.

- **Command:** `python experiments/report_dc_perm_ex_v2_convergence.py`. Raw values: `dc_perm_ex_v2_convergence.json`. Host: Linux-7.0.0-43-generic-x86_64-with-glibc2.43, Python 3.12.13.
- **REF** = `dcm` ×10000 (the converged reference, S21-2 amendment 1); **D1000** = `dcm` ×1000; **B** = `dcm_exact`.
- **Verdict for B: MEETS THE CRITERION.**

| Case | B vs REF (ω / i_a) | D1000 vs REF (ω / i_a) | B vs D1000 (ω / i_a) | min i_a B (A) | final ω B / REF (ref) | CPU s per sim s B / D1000 |
|---|---|---|---|---|---|---|
| 1_pmg132_120_no_load | 4.0e-07 / 9.7e-06 | 1.5e-07 / 2.8e-06 | 5.4e-07 / 1.2e-05 | 0 | 120.000 / 120.000 (120) | 3.31 / 92.4 |
| 2_stock_D1_friction_120 | 4.2e-07 / 1.1e-05 | 1.5e-07 / 3.3e-06 | 5.7e-07 / 1.4e-05 | 0 | 120.000 / 120.000 (120) | 3.25 / 93.0 |
| 3_outer_race_x2_80 | 1.9e-06 / 3.0e-05 | 4.1e-06 / 5.9e-05 | 5.2e-06 / 6.7e-05 | 0 | 77.787 / 77.786 (80) | 3.35 / 110.9 |
| 4_cage_x2_80 | 4.1e-07 / 1.3e-05 | 2.0e-06 / 2.3e-05 | 2.3e-06 / 2.7e-05 | 0 | 83.634 / 83.634 (80) | 3.37 / 110.9 |
| 5_control_healthy_220 | 2.9e-08 / 1.1e-06 | 6.8e-08 / 5.4e-07 | 9.6e-08 / 6.9e-07 | 66.3 | 220.000 / 220.000 (220) | 3.24 / 91.2 |
| 6_control_load5_120 | 3.0e-07 / 1.5e-06 | 7.4e-08 / 5.4e-07 | 3.7e-07 / 1.3e-06 | 66.2 | 120.000 / 120.000 (120) | 3.24 / 91.4 |

| Check (§4) | Result |
|---|---|
| 1. NRMS(B vs REF) < 0.0001, ω and i_a, every case | met |
| 2. Controls: NRMS(B vs D1000) ≤ 0.0001 | met |
| 3. min i_a ≥ -1e-06 A | met |
| 4. Median CPU per simulated second ≥ 10× lower than D1000 (measured 28.3×) | met |

## Deviation (declared)

- The first run of this script (2026-11-08) stopped with an exception: `dcm_exact: too many conduction segments in one period`. It was not a bad metric.
  - **Cause:** at the end of a blocked segment, e(t_on) = u up to rounding was re-tested as blocked again and again.
  - **Fix:** `be843fb` → next commit. After a blocked segment that ends inside the period, conduction resumes without re-testing the rounding. The test `test_blocked_segment_ending_inside_the_period_resumes_conduction` covers it.
  - The definition of variant B (amendment 1) did not change, and no result of that first run exists: it crashed before any comparison.
- The script was then run once more, with the jobs of B first. This report is that run.
- **Cost measurement:** CPU time per process (`time.process_time`), with 6 processes and the machine otherwise idle.

## Next (pre-registration §6)

Dataset equivalence on 10 grid runs, then decision E3 (v2 as the default for new simulations), which Luciano confirms with this evidence.
