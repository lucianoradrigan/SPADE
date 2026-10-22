# E14 — noise-aware residual threshold: results (round 2026-10-22, T6)

Pre-registration: [`e14_noise_aware_preregistration.md`](e14_noise_aware_preregistration.md), commits 8615d31 and 0c76c85, both made before this run. Command: `python experiments/report_e14_noise_aware.py` (Linux x86_64, Python 3.12.13). Raw values: [`e14_noise_aware_results.json`](e14_noise_aware_results.json). Run: `configs/regressors/dc_motor_residual_pc_novv/2026-10-21_run02`; gate 411 ms. Simulated windows only.

**Verdict: NEGATIVE: criteria not met; E14 closes as a negative result, no further attempts (§6)**

## Calibration set (§3)

- Runs: 21 recorded validation runs and 21 new runs. Windows: 2520 gated.
- Electrical noise: 0–100 %.
- σ̂(current_r): 0–40.2 A. Median σ̂(rpm), not used: 0.

## Ripple-inflation check (§2)

- Data: 1920 healthy windows with noise ≥ 10 %, plus synthetic ripple of 97.0 A.
- Median relative inflation of σ̂: outer_race -0.00%, inner_race +0.00% (limit 10%).
- Result: **PASS**.

## Threshold τ(σ̂) (§4)

| Bin | σ̂ range (A) | Windows | τ (q99, monotone) |
|---|---|---|---|
| 1 | 0–5.16 | 630 | 1.340 |
| 2 | 5.16–11 | 315 | 2.293 |
| 3 | 11–15.3 | 315 | 2.982 |
| 4 | 15.3–18.8 | 315 | 3.477 |
| 5 | 18.8–24.4 | 315 | 4.290 |
| 6 | 24.4–28.7 | 315 | 4.850 |
| 7 | 28.7–40.2 | 315 | 5.603 |

Above σ̂ = 40.2 A the answer is "out of calibrated noise range". The run02 constant threshold was 4.349.

## Single evaluation on the E14 test (§5)

| Condition | Healthy windows | Out of range (healthy) | False alarms (in range) | AUROC outer / inner | Detection outer / inner (in range) | Criterion 3 |
|---|---|---|---|---|---|---|
| unseen_speed_115_rad_s | 156 | 0.0% | 0.0% | 1.000 / 1.000 | 100% / 100% | PASS |
| unseen_speed_190_rad_s | 156 | 0.0% | 0.0% | 1.000 / 1.000 | 100% / 100% | PASS |
| extrapolated_speed_250_rad_s | 156 | 0.0% | 0.0% | 1.000 / 1.000 | 100% / 100% | PASS |
| noise_100pct | 156 | 23.1% | 6.7% | 1.000 / 1.000 | 100% / 100% | FAIL |
| plant_drift_psi-10pct_J+50pct | 156 | 0.0% | 0.0% | 1.000 / 1.000 | 100% / 100% | PASS |

- **Criterion 2** (pooled AUROC ≥ 0.9, severity ×1): outer 1.000, inner 1.000. **PASS**.
- **Criterion 3** (≤ 1 % false alarms per condition, ≤ 10 % of healthy windows out of range): **FAIL**.
- Detection at severity ×0.6, in range, reported: outer 100%, inner 100%.

Commit hashes: the pre-registration was committed as eacddf7/e13d847 on the task branch and became 8615d31/0c76c85 when the branch was rebased onto `dev`. The author dates (2026-10-21 23:29) are unchanged and precede this run.
