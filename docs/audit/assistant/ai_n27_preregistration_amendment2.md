# AI-N27 — pre-registration, amendment 2 (Luciano, 2026-11-03): reconstruction precondition

**Status.** Committed **before** the impact study runs and before any classifier metric is computed. It amends §2.2 ("Reconstruction check") of [`ai_n27_preregistration.md`](ai_n27_preregistration.md); everything else (and [amendment 1](ai_n27_preregistration_amendment1.md)) stays as written.

## What was observed (before any classifier output)

The check rebuilt the default plant for 6 acquisitions (the first 3 affected of each grid) and compared it with the stored parquet:

| Channel | Max relative difference over the 6 runs |
|---|---|
| `current_r`, `rpm`, `voltage_v`, `torque_nm` (plant) | **0** (bit-identical in all 6) |
| `acc_x` | 5.2e-16 – 1.5e-15 |
| `acc_y` | 2.6e-16 – 8.5e-16 |
| `acc_z` | 7.5e-16 – 2.5e-15 |

The original text required "exactly", so the precondition failed as written and the study did not run.

**Probable cause** (cheap check, not exhaustive): not the thread count — the same difference with `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` and with the default; not the code — `src/driveflow/sim/vibration` is unchanged since 2026-09-01. The grids were generated on 2026-10-18 (the run metadata of that week record Python 3.11 / numpy 2.4.6); today's environment has Python 3.12.13 / numpy 2.5.4. The most likely cause is a different summation order in a numerical library version (last bit of a float64; ε ≈ 2.2e-16). Not investigated further, as decided.

## Amended precondition

- **Plant channels** (`current_r`, `rpm`, `voltage_v`, `torque_nm`): still **bit-identical** to the stored data.
- **Vibration channels** (`acc_x`, `acc_y`, `acc_z`): max relative difference (max |rebuilt − stored| / max |stored|) **≤ 1e-12**.
- Checked on the same 6 acquisitions. If either condition fails, the study stops and is reported.
