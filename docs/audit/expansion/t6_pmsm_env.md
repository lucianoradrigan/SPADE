# T6: SPADE/Pmsm-v0

Round 2026-10-21, task T6 (decision D10). Branch `feat/E1-pmsm-env` from `dev` (6ef9d13), worked in a parallel worktree. Machine: Linux x86-64, Python 3.12.13.

## What was built

- **`envs/native.PmsmEnv`, with the E1 pattern.**
  - **Plant:** `control.classical.pmsm_foc.build_pmsm_system()`, the one `datagen.runner` simulates, **unchanged**.
  - **Solver:** forward Euler with `EULER_NSTEPS` substeps per τ = 100 µs. It is 10 since S-13 (2026-10-20). The round document says 3, which was the S-01 value; the solver was not touched.
  - **Seeding:** with `SeedSequence(seed)`, like the runner.
  - **Action:** the normalized dq converter voltage [u_d, u_q] ∈ [−1, 1]², which is what `DqCurrentController` sends to the plant. Actions are clipped and logged (`action_applied`, `action_clipped`); a NaN action is rejected.
  - **Observation:** [i_d, i_q, i_d_ref, i_q_ref] / I_LIM (400 A), and ω / 418.9 rad/s.
  - **Reference:** a stator-current magnitude i_s and a strategy. `mtpa` uses the closed-form MTPA locus (FOC_MTPA); `naive` sets i_d = 0 (FOC_naive). Both exactly as the runner. Reset options: `current_ref_a`.
  - **Reward:** the E1 terms (tracking of the dq current error, overcurrent, action rate).
  - **Episode end:** terminated if |i_s| > 1.25 × I_LIM or the state is not finite; truncated at the horizon (0.03 s = 300 steps by default).
  - **For the FOC controller:** `plant`, `plant_state`, `i_d_ref` and `i_q_ref` are exposed.
- **Registry:** `SPADE/Pmsm-v0` moved from PLANNED to ENVIRONMENTS, with the reference policy `DqCurrentController`. The `native:pmsm_v1` profile now declares stepping and reset; RL training is not declared (E8 decides).
- **Environments page:** the PMSM row comes from the real contract, with a snippet and a live contract check. The only planned row left is "PLECS plant environments".

## Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| `check_env` passes | `tests/envs/test_native_envs.py::test_check_env_passes[SPADE/Pmsm-v0]` and the other ENV_IDS-parametrized contract tests | PASS |
| `step()` reproduces the existing simulation bit for bit, with the FOC/MTPA actions | `test_pmsm_env_reproduces_run_scenario_bit_for_bit` (MTPA 250 A seed 3; naive −120 A seed 0; MTPA 390 A seed 7). i_d, i_q, ω, u_d and u_q are equal (==) to `run_scenario` at every one of the 300 steps, and so are the references | PASS |
| Physics and solver unchanged (S-01, S-13) | no file under `sim/` or `control/classical/pmsm_foc.py` was modified | PASS |
| Registered and shown in Environments | `tests/test_environments_page.py` (updated: the PMSM is no longer the "not registered" example) | PASS |

Tests changed because the decision changes what is true:
- `tests/runs/test_store.py`: the stepping set now includes `native:pmsm_v1`;
- `tests/test_environments_page.py`: the unregistered example is now the PLECS row.
No threshold was changed.

Full suite (on dev with E8 and T7): **1070 passed, 4 xfailed (strict, findings), 0 failed** (827 s) — `t6_branch_pytest.txt`.
