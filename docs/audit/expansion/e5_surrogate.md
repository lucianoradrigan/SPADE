# E5: buck surrogate, SPADE/BuckSurrogate-v0 and "Compare with SPADE"

Round 2026-10-21, task T4 (decision D3). Branch `feat/E5-surrogate` from `dev` (6a7d415). Machine: Linux x86-64, Python 3.12.13.

Report: [`docs/audit/plecs/e5_surrogate_and_comparison.md`](../plecs/e5_surrogate_and_comparison.md) / `.json`.

## Scope (D3)

- **Source:** the native buck (`native:buck_v1`), labelled **SIMULATED**, never `SIMULATED · PLECS`. **This validates the surrogate mechanics, not fidelity to PLECS.** No PLECS data exists.
- **Action:** the duty cycle. For PLECS circuits, the interface manifest documents that the duty is expected as a ModelVar `D` or as an input of the atomic subsystem (`integrations/plecs/manifest.py`).

## What was built

- **`envs/surrogate_env.py`:**
  - **Data:** open-loop duty excitation of the native buck over a 4 × 4 grid of V_in × R. Conditions are split, never windows: 12 train, 2 validation, 2 test, plus 2 extrapolation conditions outside the domain.
  - **Families:**
    - `zoh`: grey box. L, C and r_L by nonlinear least squares of the exact ZOH one-step prediction, the method of E10b, started at ×1.5 L, ×0.7 C, ×2 r_L.
    - `mlp`: a Keras 2 × 32 tanh network on (i_L, v_out, duty, V_in, log R) → one-step increment, seeded, run by a numpy forward pass.
  - **Validity domain:** the training ranges of V_in, R and duty. `domain_violations()` and `compare_condition()` produce the native-vs-surrogate comparison: same PI, same reference and load step, aligned time axes.
  - **Environment:** `BuckSurrogateEnv` keeps SPADE/Buck-v0's contract and reports `info["out_of_domain"]` at reset and at every step.
- **`experiments/fit_buck_surrogate.py`:**
  - fits on the train conditions only;
  - picks the default family on the **validation** conditions only (lower median relative v_out rollout RMSE);
  - writes `configs/surrogates/buck_surrogate/2026-10-21_run01/surrogate.json` and the report.
- **Environments:**
  - `SPADE/BuckSurrogate-v0` is registered with the label `SIMULATED · SURROGATE`;
  - the table has a "Fit error (validation)" column with the validity domain;
  - the contract tab warns "Surrogate, not the plant… outside that range every step reports info['out_of_domain']".
- **PLECS Plant Studio → Compare with SPADE:**
  - **Native vs. surrogate:** family, V_in, R, v_ref and load step → overlaid v_out and i_L on the same time axis, RMSE and max error, the "Surrogate outside validated operating range" warning (it "does not claim equivalence"), and a "failed" state for a non-finite surrogate.
  - **Condition grid:** every validation, test and extrapolation condition is listed, failed ones included.
  - **PLECS vs. SPADE:** still the empty state.

## Results

**ZOH grey box: exact, as expected.** It identifies L = 220.000 µH, C = 220.000 µF and r_L = 0.05000 Ω, the native values, from a wrong start. Rollout and closed-loop errors are therefore 0 everywhere, the extrapolation conditions included.

This is the expected outcome when the data come from the same noise-free model the grey box assumes. It is not evidence about a switched PLECS circuit, whose model mismatch this run cannot measure. The ZOH family is the default by the validation rule.

**MLP: a black box that generalises but drifts in open loop.** Relative v_out rollout RMSE / closed-loop v_out RMSE:

| Condition | Split | Rollout v_out | Closed-loop v_out |
|---|---|---|---|
| 36 V, 6 Ω | validation | 4.9 % | 0.023 V |
| 60 V, 24 Ω | validation | 27.1 % | 0.104 V |
| 24 V, 3 Ω | test | 4.2 % | 0.066 V |
| 48 V, 12 Ω | test | 7.4 % | 0.037 V |
| 72 V, 12 Ω | extrapolation | 9.4 % | 0.084 V |
| 48 V, 40 Ω | extrapolation | 56.2 % | 0.067 V |

In closed loop the PI corrects most of the drift: errors stay under about 0.1 V on a 0.5·V_in reference.

## Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| Linear ZOH and a small Keras network, both reported | `fit_buck_surrogate.py` and the report | PASS |
| Split by condition; error per channel and per condition; validity domain; out-of-domain warning | `test_split_is_by_condition_and_fixed`, `test_env_flags_out_of_domain_and_uses_the_surrogate[zoh, mlp]`, `test_compare_condition_reports_errors_and_domain`; report table | PASS |
| `SPADE/BuckSurrogate-v0` labelled SIMULATED · SURROGATE, fit error visible in Environments | `test_registered_and_contract` (check_env, describe); registry `verify` passes (check_env + determinism) | PASS |
| Compare with SPADE: native vs. surrogate, aligned axes, error per condition, failed cases visible, no claim of equivalence; PLECS comparison still empty | `tests/test_compare_with_spade.py` (3 AppTests) | PASS |
| Report states it validates mechanics, not PLECS fidelity | report header and the "Reading" section | PASS |

## Tests

| File | Tests |
|---|---|
| `tests/envs/test_surrogate_env.py` | 9 |
| `tests/test_compare_with_spade.py` | 3 |

Full suite: **1050 passed, 4 xfailed (strict, findings), 0 failed** (856 s; 1038 after E7 + 12 new) — `e5_branch_pytest.txt`.

## Limits

- No PLECS data: the surrogate pipeline has never seen a switched circuit.
- Noise-free data. The grey box is exact here by construction.
- The MLP is a one-step model trained without multi-step loss, hence its open-loop drift.
