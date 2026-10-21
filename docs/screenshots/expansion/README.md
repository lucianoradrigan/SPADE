# Screenshots — expansion round (E0–E14), 2026-10-21

Every new screen of the round, captured in the dark and light themes (`<screen>_<state>_<theme>.png`, 1440 px wide, full page).

## How they were made

- Eight local Streamlit servers ran from `dev` at 757fb10, one per (state, theme) pair: `--theme.base dark|light`.
- Each server had its own `SPADE_WORKSPACE`:

| Workspace | Contents |
|---|---|
| empty | No profile and no job. |
| loading | A PLECS sweep, a tuning job and an RL job, queued with no worker running. |
| error | A PLECS sweep against an endpoint with nothing listening, run by the worker. The job ends `failed`. |
| success | The same three jobs, run by `Worker().serve(once=True)` against the PLECS test double `tests/integrations/plecs/fake_plecs.py`. The sweep ends `partially_completed`, because the double fails some conditions on purpose. |

- Capture used Playwright (Chromium, headless).
- All data is SIMULATED. The PLECS rows come from the test double, not from PLECS.

## State index

✓ = captured. Where a state was not captured, the cell says why.

| Screen | Empty | Loading | Error | Success | Negative result |
|---|---|---|---|---|---|
| Tuning Studio (E7) | ✓ `tuning_studio_empty` | ✓ `tuning_studio_loading` (job queued, no worker) | Not captured. Invalid ranges are refused when the job is submitted (the sliders cannot produce low ≥ high), and a handler failure needs fault injection. That path is tested in `tests/jobs`. | ✓ `tuning_studio_success` | ✓ In `tuning_studio_success`, at the bottom: the approval gate is blocked ("Use as default" not authorized, D1). |
| RL Lab (E8), Training/Benchmark tabs | ✓ `rl_lab_empty` | ✓ `rl_lab_loading` | Not captured, for the same reason as Tuning Studio. The failed-run message is `views/rl_lab.py:88`. | ✓ `rl_lab_success` (Training) | ✓ `rl_lab_negative` (Benchmark): "The RL policy performs worse than PI", approval blocked |
| Compare with SPADE (E5) | ✓ `compare_empty` | Not applicable: the comparison runs synchronously in under a second, with no intermediate state. | ✓ `compare_error` (v_ref ≥ V_in) | ✓ `compare_success` | ✓ `compare_negative` (load 1 Ω, outside the validated range: warning, no equivalence claim) |
| PLECS Plant Studio → My Plants (E2) | ✓ `plant_studio_runs_empty` | ✓ `plant_studio_runs_loading` | Not shown on this tab. See `plant_studio_runs_after_failed_sweep`: "Last sweep —". The failure is visible only in Experiment Runs (pending note in `t9_closure.md` §6). | ✓ `plant_studio_runs_success` | Not applicable |
| Experiment Runs (E0) | ✓ `experiment_runs_empty` | ✓ `experiment_runs_loading` | ✓ `experiment_runs_error` (failed sweep, "NEGATIVE RESULT" and "FAILED" badges) | ✓ `experiment_runs_success` | Shown in `experiment_runs_error` and in the `partially_completed` sweep of `experiment_runs_success` |
| DC-DC Converters → DAB tab (T8/E9) | ✓ `dab_tab_empty` | Not reachable | Not reachable | Not reachable | Not reachable. The other four states need a trained DAB model, and there is no PLECS sweep table (D8). |
| AI Diagnostics → Forecast residual (E14) | ✓ `residual_empty` | Not applicable (synchronous) | Not captured | Not reachable: no regressor is promoted (decision pending, VALIDATION_SUMMARY §7.2) | ✓ `residual_unavailable`: "Residual detector not available" next to the classifier |
| Environments (E1/T6/E5) | Not applicable: the catalogue always lists the registered environments | Not applicable | Not captured (the contract check is covered by `test_contract_check_button_runs_check_env`) | ✓ `environments_success` (Pmsm-v0 and BuckSurrogate-v0 included) | Not applicable |
| Algorithm Guide (E12) | Not applicable (static guide) | Not applicable | Not applicable | ✓ `algorithm_guide_success` | Not applicable |
| DC-DC Converters → Buck (E6) | Not captured | Not applicable | Not captured (`test_invalid_reference_is_rejected_by_the_backend`) | ✓ `dcdc_buck_success` | Not captured |

Not measured: colour contrast, mobile and tablet widths, keyboard navigation (see `docs/audit/expansion/t9_closure.md` §6).
