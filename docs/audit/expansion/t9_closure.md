# T9 — partial closure of the expansion round (E0–E14), 2026-10-21

Branch `feat/E13-partial-closure`. Partial closure: there is no PR from dev to main, by instruction. Machine-readable copy: [`t9_closure.json`](t9_closure.json).

## 1. Test suite per task

Each count comes from the branch log in this folder, produced with `PYTHONPATH=src python -m pytest -q` (full suite, slow tests included) before the merge into `dev`.

| Task | Branch | Suite | Log |
|---|---|---|---|
| Baseline before E0 | dev | 760 passed | `e0_baseline_pytest.txt` |
| E0 job runner, runs store | feat/E0-foundations | 823 passed | `e0_branch_pytest.txt` |
| E1 Gymnasium envs (dc_motor, vsc) | feat/E1-gym-envs | 845 passed | `e1_branch_pytest.txt` |
| E6 buck converter | feat/E6-buck | 871 passed (the log file keeps only the later page re-run, 7 passed) | `e6_branch_pytest.txt` |
| E10a Unknown condition | feat/E10a-unknown-condition | 890 passed | `e10a_branch_pytest.txt` |
| E10b PINN identification | feat/E10b-pinn-identification | 905 passed | `e10b_branch_pytest.txt` |
| E12 Algorithm Guide | feat/E12-algorithm-guide | 949 passed | `e12_branch_pytest.txt` |
| E2 PLECS RPC dataset route | feat/E2-plecs-rpc | 1002 passed | `e2_branch_pytest.txt` |
| T1 findings (E10a/E10b) | feat/E10-findings | 1007 passed, 2 xfailed | `t1_branch_pytest.txt` |
| T2 E14 residual forecaster | feat/E14-residual-forecaster | 1023 passed, 4 xfailed | `e14_branch_pytest.txt` |
| T3 E7 Tuning Studio | feat/E7-tuning-studio | 1038 passed, 4 xfailed | `e7_branch_pytest.txt` |
| T4 E5 surrogate, Compare with SPADE | feat/E5-surrogate | 1050 passed, 4 xfailed | `e5_branch_pytest.txt` |
| T5 E8 RL Lab | feat/E8-rl-lab | 1063 passed, 4 xfailed | `e8_branch_pytest.txt` |
| T6 SPADE/Pmsm-v0 | feat/E1-pmsm-env | 1070 passed, 4 xfailed | `t6_branch_pytest.txt` |
| T7 E11 IGBT RUL | feat/E11-igbt-rul | no code: stopped at the feasibility check (docs only) | `t7_igbt_feasibility.md` |
| T8 E9 DAB prep | feat/E9-dab-prep | 1083 passed, 4 xfailed | `t8_branch_pytest.txt` |
| T9 this closure | feat/E13-partial-closure | 1083 passed, 4 xfailed | `t9_branch_pytest.txt` |

The E6 total (871) is recorded in `e6_buck.md` and in commit c619a0c. The E6 log file was overwritten by the later page re-run.

The 4 xfails are all `strict=True` and carry a finding ID: AI-N15 and AI-N16 (T1), plus AI-N18 and AI-N20 (T2). No other xfail was added.

## 2. What ran with PLECS

**Nothing.** No machine with PLECS 5.0 and a license was available in this round.

- Every PLECS RPC test (E2) runs against the local test double `tests/integrations/plecs/fake_plecs.py`. The screenshots in this round use the same double.
- E3/E4 (FMU co-simulation) were not started (decision D7).
- The real 3×3 E2 sweep was not run.
- The DAB performance model (T8) has code and tests, but no trained run, because there is no PLECS sweep table.

## 3. Findings

| ID | Origin | Severity | State | Test |
|---|---|---|---|---|
| AI-N15 | T1 (E10b) | Medium | Open: PINN estimates C low, −1.3 % to −9.1 % at 0–5 % noise | `tests/test_findings_e10.py` (strict xfail plus pinned behaviour) |
| AI-N16 | T1 (E10b) | Medium | Open: bootstrap coverage 0/52/70/78 % vs 90 % nominal; GUI says "Coverage not validated" | `tests/test_findings_e10.py` (strict xfail plus GUI text) |
| AI-N17 | T1 (E10a) | High | Open: promoted classifier has no OOD block (D6) | `tests/test_findings_e10.py` |
| S-14 | T1 (E10b) | Medium | Documented: the DPC diverges for C ≤ ×0.84 at 4 Ω, ≤ ×0.67 at 5 Ω and ≤ ×0.42 at 8.0064 Ω; stable to ×0.40 at 10–20 Ω | `tests/test_findings_e10.py`, `docs/audit/stability/dpc_capacitor_divergence.md` |
| AI-N18 | T2 (E14) | Medium | Open, decision pending: start-up windows inflate the residual threshold (≈ 50 % detection) | `tests/test_findings_e14.py` (strict xfail) |
| AI-N19 | T2 (E14) | Medium | Documented: AUROC 1.000 is optimistic on noise-free simulated signals | `tests/test_findings_e14.py` |
| AI-N20 | T2 (E14) | High | Open: the RPi5/ESP32 residual detectors are not usable | `tests/test_findings_e14.py` (strict xfail) |

Negative results without a finding ID, documented in their reports:

| Task | Result | Report |
|---|---|---|
| E7 | The bandwidth goes to its bound because the objective has no effort term. | `e7_tuning_s04.md` |
| E8 | RL loses to the PI on steady-state tracking; DDPG is seed-sensitive. | `e8_rl_benchmark.md` |
| T7 | At most 2 run-to-failure IGBT devices, with different protocols. | `t7_igbt_feasibility.md` |

## 4. Spec §14 matrix (docs/design/SPADE_GUI_Complete_Specification.md)

"Passed" means the cited test exists and passed in the T9 suite (`t9_branch_pytest.txt`). Nothing below was checked by hand only.

| Case | Expected | Result | Evidence / reason |
|---|---|---|---|
| Change page during RL | Worker continues; state persisted; no duplicate training | **Passed (generic job path)** | `tests/test_experiment_runs_page.py::test_job_queued_from_the_page_survives_navigation_and_reload`; `tests/jobs`: `test_killed_worker_leaves_the_real_state_and_the_next_worker_carries_on`, `test_orphaned_job_is_closed_as_worker_lost_not_requeued`. RL jobs use the same queue (`rl_training` handler), but no test switches pages while an RL job runs. |
| Reload browser | Jobs and runs recovered from storage | **Passed** | `test_job_queued_from_the_page_survives_navigation_and_reload`; `tests/runs/test_store.py::test_runs_experiments_and_artifacts_persist_across_store_instances` |
| PLECS RPC not configured | Actionable error; rest of SPADE works | **Passed** | `tests/integrations/plecs/test_plecs_studio_page.py::test_navigable_without_plecs_and_no_gym_promise` ("Cannot reach RPC endpoint"); `test_unreachable_endpoint` |
| PLECS on a remote machine | Blocked, local requirement explained | **Passed** | `test_only_local_http_endpoints`; `test_connection_test_reports_unreachable_and_remote_hosts` ("Backend running on a different host") |
| Import model without required ports | Wizard blocks execution; mapping editable | **Partial: blocking passed; editable mapping not run** | `test_invalid_manifest_is_reported_and_nothing_is_imported` blocks the import. There is no mapping wizard: it belongs to the Gym route (E3/E4), which was not started (D7). |
| Partially failed sweep | Successes/failures per condition, no imputation | **Passed** | `test_sweep_in_the_worker_records_failures_and_each_row_carries_its_own_parameters`; `test_timeouts_and_a_dropped_connection_are_recorded_per_condition`; `tests/runs/test_store.py::test_failures_are_shown_not_imputed_and_block_an_improvement` |
| Surrogate out of range | Warning/block per policy; link to the real sweep | **Partial: warning passed; link not run** | `tests/test_compare_with_spade.py::test_out_of_domain_is_warned_and_grid_lists_every_condition`, `tests/envs/test_surrogate_env.py::test_env_flags_out_of_domain_and_uses_the_surrogate`. The surrogate is fitted to the native buck (D3), so there is no real PLECS sweep to link to. |
| Stepwise without validated FMI | `Unavailable`, button disabled, reason visible | **Passed** | `test_navigable_without_plecs_and_no_gym_promise`: "Interactive stepping is not available with the detected backend."; Gym goal button disabled |
| DQN with continuous actions | Block and explain the need for a discrete action | **Passed** | `tests/test_rl_lab_page.py::test_dqn_asks_for_a_discretization`; `tests/control/test_rl_agents.py::test_dqn_needs_a_discretizer_in_the_job_spec` (refused without a discretizer) |
| Convergence without improvement | `No improvement`, no hidden promotion | **Passed** | `tests/runs/test_store.py::test_worse_candidate_is_no_improvement`; Tuning/RL approval refused (`test_blocked_approval_is_refused_by_the_backend`, `test_training_job_end_to_end_and_blocked_default`) |
| Comparison with different grids | Blocked or explicit intersection, no misleading ranking | **Passed** | `test_disjoint_grids_or_kinds_are_not_comparable`; `test_partial_overlap_compares_the_intersection_and_counts_the_rest` |
| OOD on, high confidence in a known class | `Unknown condition` takes priority | **Passed** | `tests/test_ai_condition_page.py::test_unknown_condition` |
| OOD detector absent | No fictitious third class | **Passed** | `tests/test_findings_e10.py::test_ai_n17_promoted_classifier_has_no_ood_detector`; the GUI says "OOD detection unavailable" (`tests/test_ai_condition_page.py`) |
| PINN with unvalidated uncertainty | Method declared, limitation warned | **Passed** | `tests/test_findings_e10.py::test_ai_n16_gui_says_coverage_not_validated` |
| RUL dataset with unverified license | Explicit "verification pending" label | **Not run** | There is no RUL screen: T7 stopped at the feasibility check. The license is documented in `DATA.md`, but no GUI label exists to test. |
| Change a default without the gate | Refused by the backend, not just a disabled button | **Passed** | `tests/runs/test_store.py::test_approval_is_enforced_in_the_backend`; `tests/test_tuning_studio_page.py::test_blocked_approval_is_refused_by_the_backend` |
| Dark/light/mobile/keyboard | Contrast, focus, units, legibility, no hover required | **Partial: light/dark captured; rest not run** | Screenshots of every new screen in both themes are in `docs/screenshots/expansion/`. Not measured: token contrast (pending §6), mobile and tablet widths, keyboard-only navigation. |
| Copy reproducible command | Config + seed + versions, no secrets, safe paths | **Passed** | `tests/runs/test_store.py::test_commands_carry_no_absolute_paths_or_credentials`, `test_environment_info_records_versions_without_importing_tensorflow`; `tests/jobs::test_scenario_sweep_runs_in_the_worker_and_its_command_reproduces_it` |

Totals:

| Result | Rows |
|---|---|
| Passed | 14 (one of them on the generic job path only) |
| Partial | 3 |
| Not run | 1 |
| Failed | 0 |

## 5. Spec §20 criteria

| Criterion | Result | Reason |
|---|---|---|
| **PLECS: Gym creation** | | |
| Gym/Dataset/Surrogate difference clear in SPADE | Passed | Goal step with three cards; `test_navigable_without_plecs_and_no_gym_promise` |
| Start without PLECS, save a draft and reopen it | Passed | Profiles persist without a connection (`test_profiles_versioned_and_stale_when_the_file_changes`, `test_dataset_route_end_to_end`) |
| Distinguish `.plecs` and `.fmu` | Not run | The `.fmu` route is E3/E4 (D7) |
| Detect real variables; no invented ports | Partial | The manifest is user-declared and validated (`test_manifest_validation`). There is no automatic detection (needs PLECS). |
| Readiness panel navigates to the missing field | Not run | Part of the Gym wizard (E4) |
| Auditable action/observation/reset/reward/safety contracts | Passed for native envs | `tests/envs/test_native_envs.py`, Environments page; for PLECS Gym: not run (E3/E4) |
| Only a stepping adapter enables Gym publication | Passed | `test_capabilities_default_off_and_rl_needs_stepping` |
| Persistent report of all tests, even failures | Passed | Run records keep failures (`test_handler_exception_is_a_failed_run_with_a_traceback`) |
| Published Gym shown as simulated with version and hash | Not run | No PLECS Gym can be published yet (E3/E4) |
| **PLECS: dataset / surrogate / comparison** | | |
| Persistent sweeps with per-condition metrics and failures | Passed | E2 tests, against the test double only |
| An RPC dataset never appears as a Gym plant | Passed | `test_navigable_without_plecs_and_no_gym_promise`; the RPC profile shows "Interactive stepping: Not supported" |
| Surrogate fit uses a declared split and shows error per condition | Passed | `test_split_is_by_condition_and_fixed`; `test_compare_condition_reports_errors_and_domain` |
| Each SPADE-vs-PLECS curve comes from a real run | Not run | No PLECS run exists. The PLECS-circuit comparison shows its empty state. |
| Incompatible comparisons claim no improvement or equivalence | Passed | `test_disjoint_grids_or_kinds_are_not_comparable`; the out-of-domain surrogate warning says "does not claim equivalence" |
| **Platform** | | |
| New navigation coexists without duplicating simulation modules | Passed | Envs wrap `run_scenario` bit for bit (`test_*_reproduces_run_scenario_bit_for_bit`) |
| Jobs survive navigation and refresh; real cancel/resume | Passed (cancel); resume partial | `test_cancel_*`. Resume exists only where a checkpoint is saved; the UI says "Resume unavailable" otherwise. |
| Run Comparison reused in Tuning, RL, native and PLECS | Partial | `compare_runs` drives Experiment Runs. Tuning and RL show their own baseline-vs-candidate tables on a fixed grid, not the shared component. |
| AI predictions labelled `AI SUGGESTION — REQUIRES VERIFICATION` | Passed (AI Diagnostics); DAB not exercised | `tests/test_ai_condition_page.py`, `tests/test_residual_gui.py`. The DAB suggestion carries the badge (`views/dcdc_converters.py:153`), but no test reaches it: there is no trained DAB model. |
| No misleading `LIVE` / `production ready` / `hardware verified` | Passed | `test_live_is_not_a_provenance_label`; banned strings checked in the page tests |
| Light/dark/keyboard/accessibility/responsive on 4 device sizes | Not run (light/dark screenshots only) | See §4 and §6 |
| Default changes and model promotion need persisted human approval | Passed | `test_approval_needs_a_named_person_a_decision_and_a_gate_report`, `test_approval_is_enforced_in_the_backend` |

## 6. Pending (annotated, not done in this round)

- **Contrast of the colour tokens (`src/driveflow/viz/theme/tokens.py`, DARK/LIGHT):**
  - not measured: no WCAG contrast ratio has been computed for text, badge or chart tokens against their surfaces in either theme;
  - the screenshots show legibility by eye only, which is not a measurement.
- **Worker heartbeat on Windows:**
  - `jobs/persistence.pid_alive` returns `None` outside POSIX, so on Windows a dead worker is detected only by heartbeat age (`HEARTBEAT_S = 2.0`).
  - `start_local_worker` uses `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` there.
  - Neither path has run on Windows: CI and every suite in this round ran on Linux.
- **UI observation (no finding ID; GUI, not AI or simulation):**
  - after a failed PLECS sweep, PLECS Plant Studio → My Plants still shows "Last sweep —" and "Batch sweep: Not shown yet";
  - the failure is visible only in Experiment Runs (screenshot `plant_studio_runs_after_failed_sweep_*.png` vs `experiment_runs_error_*.png`).
- **Blocked or decision-dependent:**
  - E3/E4 (PLECS machine);
  - the real 3×3 E2 sweep;
  - E14 promotion (decisions a/b/c in VALIDATION_SUMMARY §7.2);
  - an E7 effort term before any default;
  - E11 option a/b/c (`t7_igbt_feasibility.md`);
  - DAB training (PLECS sweep table).

## 7. Screenshots

The state index is in [`docs/screenshots/expansion/README.md`](../../screenshots/expansion/README.md). It lists 25 screen-state pairs in light and dark, 50 files in all. For every state that could not be reached, it gives the reason.
