# E0: foundations (contracts, run store, jobs, components, Experiment Runs)

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E0. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §2, §6, §7, §9, §12, §13 Phase 0.
Branch: `feat/E0-foundations` from `dev` (8a69555). Date: 2026-10-20. Machine: Linux x86-64, Python 3.12.13, Streamlit 1.66.0.

## 0. Baseline before E0 (P0)

| Measurement | Result |
|---|---|
| Full suite on `dev` @ 8a69555, `SPADE-dev/.venv` | 760 passed, 0 failed (587 s). This run overlapped my first edits to `viz/app.py` and `viz/adapters/session.py`, so it is not counted as the clean baseline. |
| Full suite on `main` @ a2f2cd3 (same code as dev; dev only adds two documents), `SPADE/.venv` | **760 passed, 0 failed** (592 s) — `docs/audit/expansion/e0_baseline_pytest.txt` |

The figure of 760 that Luciano reported is confirmed.

## 1. Inventory before editing (spec §21.1–2)

Existing pieces that were reused, not duplicated:

| Piece | Signature / role | Used by E0 |
|---|---|---|
| `viz/components/page_header.py` | `badge_html(label, tone, glyph, title)`, `status_badge_html(status)`, `source_badge_html(source)`, `page_header(eyebrow, title, description, badges)`, `provenance_table(rows, title)`, `empty_state(text)` | Header, badges and provenance table of the new page and components |
| `viz/components/metric_cards.py` | `kpi_row(items)`, `check_list(items)` | The `.spade-check` row style is the base of the state panel |
| `viz/components/promotion_gate.py` | `promotion_gate(decision)` | Kept for models; `ApprovalGate` is separate (controllers and defaults) |
| `viz/components/artifact_table.py`, `model_comparison.py`, `signal_chart.py`, `alerts_table.py`, `motor_status.py`, `inputs.py`, `presets.py` | Unchanged | Not touched |
| `viz/adapters/session.py` | `record_run(kind, params, summary)`, `run_log()`, `config_hash(params)` (12-hex prefix), `MAX_RUN_LOG = 30` | `record_run` now also mirrors to the durable store (see §2.6) |
| `viz/adapters/workspace.py` | `workspace_status()` | Unchanged |
| `viz/theme/tokens.py` | `DARK`, `LIGHT`, `SERIES`, `STATUS_META`, `SOURCE_META`, `palette()` | Extended with job/run states, the five component states and the provenance labels |
| `viz/theme/css.py` | `inject_base_css()`, `.spade-*` classes | Unchanged; the new components only use existing classes |
| `viz/app.py` | `PAGES` dict, `_sidebar_status()`, `main()` with `st.navigation` | One page added and a jobs counter in the sidebar footer |

## 2. What was built

Package layout as fixed in the list (decision 1). `src/driveflow/experiments/` is **not** created.

### 2.1 `src/driveflow/runs/`

- `schemas.py`: `PlantProfile`, `EnvironmentCapability`, `ExperimentSpec`, `Run`, `SignalSeries`, `ApprovalRecord`, `ErrorCode` and `SpadeError`.
  - Frozen dataclasses with `SCHEMA_VERSION = 1`.
  - `from_dict()` refuses a newer version, another entity or unknown fields.
  - Validation happens in the backend. Examples: a failed run needs `failure_reason`; `origin` must be one of the five provenance labels (there is no LIVE label); a file-backed profile needs its SHA-256; `can_train_rl` requires stepping and reset.
  - `ExperimentSpec.config_hash()` is the full SHA-256 of everything except the id and the creation time.
- `store.py`: `RunStore` on SQLite (standard library) at `<workspace>/spade.db`, with artifacts in `<workspace>/runs/<run_id>/`.
  - The workspace is `$SPADE_WORKSPACE`, otherwise `<repo>/workspace` (git-ignored).
  - Runs and approvals are immutable: saving the same id twice is an error.
  - `record_approval()` enforces the gate in the backend: an `approved` decision is refused while the gate report has blocking items, and the candidate run must exist.
  - Each run records its environment (`environment_info()`: package versions read from metadata without importing TensorFlow, git commit, and whether the worktree has uncommitted changes).
  - The reproducible command is `python -m driveflow.jobs.worker run runs/<run_id>/spec.json`, relative to the workspace. `safe_command()` rejects absolute paths and URLs with credentials.
- `profiles.py`: native profiles for DC motor, VSC and PMSM.
  - Their only declared capability is batch simulation. Stepping and RL stay off until E1 proves them.
  - `is_stale()` and `mark_stale()` compare the profile's file hash with the file on disk.
- `compare.py`: `compare_runs()` compares on the intersection of the condition grids.
  - It reports how many conditions were left out.
  - Failed, skipped and missing conditions are shown as such and never imputed.
  - A new failure blocks an "improved" verdict.
  - It returns "not comparable" when the grids are disjoint or the experiment kinds differ.

### 2.2 `src/driveflow/jobs/`

- `schemas.py`: the `JobState` machine from spec §6 and the `Job` entity.
  - `resumable` is an attribute. It is always False today because no handler writes checkpoints.
  - A cancel request moves the job to `cancelling`. The job ends as whatever actually happened: `cancelled`, or `completed` if the work finished before the next cancellation point.
- `persistence.py`: `JobDB`, which shares `spade.db`.
  - Every read-modify-write runs under `BEGIN IMMEDIATE`.
  - Jobs are claimed FIFO by insertion order. They were first ordered by `queued_at`, but that has one-second resolution and made the lock test flaky; it was fixed.
  - Locks are per engine (one PLECS RPC command at a time, decision 5), with an event log, worker heartbeats and `recover_orphans()`.
- `handlers.py`: the job kinds `scenario_sweep` and `selftest.steps`.
  - `scenario_sweep` calls `datagen.runner.run_scenario` once per condition and saves one Parquet file per condition. A failed condition never hides the others.
  - `selftest.steps` has no physics; it lets the tests exercise cancellation and loss of the worker deterministically.
- `service.py`: `JobService`.
  - `submit` validates the spec against its handler before queueing.
  - The ETA is None until 3 jobs of the same kind have completed; after that it is the median seconds per unit times the units left.
  - `start_local_worker()` starts a detached process.
- `worker.py`: `python -m driveflow.jobs.worker serve [--once]` runs the queue, and `... run <spec.json>` re-runs a saved spec.
  - On start, the worker closes jobs left active by a dead worker as `failed` with `WORKER_LOST`. They are not re-queued.
  - It writes `spec.json`, `log.txt` and `traceback.txt` (on error) as artifacts.

Nothing in `runs/` or `jobs/` imports Streamlit.

### 2.3 Components (spec §9)

The new components are `SourceBadge`, `ProvenanceFooter`, `ConnectionStatus`, `JobCard`, `RunComparison`, `ApprovalGate` and `FailurePanel`, built on `state_panel.py`.

- Each one has a pure `*_view()` function that returns one of the five states (empty, loading, error, success, negative), plus a render function.
- The messages are the spec's microcopy (§9, §10).
- The status is shown as a glyph plus a text label. Colours come from `tokens.py`, and badges use translucent tints of the same tokens, as the existing components do.

### 2.4 Page `Workspace / Experiment Runs` and navigation

The page is `viz/pages/experiment_runs.py` → `viz/views/experiment_runs.py` → `viz/adapters/jobs_adapter.py`. It shows:

- the worker status, with "Start local worker" and the command to start one by hand;
- a "New experiment" form for a DC-motor or VSC scenario sweep, with an experiment summary and the config hash before "Queue experiment";
- the job list, read from the database, with filters and search;
- the selected job as a JobCard (live refresh every 2 s while it is active), with its logs and run details (Config / Metrics with the FailurePanel / Artifacts with SHA-256 / Logs / Provenance);
- Compare runs, using RunComparison;
- the records mirrored from the interactive pages.

In `app.py`, only this page was added. No links to pages that do not exist yet. The sidebar footer now shows "Jobs: N running · M queued · K worker(s)" and an "Open queue" link.

### 2.5 Things not changed

Other existing pages behave as before; the optional ProvenanceFooter was not added to them in this task. No physics, defaults or models changed. The byte-identical datagen tests were not touched.

### 2.6 Session runs (spec §13 Phase 0: "conserve compatibility, migrating or marking legacy")

`record_run()` keeps its session log (30 entries) and now also saves a `legacy_session` Run in the store. That copy has parameters and summary only: no artifacts and no command, because the interactive pages do not run as specs. Failures are swallowed so that persistence can never break a page.

`tests/conftest.py` gives every test its own workspace, so the existing page tests do not write into the repository.

## 3. Acceptance (spec §13 Phase 0 and the E0 closing criterion)

| Criterion | Evidence | Result |
|---|---|---|
| Launch a job, change page or close the tab, come back and see the real state | `tests/test_experiment_runs_page.py::test_job_queued_from_the_page_survives_navigation_and_reload` (queue from the page, Overview and back, then a new AppTest session). Checked in Chrome: the job queued from the browser ran in the worker started with "Start local worker", and after a page reload it showed `completed` with its run. | PASS |
| Honest cancellation | `tests/jobs/test_worker.py::test_cancel_is_requested_then_confirmed_by_the_worker` (separate process): the state is `cancelling` until the worker confirms; conditions already done stay `ok` and the rest are `skipped`. A queued job is cancelled at once and nothing runs (`test_cancel_while_queued...`). Cancel from the page: `test_cancel_from_the_page_is_recorded`. | PASS |
| Kill the worker and recover the real state | `test_killed_worker_leaves_the_real_state_and_the_next_worker_carries_on`: `kill -9` of the worker while a job is running gives `failed` / `WORKER_LOST` by dead pid (POSIX) or heartbeat age, with no resume promised; the next worker does not re-run the job and does run new work. `test_worker_start_closes_jobs_left_by_a_dead_worker` checks the recovery at worker start. | PASS |
| Recover artifacts and logs | Every run saves `spec.json`, `log.txt` and its own artifacts with SHA-256 (`test_worker_completes_and_records_partial_failures`). | PASS |
| Copy the command and keep the hash | `test_scenario_sweep_runs_in_the_worker_and_its_command_reproduces_it`: a 2-condition DC sweep (0.05 s each) in the worker; the stored command, re-run, gives the same `config_hash`, the same metrics and byte-identical Parquet files. | PASS |
| History with an error per condition | `partially_completed` with a failure table (`test_finished_job_shows_run_details_failures_and_command`); a handler exception becomes `failed` with `traceback.txt`. | PASS |
| Components in 5 states | `test_each_component_reaches_all_five_states` (7 components × 5 states) and `test_components_render_in_streamlit_without_errors`. | PASS |
| Both themes | Screenshots of the page in Chrome with `prefers-color-scheme` dark and light (not committed; the systematic capture of every screen in its 5 states belongs to E13). Contrast was not measured with a tool (spec §11 asks for that; still pending, for E13). | Checked visually, not measured |
| Basic store concurrency | `test_concurrent_writers_do_not_lose_runs` (6 threads × 20 runs); FIFO plus the engine lock (`test_engine_lock_serialises_jobs_of_one_engine`, 5 consecutive runs green after the FIFO fix). | PASS |

## 4. Tests

| File | Tests |
|---|---|
| `tests/runs/test_schemas.py` | 18 |
| `tests/runs/test_store.py` | 15 |
| `tests/jobs/test_service.py` | 10 |
| `tests/jobs/test_worker.py` | 4 |
| `tests/test_experiment_runs_page.py` | 15 |
| `tests/test_ui_contract.py` | +1 (`experiment_runs` added to T-UI-01) |

Full suite on the branch: **823 passed, 0 failed** (595 s; 760 previous + 63 new) — `docs/audit/expansion/e0_branch_pytest.txt`.

## 5. Limits and pending items

- Only one worker per engine executes at a time. Redis/RQ and multi-user deployment are out of scope (decision 5).
- "Start local worker" starts the process on the machine running Streamlit. On a remote host the user starts it by hand (`python -m driveflow.jobs.worker serve`).
- Windows: dead-pid detection falls back to the heartbeat age (30 s by default). That path is covered by the test branch but has not run on Windows; the multi-OS CI job only runs `tests/compat`.
- `ETA` and `Resume`: there is no ETA until there is history, and no resume because no handler writes checkpoints. The UI says so with the spec's microcopy.
- Systematic screenshots of every state and a contrast measurement: E13.
