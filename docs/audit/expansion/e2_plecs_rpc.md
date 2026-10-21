# E2: PLECS RPC client, plant profiles and PLECS Plant Studio (batch routes)

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E2. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §3.2–3.4 (stages 1–3), §3.8–3.9, §3.14.1, §3.14.2, §3.14.5, §3.15.
Branch: `feat/E2-plecs-rpc` from `dev` (d1c96f6, after E12). Date: 2026-10-21. Machine: Linux x86-64, Python 3.12.13, Streamlit 1.66.0.

- **Origin:** Fund. `8_PE_Simulation_Automation/PlecsAutomation/Data acquisition.ipynb`, the `plecs.load` / `plecs.simulate(model, {"ModelVars": …})` / `plecs.close` pattern. It was rewritten on `xmlrpc.client` from the standard library, with no new dependency.
- **PLECS facts used:** taken from the official documentation (https://docs.plexim.com/plecs/latest/scripting/, read 2026-10-21):
  - RPC is disabled by default and accepts local clients only;
  - the examples use `http://localhost:1080`;
  - `simulate` is blocking, and a second blocking call is an error;
  - the options are `ModelVars` and `SolverOpts` (`TimeSpan`, `Timeout`, `OutputTimes`; `StopTime` is obsolete);
  - the result is `Time` and `Values`, one signal per top-level outport;
  - there is **no RPC command for the PLECS version**.
- **Real PLECS evidence:** **no ejecutado: sin PLECS** (`docs/audit/plecs/e2_rpc_sweep.md` explains how to produce it).

## 1. What was built

### 1.1 `src/driveflow/integrations/plecs/rpc_client.py`

- **Endpoint.** `check_local_url` accepts only `http://localhost` or loopback addresses. A remote host is rejected with the spec's message "Backend running on a different host".
- **`PlecsRpcClient(url)`:**
  - `test_connection()` returns an `EngineCapabilityReport`:
    - reachability and latency;
    - `system.listMethods` if the server answers it; a server that answers with a fault still counts as reachable;
    - `plecs_version = "not reported by the RPC interface"`;
    - an explicit list of what was **not** checked: version, license, interactive stepping, circuit loading, and batch simulation (shown only by a completed sweep).
  - `load`, `close`.
  - `simulate(model, output_names, model_vars, solver_opts, time_limit_s)`:
    - passes the per-run limit to PLECS as `SolverOpts.Timeout` and bounds the socket wait by that limit plus a margin;
    - checks the result against the manifest's outputs;
    - accepts both a signal-major and a time-major `Values` matrix;
    - rejects a wrong shape, non-finite values or time that goes backwards. Equal consecutive time stamps are allowed, because PLECS outputs both sides of a switching event.
- **Locking.** One lock per endpoint inside the process. Jobs from different processes are serialized by the job engine lock `plecs_rpc` (E0).
- **Typed errors.** `PlecsRpcError(code, …)` with a hint for each code:

| Code | Message |
|---|---|
| `unreachable` | Cannot reach RPC endpoint |
| `remote_host` | Backend running on a different host |
| `bad_url` | — |
| `timeout` | — |
| `simulation_failed` | the solver message |
| `busy` | — |
| `connection_lost` | — |
| `model_not_found` | — |
| `bad_result` | — |

- **Notebook defects not reproduced:**
  - each result carries its own ModelVars;
  - the URL is checked instead of hand-written (the notebook had a stray quote);
  - paths are `pathlib` paths;
  - there are no threads against a server that runs one blocking command at a time.

### 1.2 `integrations/plecs/manifest.py`

- An `InterfaceManifest` that the user **declares by hand**: SPADE does not read the circuit's blocks.
  - `model_vars`: name, unit, default, minimum, maximum.
  - `outputs`: the top-level outports in port order, each with a unit.
- Validation: valid names, units required, a consistent range and default, no duplicates, no reserved column names, and no name used both as a ModelVar and as an output.
- `check_values` rejects unknown or non-finite ModelVars and values outside the declared range.
- `EXAMPLE_BUCK` is the spec's example. In the GUI it is labelled "EXAMPLE CONFIGURATION — NOT EXECUTED".

### 1.3 `integrations/plecs/batch.py`

- **Profiles.** A `PlantProfile` with `kind = plecs_rpc_batch`, engine `plecs_rpc` and origin `SIMULATED · PLECS`. It stores:
  - the circuit's SHA-256;
  - the name and path of the circuit on the backend's computer;
  - the manifest and the endpoint.
- **Profile lifecycle:**
  - a new profile starts as `imported`, with `can_simulate_batch = False`;
  - `mark_batch_shown` creates the next version with `can_simulate_batch = True` and cites the sweep run that showed it;
  - interactive stepping is never enabled;
  - re-importing the same ID creates a new version;
  - if the file's hash changes, the profile is `Stale` (`runs/profiles.is_stale`).
- **Sweep design.** `design_conditions` builds a grid (levels per variable) or a seeded Latin hypercube (`scipy.stats.qmc`, already a dependency), within the declared ranges, with a budget of 2000 conditions.
- **One condition.** `run_condition` returns a table in which **every row carries its own condition's ModelVars**.
- **Uploads.** An uploaded `.plecs` is stored under `workspace/plecs/circuits/<sha12>/`. Datasets go to `data/plecs/<profile>/<run_id>/` (git-ignored; `$SPADE_PLECS_DATA` overrides the location).

### 1.4 Job `plecs_sweep` (`jobs/handlers.py`)

- **Engine.** `plecs_rpc`, so the worker runs one sweep at a time.
- **Before the sweep:**
  - the spec is validated (profile kind, conditions within the manifest, simulated time, time limit up to 3600 s);
  - a stale profile fails every condition with "The circuit or mapping changed. Re-run verification." and never calls PLECS.
- **During the sweep:**
  - the circuit is loaded once and closed in a `finally`;
  - each failed condition keeps its typed error (`[timeout]`, `[simulation_failed]`, `[unreachable]`/`[connection_lost]`) and is **not interpolated**;
  - cancellation marks the remaining conditions `skipped`.
- **Output.** One Parquet per completed condition, plus `conditions.csv` and `manifest.json` (circuit hash, interface, settings, method and seed, and the note "A completed PLECS sweep does not provide an interactive Gymnasium plant."). They go both to the dataset folder and to the run's artifacts.
- **Profile update.** If at least one condition completes, the profile's next version records that batch simulation was shown by this run.

### 1.5 Scenario and runner

- **Scenario.** `plant_config_id = "plecs:<profile_id>"` with `controller_type = "circuit"`: the control lives in the circuit. This is the dataset route only.
- **New fields:** `plecs_model_vars`, `plecs_time_limit_s`, `plecs_output_points`.
- **Validation in the style of S-03:**
  - the profile ID must be valid;
  - ModelVars must be finite;
  - output points must be an integer ≥ 2;
  - the PLECS fields are rejected on other plants;
  - any other pairing with `circuit` or `plecs:` is refused.
- **Runner.** `run_scenario` dispatches `plecs:` to `_run_plecs_scenario`. It refuses a stale profile. ModelVars that are not set take the manifest's default. Its records do not follow the motor schema of `export_parquet`.

### 1.6 GUI: `Simulation / PLECS Plant Studio` (`viz/pages/plecs_plant_studio.py`)

Layering: `viz/adapters/plecs_adapter.py` → `viz/views/plecs_plant_studio.py`. The page is in `app.py` under Simulation.

| Tab | Content |
|---|---|
| **My Plants** | Search; table with name, ID, version, source, readiness (Imported / Stale / …), "Batch sweep: Shown / Not shown yet", "Interactive stepping: Not supported", last sweep, circuit, short hash. A stale profile shows "This profile changed since its last verification. Revalidation is required." Empty state as in the spec. |
| **Create Plant** | Stage 1 "What would you like to do with your PLECS circuit?" with the 3 cards. "Use PLECS as a Gymnasium plant" is visible, its button is disabled, and it says "Interactive stepping is not available with the detected backend." The surrogate card says "Not available in this version". Only "Generate simulation datasets" continues. Dataset route: import (upload or a path on the backend's computer, ID, name, manifest JSON) → profile → sweep form (variables, Grid/LHS, levels or samples, seed, simulated time, per-run time limit, output points, budget and condition table, a warning when no worker is running) → Start sweep (a job) → results. |
| **Compare with SPADE** | Empty state "No comparison has been generated yet. Select compatible plants and run the comparison." The comparison arrives with E5. |
| **Connection & Diagnostics** | RPC endpoint and Test connection; five separate panels (RPC connectivity, FMU import support, Step capability, Circuit readiness, Environment readiness: "Not derived from RPC connectivity"); report details and a shareable JSON. |

**Results of a sweep:**
- source badge `SIMULATED · PLECS` and the run's state;
- completed / failed counts;
- the curves of up to 12 completed conditions and the distribution of wall time per run;
- the conditions table with each error, and a list of failures;
- export of the dataset (zip) and of the manifest;
- a full-width provenance table: run, profile and version, circuit SHA-256, seed, configuration hash, reproducible command;
- a refresh button and the state of active sweep jobs.

The pages read profiles and runs from the store, and only the worker talks to PLECS. The only exception is Test connection, which runs on a button.

**Checked in Chrome** against the fake server, with a real worker process and an isolated workspace:
- Test connection → "PLECS available (RPC answered in 1 ms)";
- import → a 3-point grid sweep;
- the worker completes it;
- curves, table, export and provenance render;
- My Plants shows "Batch sweep: Shown", "Interactive stepping: Not supported" and the run ID.

Screenshots in dark and light (not committed).

## 2. Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| Fake XML-RPC server imitating load/simulate/close, with a timeout, a non-convergence and a dropped connection | `tests/integrations/plecs/fake_plecs.py` (threaded, so it answers "busy" like PLECS); `test_rpc_client.py` (21 tests: reachable with and without introspection, unreachable, local-only URLs, load/simulate/close with `Timeout` in SolverOpts and no `StopTime`, time-major results, missing file, non-convergence, PLECS-side timeout, client-side timeout of a hung server, connection dropped mid-sweep, busy, malformed results, equal time stamps) | PASS |
| Regression of the notebook defect: each dataset row carries its own combination's parameters | `test_plecs_batch.py::test_sweep_in_the_worker_records_failures_and_each_row_carries_its_own_parameters`: every row of every Parquet equals its condition's ModelVars, and v_out matches the condition's own Vref to 1e-12 | PASS |
| Sweep as a job; failures recorded and not interpolated; dataset with manifest | same test (3 of 6 conditions fail by non-convergence → `partially_completed`; failed rows have no file); `test_timeouts_and_a_dropped_connection_are_recorded_per_condition`; `test_stale_profile_refuses_to_sweep`; `test_spec_validation` | PASS |
| PlantProfile `plecs_rpc_batch` with SHA-256 and versions; Stale if the hash changes | `test_profile_records_hash_and_is_stale_when_the_circuit_changes`; the profile's version 2 after a sweep (`can_simulate_batch` True, stepping False) | PASS |
| `plant_config_id = "plecs:<profile_id>"` in Scenario, validated in the style of S-03 | `test_scenario_plecs_plant_runs_through_run_scenario`; `test_scenario_validation` (6 cases) | PASS |
| AppTest of the page without PLECS configured: navigable, no exceptions | `test_plecs_studio_page.py::test_navigable_without_plecs_and_no_gym_promise` (4 tabs; Gym and surrogate cards disabled; no "Gym ready", "Publish to Environments", LIVE or Connected); `test_connection_test_reports_unreachable_and_remote_hosts`; `test_connection_test_against_a_running_server`; `test_dataset_route_end_to_end` (import → sweep → worker → results; the profile shows "Shown" and "Not supported"); `test_invalid_manifest_is_reported_and_nothing_is_imported` | PASS |
| `plecs_plant_studio` in the UI contract | `tests/test_ui_contract.py` `PAGES` | PASS |
| Real evidence on a machine with PLECS | **no ejecutado: sin PLECS** — `docs/audit/plecs/e2_rpc_sweep.md` | NOT RUN |

## 3. Tests

| File | Tests |
|---|---|
| `tests/integrations/plecs/test_rpc_client.py` | 21 |
| `tests/integrations/plecs/test_plecs_batch.py` | 26 |
| `tests/integrations/plecs/test_plecs_studio_page.py` | 5 |
| `tests/test_ui_contract.py` | `plecs_plant_studio` added |

Full suite on the branch: **1002 passed, 0 failed** (671 s; 949 after E12 + 52 + 1 UI-contract case) — `e2_branch_pytest.txt`.

## 4. Limits

- Never run against PLECS. In particular:
  - PLECS's real fault messages are classified by keywords ("timeout", "busy", "not found"), so an unrecognised message falls back to `simulation_failed`;
  - the real orientation of `Values` is handled both ways, not verified.
- One endpoint per profile. The engine lock is per engine kind (`plecs_rpc`), not per URL, so two different local PLECS instances would also be serialized.
- No parallel runs (`optStructs`). The documentation allows them, but this version runs conditions one by one.
- Uploaded circuits that reference other files by relative path will not find them. Such circuits must be selected in place.
