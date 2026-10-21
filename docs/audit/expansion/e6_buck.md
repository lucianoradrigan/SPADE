# E6: Buck plant, voltage controller, SPADE/Buck-v0 and the DC-DC Converters page

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E6. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §8.1.
Branch: `feat/E6-buck` from `dev` (4820e88, E0 and E1 merged). Date: 2026-10-21. Machine: Linux x86-64, Python 3.12.13, SciPy from `SPADE-dev/.venv`, Streamlit 1.66.0.

Origin: Fund. `BuckAvgEnv` (averaged model: L, C, R, V_in, duty) and `9_Case_Studies_PE/Buck_Design/`. This is a reimplementation; no code was copied, and no file from Fund. was downloaded or committed.

## 1. What was built

### 1.1 `src/driveflow/sim/buck_system.py`

- **Model:**
  - It is averaged, not switched, and models a synchronous buck: the duty is held over each period and the switch node is replaced by d·V_in.
  - The switching ripple is not represented. Results are valid well below f_sw.
  - The inductor current may go negative, so the converter never enters DCM. A diode buck at light load would.
  - There is no capacitor ESR.
  - The module docstring says all of this.
- **Equations:**
  - L di/dt = d·V_in − r_L·i − v
  - C dv/dt = i − v/R
- **Discretization:** exact zero-order hold via the matrix exponential (`scipy.linalg.expm`, already a dependency).
- **Time step:** τ is configurable, default 20 µs (one update per switching period at 50 kHz).
- **Interface:** `BuckSystem(params, tau)` with `reset(i_l, v_out)` and `simulate(duty)`, the same as `VscSystem`.
  - `set_load(R)` re-discretizes for a load step.
  - The duty is clipped to [0, 1].
- **`BuckParams` defaults:**
  - V_in 48 V, L 220 µH, C 220 µF, r_L 0.05 Ω, R 6 Ω, i_max 15 A.
  - L-C resonance: 723 Hz.
  - Every field is validated: finite and > 0, except r_L, which may be 0.

### 1.2 `src/driveflow/control/classical/buck_pi.py`

- `BuckVoltageController` is an average-current-mode cascade with the same structure as `PICascadeController` for the DC motor.
  - Outer loop: a PI on v_out produces i_ref, capped at i_max, with a feedforward v_ref/R.
  - Inner loop: a PI on i_L with feedforward of v_out.
  - Both loops have conditional anti-windup.
- The gains come from the plant parameters:
  - Inner loop: kp_i = L·ω_ci and ki_i = r_L·ω_ci, with f_ci = f_sw/10 = 5 kHz.
  - Outer loop: kp_v = C·ω_cv and ki_v = kp_v·ω_cv/4, with f_cv = f_ci/5 = 1 kHz.
- I tried three voltage bandwidths for the outer loop (f_cv) and kept 1 kHz:

| f_cv | Start-up settling (2 %) | Peak | Dip for a 6→3 Ω step | Back within 1 % |
|---|---|---|---|---|
| 500 Hz | 3.02 ms | 25.48 V | 20.69 V | 5.0 ms |
| **1 kHz (chosen)** | **1.32 ms** | **24.94 V** | **22.02 V** | **1.7 ms** |
| 1.5 kHz | 0.80 ms | 24.60 V | 22.53 V | 0.9 ms |

1.5 kHz is faster, but it is already ~1/33 of f_sw, where the averaged model starts to be optimistic. I kept a margin. It is a parameter of the controller.

### 1.3 Scenario and runner (`plant_config_id = "buck_v1"`)

- `Scenario`:
  - The new pair `("PI", "buck_v1")` is valid.
  - New fields, all optional with default None: `buck_vout_ref_v` (24 V), `buck_vin_v` (48 V), `buck_load_step_ohm`, `buck_load_step_time_s` (default: half of `duration_s`) and `buck_parameter_overrides` (only `l_h`, `c_f`, `r_l_ohm`, `i_max_a`).
  - The initial load reuses `load_resistance_ohm`.
  - The backend validates every field. It also rejects v_ref ≥ V_in, since a buck only steps down.
- `runner.build_buck(scenario)` is shared with the Gym environment, so both build the identical plant. `_run_buck_scenario` writes the `buck_*` columns, and every other plant's columns are NaN.
- `export_parquet.BUCK_COLUMNS` are appended only when present, so the DC motor, VSC and PMSM exports keep exactly their columns (`test_existing_plants_export_no_buck_columns`).
- The byte-for-byte datagen tests were not touched.
- `jobs/handlers.py`: the scenario sweep gets the metric `rms_voltage_error_v_second_half` for the buck. In `viz/views/experiment_runs.py`, the "New experiment" form offers "Buck (output voltage sweep)".

### 1.4 `SPADE/Buck-v0` (`envs/native.BuckEnv`)

- Action: duty cycle in [0, 1].
- Observation, `float32` with 4 values: i_L/i_max, v_out/V_in, v_ref/V_in, (v_ref − v_out)/V_in.
- Reset:
  - The reference is drawn uniformly from 5–40 V, and the load log-uniformly from 3–30 Ω, unless they are fixed.
  - Options for each episode: `vout_ref_v`, `load_resistance_ohm`, `load_step_ohm`, `load_step_time_s`.
- Terminated when |i_L| > 2·i_max (30 A) or the state is not finite. Truncated at the horizon (default 0.02 s = 1000 steps).
- Same reward terms as E1.
- The registry moves it from PLANNED to ENVIRONMENTS. The new profile `native:buck_v1` declares stepping and reset, but not RL training (E8).

### 1.5 Page `Simulation / DC-DC Converters` (spec §8.1)

- Layering: `viz/adapters/dcdc_adapter.py` → `viz/views/dcdc_converters.py` → `viz/pages/dcdc_converters.py`, added to `app.py` under Simulation.
- Only the Buck tab exists. DAB comes with E9, as the list says.
- Same Generate pattern as the VSC / DPC page:
  - Controls on the left: operating conditions, an optional load step, and plant parameters in an expander.
  - Nothing is drawn before a run. The button is disabled while the controls match the shown run, and a notice appears if they change.
- After a run:
  - KPIs from `validation/step_metrics`: start-up settling and overshoot, steady-state error, load-step dip and recovery, final operating point.
  - Charts: v_out vs v_ref, i_L vs i_ref, and the duty cycle.
  - A note that the model is averaged, and a warning if i_L passes its rating.
  - A provenance table and the parameter table.
  - CSV export.
- The run is recorded in the session and mirrored to the durable store (E0).
- Screenshots in Chrome in dark and light mode after Generate (not committed). Measured on the page: settling 1.34 ms, error 0.79 mV, dip 1.98 V, recovery 1.70 ms, final 24.00 V / 8.00 A / duty 0.508.

## 2. Acceptance

| Criterion of the list | Evidence | Result |
|---|---|---|
| Steady state against the analytic solution, Vout = D·V_in in CCM | `test_steady_state_is_d_times_vin_without_winding_resistance` (D = 0.25, 0.5, 0.8, r_L = 0: relative error < 1e-9); `test_steady_state_with_winding_resistance` (D·V_in·R/(R + r_L)); `test_load_step_rediscretizes` | PASS |
| Transient against `solve_ivp`, with the error measured in the docstring | `test_transient_matches_solve_ivp`: a duty step 0 → 0.5 from rest, 1000 steps (20 ms), Radau with rtol 1e-10. Measured max error: 3.0e-10 V on v_C and 3.2e-10 A on i_L, written in the docstring of `sim/buck_system.py`. The test bound is 1e-8. | PASS |
| `BuckSystem` with `reset()`/`simulate(duty)` and a configurable τ; averaged vs switched documented | Module docstring; `test_duty_is_clipped_and_bad_parameters_rejected` | PASS |
| Voltage PI controller and `SPADE/Buck-v0` | `tests/control/test_buck_controller.py` (5 tests); `check_env` passes on `SPADE/Buck-v0`; `test_buck_env_reproduces_run_scenario_bit_for_bit` with and without a load step (`==` at each of 1000 steps) | PASS |
| `buck_v1` in Scenario and runner.py without changing the rows of the existing plants | Byte-for-byte datagen tests untouched and green in the full suite; `test_existing_plants_export_no_buck_columns` | PASS |
| Page following the Generate pattern | `tests/test_dcdc_page.py` (7 tests: empty state, Generate runs the real scenario with 3 charts, button disabled when unchanged, metrics, validation, overrides, sweep through the worker); `dcdc_converters` added to `tests/test_ui_contract.py` | PASS |

## 3. Tests

| File | Tests |
|---|---|
| `tests/simulation/test_buck_system.py` | 7 |
| `tests/control/test_buck_controller.py` | 5 |
| `tests/envs/test_native_envs.py` | +6 (Buck in the 3 parametrized contract tests, 2 equivalence cases, 1 reset and termination test) |
| `tests/test_dcdc_page.py` | 7 |
| `tests/test_ui_contract.py` | `dcdc_converters` added |
| `tests/runs/test_store.py`, `tests/test_environments_page.py` | Updated: Buck now steps and is registered; PMSM is the unregistered example |

Full suite on the branch: **871 passed, 0 failed** (656 s; 845 after E1 + 26 new) — `docs/audit/expansion/e6_branch_pytest.txt`.

## 4. Limits

- **Averaged model:** no ripple, no switching losses, no dead time, no ESR, no DCM. This is enough for control and RL at frequencies well below f_sw. It is not enough for ripple or loss design, which PLECS (E2–E5) is meant to cover; this buck is the reference those tasks will compare against.
- **Controller tuning:** tuned for the default parameters and checked there. With other L, C or R the gains follow the formulas, but performance was not measured outside the default point.
- **Faults and labels:** there is no fault model for the buck. Every record is labelled "normal".
- **Plant parameters are not identified against real hardware:** they are a representative 48 V → 24 V / 100 W design.
