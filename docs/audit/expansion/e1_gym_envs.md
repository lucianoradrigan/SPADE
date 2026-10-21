# E1: Gymnasium environments for the native plants + Environments page

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E1. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §3.4 stage 5C–D, §3.7, §8.5.
Branch: `feat/E1-gym-envs` from `dev` (07986df, E0 merged). Date: 2026-10-21. Machine: Linux x86-64, Python 3.12.13, Gymnasium 1.4.0, Streamlit 1.66.0.

Origin of the design: Fund. `7_Reinforcement_Learning/RL_buck_control.ipynb` (`BuckAvgEnv`): normalized observation, reset with seeded random conditions, reward made of terms. Only the design was taken; no code was copied (that class does not even subclass `gymnasium.Env`).

## 1. What was built

### 1.1 `src/driveflow/envs/native.py`

- `DcMotorEnv` (plant `dc_perm_ex_v1`): speed tracking.
  - The plant comes from `datagen.runner.build_plant`, the same function `run_scenario` uses. It is rebuilt and seeded on every `reset` exactly like `_run_dc_motor_scenario` (`plant.seed(SeedSequence(seed))`, then `plant.reset()`).
  - Action: duty cycle, `Box(0, 1, (1,), float64)`.
  - Observation, `float32` with 5 values: ω, i, u, ω_ref and ω_ref − ω, each divided by the plant limit (400 rad/s, 210 A, 60 V).
  - Terminated when |ω| > 400 rad/s, |i| > 210 A or the state is not finite.
- `VscEnv` (plant `vsc_dpc_v1`): rotating voltage reference.
  - The plant is `VscSystem`, or `DegradedVscSystem` when a capacitor scale or an ESR is given, with the same switch as `_run_vsc_scenario`. The phase of the reference follows the seed with the runner's formula.
  - Action: (v_o_real, v_o_imag) in V, `Box(±voltage_limit_v)`. The default limit is 2000 V, because the native model has no converter rating and the DPC start-up transient reaches about 1000 V at 100 V / 8 Ω (measured: 572 V at 50 V / 8 Ω and 1864 V at 100 V / 1 kΩ).
  - Observation, `float32` with 6 values: i_f, v_c and the reference of the next step. Voltages are divided by 100 V, currents by `current_limit_a` (40 A).
  - Terminated when |v_c| > voltage limit or the state is not finite.
- Common contract:
  - Episode horizon:
    - Truncated at the horizon, with `info["truncation"] = "horizon"`.
    - Defaults: 0.5 s (5000 steps) for the DC motor and 0.1 s (1000 steps) for the VSC.
  - Reset:
    - `reset(seed, options)`: the options override the reference and the load for one episode.
    - With no reference fixed, it is drawn from the seeded generator: uniform for the DC motor; magnitude uniform and load log-uniform for the VSC.
  - Actions out of bounds:
    - They are clipped ("Clip and log", spec §3.4 D), and `info["action_applied"]` and `info["action_clipped"]` show what reached the plant.
    - A NaN or infinite action raises `ValueError`.
  - Units are in `metadata`: `observation_names`, `observation_units`, `observation_scales`, `action_names`, `action_units`, `limits` and `tau_s`. `info` carries the physical values.
- The plant physics was not modified. One `step()` is exactly one `simulate()` call of 100 µs.

### 1.2 `src/driveflow/envs/reward.py`

- `RewardTerm(kind, weight, enabled, threshold)` with three kinds from spec §3.4 C:
  - `tracking`: w·e²;
  - `overcurrent`: w·max(0, |i|/i_limit − threshold)²;
  - `action_rate`: w·(Δa / action width)².
- All signals are dimensionless. The reward is `−Σ`.
- `parse_terms` validation:
  - weights must be finite and ≥ 0;
  - at least one term must have a positive weight;
  - each kind may appear only once, which avoids counting the same signal twice.
- The per-step breakdown is in `info["reward_terms"]`.
- Default terms: `tracking` 1.0, `overcurrent` 10.0 (threshold 0.8) and `action_rate` 0.01.
- The same schema is meant for E4 (PLECS) and E8 (RL Lab).

### 1.3 `src/driveflow/envs/registry.py`

- Stable IDs `SPADE/DcMotor-v0` and `SPADE/Vsc-v0`, registered with `gymnasium.register` when `driveflow.envs` is imported (idempotent).
- `PLANNED` lists `SPADE/Buck-v0` (E6), `SPADE/Pmsm-v0` (no task) and PLECS (E3/E4). They are never registered.
- `describe(env_id)` reads the contract from a real instance.
- `snippet(env_id)` exists only for registered IDs and raises `KeyError` otherwise.
- `verify(env_id)` runs `gymnasium.utils.env_checker.check_env` and a determinism check: two fresh instances with the same seed and actions must give identical observations and rewards. It returns an `EnvironmentCapability` with `validated_at`.
- `can_train_rl` stays False until there is a trainer (E8).
- `runs/profiles.py`: the native DC and VSC profiles now declare `can_step_interactively` and `can_reset_episode`, with the test that backs them. The PMSM stays batch-only.

### 1.4 Page `Workspace / Environments` (spec §8.5)

- Files: `viz/adapters/environments_adapter.py` → `viz/views/environments.py` → `viz/pages/environments.py`. The page is added to `app.py` under Workspace.
- Table with the spec's columns: plant, version, source, observation (shape and units), action (type and bounds), default reward, physical limits, step and horizon, terminations, compatible algorithms.
  - Rows for planned environments are marked "Not registered (task: …)", with no snippet.
- One expander per environment with four tabs:
  - Contract.
  - Reward.
  - Capabilities: the one declared on the profile, plus a "Run contract check now" button that runs `verify` live.
  - Python snippet: an `st.code` block with a copy button.
- Screenshots in Chrome, dark and light, with the contract check run (not committed). No chart is drawn: the page shows contracts, not runs.

## 2. Acceptance

| Criterion of the list | Evidence | Result |
|---|---|---|
| `check_env` passes for each environment | `test_check_env_passes[SPADE/DcMotor-v0]`, `[SPADE/Vsc-v0]` | PASS. Gymnasium leaves three advisory warnings. Two say the observation `Box` is unbounded; it is, on purpose: the bounds are the termination limits, not clipping. The third recommends a normalized action space for the VSC; the action is in volts so that it equals the controller's action exactly. |
| With the action of runner.py's controller, the environment reproduces `run_scenario` bit for bit | `test_dc_motor_env_reproduces_run_scenario_bit_for_bit`: PI cascade, ω_ref 150 rad/s, seed 3, 0.2 s, comparing rpm, current, voltage and torque. `test_vsc_env_reproduces_run_scenario_bit_for_bit`: scheduled DPC (the VSC's controller in runner.py; the VSC has no PI), at 8.0064 Ω / 50 V / seed 0 and 20 Ω / 30 V / seed 45, comparing v_c, i_f, v_ref and v_o. Every value is compared with `==` at each of the 2000 and 500 steps. | PASS |
| Stable IDs and an EnvironmentCapability for each | `test_registered_ids_and_verify`, `test_planned_ids_are_not_registered` | PASS |
| `simulation` marker | `pytestmark = pytest.mark.simulation` in `tests/envs/test_native_envs.py` | PASS |
| The page shows the two environments with their contract; snippet only for registered IDs | `test_page_renders_both_environments_and_planned_rows`, `test_snippets_only_for_registered_ids`, `test_contract_check_button_runs_check_env`, `test_snippet_runs` (runs the snippet) | PASS |

## 3. Tests

| File | Tests |
|---|---|
| `tests/envs/test_native_envs.py` | 17 |
| `tests/test_environments_page.py` | 4 |
| `tests/test_ui_contract.py` | `environments` added to the page list |
| `tests/runs/test_store.py` | Assertion updated: DC and VSC now step; the PMSM does not; none trains RL |

Full suite on the branch: **845 passed, 0 failed** (604 s; 823 after E0 + 22 new) — `docs/audit/expansion/e1_branch_pytest.txt`.

## 4. Limits

- The VSC action is in volts with a wide bound (±2000 V). An RL agent will need `gymnasium.wrappers.RescaleAction` (E8). The bound is not a converter rating; the native model has none (S-09).
- The VSC's `current_limit_a` (40 A) only sets the observation scale and the overcurrent term. It does not terminate the episode, because the model has no current rating either.
- The DC environment controls speed only; `torque_ref_nm` from the runner is not exposed.
- No environment for the PMSM (no task asks for it) or the Buck (E6 will add it to the registry).
- `can_train_rl` = False on both until E8.
