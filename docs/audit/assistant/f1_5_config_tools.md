# F1.5 — configuration-mode tools

**Simulated results; no language model** (the tool calls are scripted). The job ran on a real SPADE worker in a temporary workspace.

- **Command:** `python experiments/report_f1_5_config_tools.py`. Raw values: [`f1_5_config_tools.json`](f1_5_config_tools.json). Tests: `tests/assistant/test_assistant_tools.py`.
- **Code:** `src/driveflow/assistant/tools.py` (`CONFIG_TOOLS`): recommend_plant, describe_plant, suggest_value, update_draft, validate_draft, estimate_cost, propose_experiments, run_experiments, get_job_status, summarize_results.

## End-to-end conversation

> Simula el motor DC de stock con PI a 50 y 100 rad/s, 0.5 s cada una, sin carga extra.

Proposal shown to the user: 2 experiments, estimated 3.2 CPU s. Launched only after the interface recorded the user's "sí": job `job-20261102T223053-a679f01c` → run `run-20261102T223053-6c0131b7`, status **completed**, origin `SIMULATED · AI-PLANNED`.

| # | ω_ref (rad/s) | Final speed (rad/s) | Rise time (s) | Settling time (s) | Overshoot (%) | Steady current (A) | Peak current (A) |
|---|---|---|---|---|---|---|---|
| 0 | 50 | 50.000 | 0.0418 | 0.0552 | 1.503 | 15.212 | 168.00 |
| 1 | 100 | 100.000 | 0.0874 | 0.11 | 0.487 | 30.364 | 168.00 |

No-invention replay of the session (`audit_trace`): 0 problems. KPIs come from the run's parquet files through `validation.step_metrics` (rise 10–90 %, settling ±2 % band, overshoot % of the step).

## recommend_plant

| User's description | Recommended | Not simulable (closest substitute) |
|---|---|---|
| motor DC de imanes permanentes de 220 V | `dc_perm_ex_v1` | — |
| PMG 132 | `dc_pmg132_v1` | — |
| servo brushless PMSM | `pmsm_v1` | — |
| inversor trifásico con filtro LC | `vsc_dpc_v1` | — |
| buck de 48 V a 12 V | `buck_v1` | — |
| motor de inducción de 4 polos | — | induction motors are not simulated in SPADE today (planned after phase 2, decision D3) → `pmsm_v1`, `dc_perm_ex_v1` |
| convertidor boost | — | only the buck (step-down) DC-DC converter exists in SPADE → `buck_v1` |
| un sistema de bombeo | — | — |

## suggest_value: internal sources only (decision P5)

| Field (DC stock motor) | Suggested | Rule / why not |
|---|---|---|
| `omega_ref_rad_s` | 300.0 rad/s | the motor's rated speed |
| `load_torque_nm` | 16.0 N*m | the motor's rated torque (100 % load) |
| `duration_s` | 1.0 s | SPADE's default for this field |
| `dc_supply_v` | — (ask)  | no verifiable internal source for a suggestion; ask the user without suggesting a number (phase 1, P5) |
| `motor_parameter_overrides.l_a` | — (ask)  | machine parameters are not suggested; ask the user, or keep the stock motor with use_defaults |

## Observations

- `dc_pmg132_v1` has no native plant profile (`runs.profiles.NATIVE_PROFILES`) and no per-condition metric in `jobs.handlers._condition_metrics`; `run_experiments` launches it with `plant_profile_id = None` and `summarize_results` computes its KPIs from the parquet, so the assistant is not affected. Left as is (Phase 0b scope).

## Not executed

- The same conversation with a real local model: no ejecutado (F1.1).
