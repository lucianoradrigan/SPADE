# F1.6 — objective mode (a)

**Simulated results; no language model.** Tuning with SPADE's fixed objective `J` (PSO, 48 evaluations = Tuning Studio's default, seed 0) on a real worker; each user threshold is then checked on the **evaluation grid** (separate from the tuning grid). Results are candidates: no default changes.

- **Command:** `python experiments/report_f1_6_objective.py`. Raw values: [`f1_6_objective.json`](f1_6_objective.json). Tests: `tests/assistant/test_assistant_objective.py`.
- **Code:** `src/driveflow/assistant/objective.py` (`OBJECTIVE_TOOLS`: describe_tunable, update_objective, propose_tuning, run_tuning, check_requirements). Tuning runs planned by the assistant are labelled `SIMULATED · AI-PLANNED` (`jobs.handlers._origin`).

## Case A: `dc_perm_ex_v1`

> Busca un PI para el motor DC de stock con sobrepaso menor a 5 % y asentamiento menor a 50 ms.

Run `run-20261102T224223-b143bce5` (SIMULATED · AI-PLANNED), 20 s wall clock. Evaluation grid: step_5rad_s_at_120, saturating_step_30rad_s_at_100, drift_step_2rad_s_at_150, drift_load_15Nm_at_200.

| Controller | Parameters | overshoot_pct < 5 % | settling_time_s < 50 ms |
|---|---|---|---|
| baseline | ω_bw 80.0, ζ 0.900, b 1.000 | not met: worst 20.33 %, margin -15.3 | not met: worst 0.0775 s, margin -0.0275 |
| best | ω_bw 150.2, ζ 1.044, b 0.655 | **met**: worst 4.874 %, margin +0.126 | not met: worst 0.0532 s, margin -0.0032 |

**Verdict:** no candidate meets all 2 requirements; best compromise 'best' meets 1.

## Case B: `dc_pmg132_v1`

> Para el PMG 132 quiero sobrepaso menor a 2 % y corriente pico de 150 A como máximo.

Run `run-20261102T224243-0927719e` (SIMULATED · AI-PLANNED), 20 s wall clock. Evaluation grid: step_5rad_s_at_120_scaled, saturating_step_30rad_s_at_100_scaled, drift_step_2rad_s_at_150_scaled, drift_load_15Nm_at_200_scaled.

| Controller | Parameters | overshoot_pct < 2 % | peak_current_a <= 150 A |
|---|---|---|---|
| baseline | ω_bw 80.0, ζ 0.900, b 1.000 | not met: worst 2321 %, margin -2.32e+03 | not met: worst 196.1 A, margin -46.1 |
| best | ω_bw 150.2, ζ 1.044, b 0.655 | not met: worst 5.154 %, margin -3.15 | not met: worst 160 A, margin -10 |

**Verdict:** no candidate meets all 2 requirements; best compromise 'best' meets 0.

Worst case = the largest value over the evaluation-grid conditions that measure the metric (step conditions for overshoot and settling, every condition for peak current); a step that never settles fails a settling threshold. A negative verdict is a deliverable, not an error.

**Case B is affected by finding S-21** (`s21_one_quadrant.md`): on a motor with the 0b-D1 friction the one-quadrant drive does not coast down after an overshoot but speeds up, so the baseline's "overshoot" of thousands of % on the scaled grid is that model effect, not the PI. The tools report it faithfully; the numbers of case B are not a valid tuning result until S-21 is resolved.

## Scope decisions

- Only the DC-motor speed PI is tuned from the chat (`dc_perm_ex_v1` with any motor; `dc_pmg132_v1` passed as its datasheet values, identical simulation checked by a test). The buck PI stays in Tuning Studio; MPC, DPC and FOC are not tunable and the assistant says so.
- Thresholds must be quantities the user wrote (same no-invention rule as the draft); the budget comes from the user or from Tuning Studio's default.

## Not executed

- `J_spec` with the user's constraints (phase 3).
- The same conversation with a real local model: no ejecutado (F1.1).
