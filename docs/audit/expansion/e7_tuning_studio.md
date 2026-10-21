# E7: Tuning Studio (PSO, Bayesian optimization, NSGA-II)

Round 2026-10-21, task T3 (decision D1). Branch `feat/E7-tuning-studio` from `dev` (a6bcb56), worked in a parallel worktree. Machine: Linux x86-64, Python 3.12.13.

Result report: [`e7_tuning_s04.md`](e7_tuning_s04.md) / `.json` (S-04 case).

## What was built

### Own optimisers, no new dependency (`src/driveflow/optim/`)

- **`pso.py`**: global-best PSO with inertia, clipped positions and velocities. Seeded. Every evaluation is recorded.
- **`bayes.py`**: GP surrogate (`sklearn.gaussian_process.GaussianProcessRegressor`, constant × Matérn 2.5 + white noise), expected improvement over seeded random candidates, and a Latin-hypercube start.
- **`multiobjective.py`**: NSGA-II (non-dominated sort, crowding distance, SBX, polynomial mutation, (μ+λ) survival). Includes ZDT1 and IGD for testing.
- **`objectives.py`**:
  - **Search space:** the PI speed loop of `PICascadeController`: bandwidth [40, 160] rad/s, damping [0.6, 1.2], setpoint weight b [0.3, 1.0]. The weight b was added by C2.
  - **Simulation:** each candidate runs on `validation.benches.run_dc_closed_loop`.
  - **Grids, both fixed in the module before optimising:** the TUNING grid (the S-04 2 rad/s step, a 10 N·m load step) and a distinct EVALUATION grid (a 5 rad/s step, a saturating 30 rad/s step, plant drift on a step and on a 15 N·m load step at 200 rad/s).
  - **Overlap check:** `grids_overlap()` triggers the spec's warning "The evaluation reused tuning conditions…".
  - **Objectives:** scalar J = overshoot + 0.1 × settling ms + 2 × load dip; NSGA-II uses the three terms as separate objectives. CPU time is recorded.

### Job and GUI

- **Job `controller_tuning`** (`jobs/handlers.py`):
  - validated spec: method, seed, bounds inside the allowed box, budget 4–2000;
  - progress and cancellation per evaluation;
  - artifacts `tuning_history.json` and `tuning_report.json`, the latter with the baseline and every candidate on both grids;
  - the result is always a candidate.
- **Page `Design & Control / Tuning Studio`** (new navigation section, spec §4), in five steps:
  1. **Select system:** the DC motor and its PI; other plants and controllers are not available in this version. Both grids are shown.
  2. **Search space:** a range per parameter.
  3. **Objectives & method:** method, budget, seed, and the objective formula.
  4. **Execute:** a job, with a worker check and job status.
  5. **Review & approve:**
     - best J against evaluations, with the baseline line;
     - the NSGA-II Pareto front with selection by click (`on_select`), plus a select box;
     - baseline vs. candidate on the evaluation grid, per condition;
     - **Save candidate**, which stores `saved_candidate_*.json` next to the run and changes nothing else;
     - **Use as default** through `ApprovalGate`. Its gate report always blocks with "Use as default is not authorized in this round (decision D1, 2026-10-21)", and the backend refuses such an approval (`RunStore.record_approval`).

## S-04 result (summary of `e7_tuning_s04.md`)

96 evaluations per method, seed 0, all three runs through the worker in 160 s.

| | J tuning | J evaluation | Overshoot, 5 rad/s step | Overshoot, drift step | Load dip with drift (rad/s) |
|---|---|---|---|---|---|
| Baseline (b = 1, 80 rad/s, ζ = 0.9) | 25.20 | 27.83 | 15.1 % | 20.3 % | 4.17 |
| PSO | 3.12 | 14.20 | 0.0 % | 5.3 % | 4.07 |
| Bayesian | 2.74 | 13.61 | 0.0 % | 5.1 % | 3.95 |
| NSGA-II (best J) | 3.28 | 13.34 | 0.0 % | 4.8 % | 3.90 |

**What the candidates do.** They remove the small-step overshoot with b ≈ 0.7–0.8 and a faster, more damped loop.

**Limit.** The objective has no control-effort, current-margin or noise term. A faster loop therefore always lowers the load dip, and the bandwidth goes to the upper bound (142–160 rad/s). The candidates are not evidence that such a loop is safe on a real drive. No default was changed.

## Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| PSO, NSGA-II and Bayesian optimization of our own, tested on Sphere, Rastrigin and ZDT1 with a seed | `tests/optim/test_optimizers.py` (9 tests). PSO Sphere < 1e-4 and Rastrigin < 1; Bayesian Sphere < 0.1 in 26 evaluations; NSGA-II IGD < 0.05 and ≤ 1/3 of random search on the same budget; identical histories with the same seed | PASS |
| Evaluation grid distinct and fixed before tuning; reuse warning | `objectives.EVALUATION_GRID`, `grids_overlap()` (False); `test_empty_state_and_fixed_grids` | PASS |
| Runs as E0 jobs | `test_tuning_job_end_to_end[pso, nsga2]` (worker in process); the report script also uses the worker | PASS |
| GUI: 5 steps, best-vs-evaluations curve, Pareto front with click selection, baseline vs. candidate | `tests/test_tuning_studio_page.py` (5 tests); `tuning_studio` in `tests/test_ui_contract.py` | PASS. Clicking on the Pareto points is wired with `on_select` but not driven by AppTest, which cannot click a chart; the select box next to it is tested. |
| Save candidate changes nothing; Use as default goes through ApprovalGate and is not authorized | Approve button disabled; the backend refuses a blocked approval (`test_blocked_approval_is_refused_by_the_backend`); saved file `status: candidate -- not a default` | PASS |
| Report `e7_tuning_s04.{md,json}`, also if there is no improvement | written; every method improves on the evaluation grid | PASS |

Full suite on the branch (rebased on 6a7d415, E14): **1038 passed, 4 xfailed (strict, findings), 0 failed** (775 s) — `e7_branch_pytest.txt`. Run from a second worktree with `PYTHONPATH=src` and the SPADE-dev virtual environment.
