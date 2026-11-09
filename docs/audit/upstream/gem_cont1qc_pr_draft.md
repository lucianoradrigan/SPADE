# Draft pull request for upstream gym-electric-motor — LOCAL ONLY, NOT PUBLISHED

**Status (2026-11-09, T7):**
- Prepared in a local fork with no public remote: `/mnt/datos/upstream/gym-electric-motor`, branch `fix/cont1qc-negative-current`, commit `6eb727fc` on `b474daf9` (`master` = v3.0.3). The patch is [`gem_cont1qc_fix.patch`](gem_cont1qc_fix.patch).
- **Nothing was published** (stop point P1). Luciano decides whether, and when, the issue and the PR go upstream. The issue would go first: [`gem_cont1qc_issue_draft.md`](gem_cont1qc_issue_draft.md).

## Check: do other continuous converters decide by the current's sign?

Read on `b474daf9`, `src/gym_electric_motor/physical_systems/converters.py`:

| Converter | Decision on the current sign | Same defect? |
|---|---|---|
| `ContOneQuadrantConverter` (Cont1QC) | `_convert` returns 1 (full supply) when `i < 0` | **Yes** (this PR) |
| `ContTwoQuadrantConverter` (Cont2QC) | output = duty for both signs; only the interlocking discount uses `sign(i)` (dead-time voltage error, physically sign-dependent) | No |
| `ContFourQuadrantConverter` (Cont4QC) | two Cont2QC | No |
| `ContB6BridgeConverter` (Cont-B6C) | three Cont2QC (one per phase); the interlocking discount uses `sign(i)` per phase | No: bidirectional current is physical in a B6 bridge |
| `ContMultiConverter` | the decisions of its subconverters | Only if a subconverter is Cont1QC: the fork warns there, no clamp |
| `FiniteOneQuadrantConverter` (discrete, not continuous) | `convert` returns 1 when `i < 0` | Same logic; not changed in this PR (switched model, its own step) — noted in the issue |

The interlocking term is evaluated from the current at the start of the step in Cont2QC/B6C too. That is a step-held approximation of a real effect, not an impossible state, so it is out of scope.

## Results in the fork

| Command | Result |
|---|---|
| GEM suite before the change (`PATH=<venv>/bin:$PATH python -m pytest -q tests`) | **1226 passed** (the 7 example tests need `python` on `PATH`) |
| GEM suite with the change | **1230 passed** (1226 + 4 new), 0 failed |
| The 4 new tests against the unfixed code | 3 failed (coasting, braking, warning), 1 passed (continuous conduction unchanged, as intended) |
| Issue reproduction (`docs/audit/upstream/gem_cont1qc_mre.py`) with the fork | i min **0.000 A**, ω **150.00 rad/s** at 0.5 s (before: −144.683 A, 181.82 rad/s) |
| `python experiments/report_gem_cont1qc_fork.py` (braking at duty 0.2 for 25 ms, then duty 0.5; friction a = 0.01, b = 0.05) | i min 0.0 A for τ = 1e-4, 1e-5, 1e-6; ω at 50 ms 170.6524 rad/s for τ = 1e-4 and 1e-5, 170.6519 for 1e-6 (difference 4.9e-4 rad/s). The default solver is `ScipyOdeSolver`, so the clamp acts at the zero crossing inside the step |

Environment: Python 3.14.4, separate venv `/mnt/datos/upstream/gem-venv` (`pip install -e .` of the fork); SPADE's environment was not touched.

**Declared:** the existing upstream test `TestContOneQuadrantConverter.test__convert` asserted the old behaviour (`u == 1` for `i < 0`). The PR changes it to expect the duty cycle, with a comment.

**Not executed:** the GEM docs build (Sphinx); a PR against upstream (P1).

---

## Title

Cont1QC: clamp the armature current at zero (discontinuous conduction)

## Description

Fixes #<issue number> (the issue is opened first).

`ContOneQuadrantConverter._convert` returns the full supply voltage whenever the armature current is negative, and `SCMLSystem.simulate` holds that decision for the whole step. A one-quadrant (buck) converter cannot carry a negative current at all, and the full-voltage answer injects energy. An unloaded, frictionless `PermExDc` at duty 0 accelerates from 150 to 182 rad/s in 0.5 s, with the current down to −145 A.

Since #138 the DC environments default to a four-quadrant converter, so this only affects users who pass `ContOneQuadrantConverter` explicitly.

**Changes**

1. `ContOneQuadrantConverter._convert` returns the duty cycle for every current sign, and the class declares `blocks_negative_current = True`. The docstring and `docs/.../1QC.rst` explain discontinuous conduction.
2. `SCMLSystem` clamps that current at zero for motors with a single current (permanently excited, series):
   - in the ODE right-hand side, where `di/dt` is set to 0 when `i <= 0` and `di/dt < 0`;
   - after each integration, which a fixed-step solver needs.

   With the clamp active the analytic jacobian is not passed to the solver, because it ignores the clamp. For motors with two currents, or with Cont1QC inside a multi converter, nothing changes except a warning.
3. Tests, in `tests/test_physical_systems/test_cont1qc_discontinuous_conduction.py`:
   - duty 0 coasting: i ≥ 0, ω constant;
   - continuous conduction identical to Cont2QC (atol 1e-9);
   - braking with friction only slows the motor;
   - the warning for a two-current motor.

   `test__convert` now expects the duty cycle for a negative current.
4. Changelog entry under *Fixed*.

**Results:** the suite passes, 1230 tests (1226 before, plus the 4 new ones). The reproduction from the issue now gives i ≥ 0 and ω = 150.00 rad/s throughout.

**Not covered:**
- `FiniteOneQuadrantConverter` has the same `i < 0 → 1` logic in the switched model; it is left for a separate change.
- Shunt and externally excited motors with Cont1QC get a warning, not a clamp.
