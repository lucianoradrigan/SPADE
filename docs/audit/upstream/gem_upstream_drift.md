# gym-electric-motor after 3.0.3: upstream drift (V6)

TODO_CLAUDE_CODE.md V6. Date: 2026-10-20.

## Question

SPADE's `sim/` package ports GEM 3.0.3. The ported parts are:
- the motors, converters, mechanical loads and voltage supplies;
- the ODE solvers;
- `physical_systems.py` (the SCML system).

Did GEM fix any physics or bugs in those files after the port?

## Method

The GEM clone (`../gym-electric-motor`) was fetched with all branches and tags. PyPI was queried for the latest release.

```bash
git -C ../gym-electric-motor fetch --tags origin
git -C ../gym-electric-motor log --oneline v3.0.3..origin/master                 # 0 commits
curl -s https://pypi.org/pypi/gym-electric-motor/json                            # latest: 3.0.3
for b in $(git -C ../gym-electric-motor branch -r); do
  git -C ../gym-electric-motor diff --name-only v3.0.3...$b -- src/gym_electric_motor/physical_systems
done
```

## Result

**No upstream drift exists in released code.**
- `origin/master` is the `v3.0.3` tag itself (commit `b474daf9`, 2025-12-19).
- 3.0.3 is also the latest release on PyPI (uploaded 2025-12-19).
- There are no commits after 3.0.3 on `master` or `nightly`.

**Unreleased branches that change `physical_systems/` relative to 3.0.3:** only one, `origin/EESM_transformations` (last commit 2024-06-13, never merged). It changes the default parameters and limits of `ExternallyExcitedSynchronousMotor`, from a different reference paper, and makes the rotor-to-stator parameter transformation optional. It is a model change, not a bug fix.

**Does it affect SPADE?** No:
- the EESM is ported in `sim/motors/externally_excited_synchronous_motor.py`, but no SPADE plant, scenario, experiment or test uses it;
- the branch is not part of any GEM release.

**Branches active after 3.0.3** (`dtc`, `274-…model-predictive-current-control…`, `VariableFrequency_development`, `277-…EESM-torque-control`, `Syntax_change`, `sphinx_conroldoc`) have no net change to `physical_systems/` relative to 3.0.3. They concern controllers, examples and documentation.

**What V4 adds.** The numerical side of the port is pinned separately: `tests/simulation/test_gem_equivalence.py` (V4) checks bit-for-bit agreement with GEM 3.0.3 on the DC motor and the PMSM.

## Recommendation

Nothing to incorporate. Re-run the commands above when GEM publishes a release after 3.0.3.
