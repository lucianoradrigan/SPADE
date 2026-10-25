# B3 — OpenModelica FMU feasibility (GO / NO-GO): results

**Pre-registration:** `b3_preregistration.md`. It was committed as `f3c9c94` and pushed to `origin/feat/B3-om-fmu-spike` before this script ran. When B3 was rebased onto dev after X2, that commit became `61fda95`: the file content is identical (`git diff f3c9c94 61fda95 -- docs/audit/plecs_om/b3_preregistration.md` is empty), and the author date is kept.
**Script:** `experiments/report_b3_om_fmu_spike.py`. Command:

```
SPADE_FMU_DIR=../SPADE-dev/data/fmu PYTHONPATH=src python experiments/report_b3_om_fmu_spike.py --out docs/audit/plecs_om
```

**Raw results:** `b3_om_fmu_spike.json`. It includes `data/fmu/manifest.json`, the FMU SHA-256 values and the `.mo` hash `ec5e4545c2a99398…`.
**Host:** Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13. Wall time 23 s.

**Execution note:** the first run stopped at the first figure, because matplotlib is not installed and is not a dependency. Nothing had been measured or written by then. The figures were rewritten as plain SVG, with no new dependency, and the script was rerun.

## Verdicts

| Model | Verdict | Failed criteria | Key result |
|---|---|---|---|
| **BuckAveraged** | **GO** | — | NRMS 1.7e-06 (v_out) / 2.1e-06 (i_L) vs native; p50 3.0 µs per doStep |
| **DcMotorAveraged** | **NO-GO** (rule) | 4 | i_a NRMS 2.6 % > 1 %, while omega is 0.11 %. The difference is the native plant's Euler step, not the FMU (§ Criterion 4) |
| **BuckSwitching** | **NO-GO** | 1, 2, 3, 5, 7 | does not simulate: CVODE aborts at t = 0.48 ms at Ts and at 0.175 ms at Ts/4 (S-15) |
| **DcMotorSwitching** | **NO-GO** | 1, 2, 3, 5, 7 | does not simulate: aborts at 0.8 ms at Ts and at 2.475 ms at Ts/4 (S-15) |

All four verdicts follow the pre-registered rules unchanged. The switching NO-GO was expected in the pre-registration.

## Criteria per model

| # | BuckAveraged | DcMotorAveraged | BuckSwitching | DcMotorSwitching |
|---|---|---|---|---|
| 1 Action affects output; same sequence gives the same output | pass | pass | fail (does not simulate) | fail |
| 2 t = n·Ts | pass (2.2e-16 rel.) | pass (2.2e-16) | fail | fail |
| 3 Reproducible reset | pass, bit-identical | pass, bit-identical | fail | fail |
| 4 Averaged vs native ≤ 1 % NRMS | pass: v_out 1.7e-06, i_L 2.1e-06 | **fail:** omega 1.12e-03, **i_a 0.026** | — | — |
| 5 Cycle average vs native ≤ 3 % | — | — | fail (does not simulate at Ts/4) | fail |
| 6 Latency p50 / p99 | 3.0 / 8.5 µs (≤ 0.5 ms: pass) | 7.2 / 42.1 µs (pass) | 18 µs over the steps before the abort | 83 µs over the steps before the abort |
| 7 10 × 1 s episodes, no failure or NaN | pass (≈ 0.9 s wall each) | pass (≈ 0.35 s wall each) | fail (all 10 abort at 0.48 ms) | fail (all 10 abort at 0.8 ms) |

**One-quadrant converter, DcMotorAveraged:** the converter's i_a < 0 branch was taken in 930 of 6000 steps by the native plant and in 925 by the FMU driver. Those steps come from the PRBS falls, and the pre-registration said this would be reported.

**Figures** (`b3_figures/`, SVG; native solid, FMU dashed):
- `BuckAveraged.svg`;
- `DcMotorAveraged.svg`;
- `DcMotorAveraged_vs_gem_v4.svg`;
- `DcMotorAveraged_vs_gem_v4_applied_voltage.svg`.

The switching models have no figure: they produce no trajectory.

## Criterion 4 of DcMotorAveraged: what the difference is

As pre-registered, the model was not adjusted.

**Parameters, checked against the native code:** r_a, l_a, psi_e, j_rotor, j_load, a, b, c, τ_decay, the 65 V supply and the converter rule. They match: B1 has the table with sources. The difference persisted, so the cause was looked for.

**Post-hoc diagnostic, declared and outside the verdict:**
- The native plant was run with the same parameters and sequence, but with `EulerSolver(nsteps=100)` instead of the production single Euler step per 100 µs.

| Comparison | omega NRMS | i_a NRMS |
|---|---|---|
| FMU vs native, production (Euler, 1 step) | 1.12e-03 | 2.60e-02 |
| FMU vs native with Euler × 100 | 2.6e-06 | 2.0e-05 |
| Native Euler × 1 vs native Euler × 100 | 1.12e-03 | 2.63e-02 |

**Conclusion:**
- **The FMU is correct.** It reproduces the native equations to 2e-5 once the native plant is integrated finely.
- **The whole 2.6 % comes from the native plant's single forward-Euler step.** The step is h = 100 µs, and the armature time constant is l_a/r_a = 1.19 ms. So the current transients after each duty step, and each converter branch change, are off by a few percent.
- **This is consistent with the existing §2 row "Production-step accuracy" of `SIMULATION_STABILITY_REPORT.md`.** That row gives 1.6 % of i_max in closed loop, and it passed its own criterion.
- **Here it is registered as S-16 (Info), a finding of the native plant.** Per §9, the native plant is not changed.
- **The GO/NO-GO rule is not reinterpreted:** DcMotorAveraged stays **NO-GO** under the registered criterion 4. Whether to accept it is the user's decision; see the proposal below.

## DcMotorAveraged vs gym-electric-motor 3.0.3 (V4), reported without a target

| Input to the FMU | omega NRMS | i_a NRMS |
|---|---|---|
| v_a = 420 · action (**as pre-registered**) | 0.275 | 0.129 |
| v_a = the voltage GEM applied (its `u` state); post hoc, declared | 8.7e-07 | 4.9e-07 |

**The pre-registered input was wrong.**
- GEM's `ContTwoQuadrantConverter` is a two-quadrant converter in **current**, not in voltage.
- For the 500 steps with action −0.4, it applied u = 0 V. The FMU instead received −168 V.
- With the voltage GEM actually applied, the FMU matches GEM's RK45 reference to about 1e-6.

This is an error of the pre-registration's set-up, not of the FMU. It is reported here as a deviation.

## Proposal for the user (B4 waits for your OK)

| Model | Proposal |
|---|---|
| **BuckAveraged** | **GO.** Continue B4–B7 with it: the `SPADE/BuckFmu-v0` environment and the GUI engine "OpenModelica FMU" |
| **DcMotorAveraged** | Formally NO-GO (criterion 4), and the cause is the native plant's Euler step. Options: **(a)** leave it out of B4 (it stays NO-GO); **(b)** you accept it as GO with this explanation. With (b), the FMU would be a more accurate DC-motor plant than the native one, and the native plant is not changed |
| **BuckSwitching, DcMotorSwitching** | **NO-GO** (S-15). Not published as Gym environments. A next step would need your decision, for example another OpenModelica version or a fixed-step solver, and it would be a new round |
