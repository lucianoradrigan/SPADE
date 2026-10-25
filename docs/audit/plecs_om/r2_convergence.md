## R2.1 Convergence of the reference (decision G7)

Finest solution: RK45, rtol = atol = 1e-09. Rule: reference = the coarsest Euler substep count whose NRMS vs the finest solution is < 1e-4 on i_a, omega and torque, in every scenario and fault case.
Normalisation: primary: RMS(diff over the segment) / range of the finest solution over the whole run; secondary (reported only): / RMS of the finest solution over the segment.

Speed reference 150 rad/s, PI controller, seed 0. Load step: 4 N·m at 1 s (validation benches' `LoadTorqueStep`).

| Scenario | Fault | Integration | i_a | omega | torque | i_a (segment-RMS norm.) |
|---|---|---|---|---|---|---|
| startup | healthy | euler_x1 | 6.81e-04 | 1.58e-04 | 6.81e-04 | 8.07e-04 |
| startup | healthy | euler_x10 | 7.51e-05 | 1.44e-05 | 7.51e-05 | 8.89e-05 |
| startup | healthy | euler_x100 | 7.63e-06 | 1.42e-06 | 7.63e-06 | 9.04e-06 |
| startup | healthy | euler_x1000 | 1.30e-06 | 1.35e-07 | 1.30e-06 | 1.54e-06 |
| startup | outer_race | euler_x1 | 7.01e-03 | 5.35e-04 | 7.01e-03 | 9.03e-03 |
| startup | outer_race | euler_x10 | 6.88e-04 | 5.21e-05 | 6.88e-04 | 8.86e-04 |
| startup | outer_race | euler_x100 | 7.23e-05 | 5.35e-06 | 7.23e-05 | 9.31e-05 |
| startup | outer_race | euler_x1000 | 1.09e-05 | 6.91e-07 | 1.09e-05 | 1.40e-05 |
| steady_state | healthy | euler_x1 | 1.62e-13 | 1.25e-14 | 1.62e-13 | 4.44e-13 |
| steady_state | healthy | euler_x10 | 1.68e-12 | 6.02e-15 | 1.68e-12 | 4.59e-12 |
| steady_state | healthy | euler_x100 | 1.87e-11 | 6.94e-14 | 1.87e-11 | 5.10e-11 |
| steady_state | healthy | euler_x1000 | 1.91e-10 | 1.92e-13 | 1.91e-10 | 5.21e-10 |
| steady_state | outer_race | euler_x1 | 8.56e-03 | 3.31e-04 | 8.56e-03 | 2.53e-02 |
| steady_state | outer_race | euler_x10 | 8.42e-04 | 3.25e-05 | 8.42e-04 | 2.49e-03 |
| steady_state | outer_race | euler_x100 | 8.87e-05 | 3.43e-06 | 8.87e-05 | 2.62e-04 |
| steady_state | outer_race | euler_x1000 | 1.36e-05 | 5.24e-07 | 1.36e-05 | 4.02e-05 |
| load_step | healthy | euler_x1 | 1.09e-04 | 3.06e-06 | 1.09e-04 | 2.04e-04 |
| load_step | healthy | euler_x10 | 2.76e-04 | 8.38e-06 | 2.76e-04 | 5.18e-04 |
| load_step | healthy | euler_x100 | 2.47e-04 | 7.55e-06 | 2.47e-04 | 4.64e-04 |
| load_step | healthy | euler_x1000 | 2.45e-04 | 7.47e-06 | 2.45e-04 | 4.59e-04 |
| load_step | outer_race | euler_x1 | 8.55e-03 | 3.31e-04 | 8.55e-03 | 1.76e-02 |
| load_step | outer_race | euler_x10 | 8.73e-04 | 3.34e-05 | 8.73e-04 | 1.80e-03 |
| load_step | outer_race | euler_x100 | 2.38e-04 | 8.21e-06 | 2.38e-04 | 4.90e-04 |
| load_step | outer_race | euler_x1000 | 2.19e-04 | 7.44e-06 | 2.19e-04 | 4.52e-04 |

| Integration | Worst NRMS vs RK45 | < 1e-4 |
|---|---|---|
| euler_x1 | 8.56e-03 | no |
| euler_x10 | 8.73e-04 | no |
| euler_x100 | 2.47e-04 | no |
| euler_x1000 | 2.45e-04 | no |

**No Euler substep count reaches the tolerance: R2 stops here (R2.1).**
## Why no Euler reference converges in the load-step scenario (finding S-17)

- In start-up and steady state, healthy and faulted, Euler × 100 is within 1e-4 of RK45 (worst 8.9e-5) and Euler × 1000 within 1.4e-5: the plant converges as expected, at first order.
- In the load-step scenario Euler × 10, × 100 and × 1000 stop at the same floor (2.2–2.8e-4); Euler × 1 is closer to RK45 than × 10. That is not a discretization error.
- Cause: `EulerSolver._integrate_nsteps` (inherited from gym-electric-motor) sets `current_t = t`, the **end** of the step, and evaluates f at t + k·h/n. With more than one substep, a right-hand side that depends on t sees time one control period late. The load step of this scenario is triggered by t, so Euler with substeps applies it one period early.
- Check (`python experiments/report_s16_discretization_sensitivity.py load_step_offset --workers 4` → `r2_load_step_offset.json`): Euler × 1000 with the step at 1.0 s against RK45 with the step at 1.0 s: 4.2e-4; half a period earlier: 2.1e-4; one period earlier: **2.2e-6**.
- The diagnosis plant has no time-dependent term (the bearing fault phase is an ODE state), and the PMSM path that uses 10 substeps has a time-independent load, so production outputs are not affected. Registered as **S-17** (strict xfail `tests/test_r2_dc_plant_integration.py::test_s17_euler_substeps_evaluate_inside_the_step`).

## Status

**R2 stops at R2.1, as the round's instructions require when the reference does not converge.** The rule (coarsest Euler within 1e-4 of the finest in all three scenarios) is not met because of S-17, not because the plant fails to converge. No diagnosis metric was computed and the pre-registration was not written. Options are put to the user; none is taken.

## Continuation, 2026-10-25 (user decision: option a, RK45 reference)

Command: `python experiments/report_s16_discretization_sensitivity.py rk45_convergence --workers 8` → `r2_rk45_convergence.json`.


Reference: rk45, rtol = atol = 1e-9. Shown converged against rk45, rtol = atol = 1e-11 (G7); Euler × 1000 is listed for comparison.

| Scenario | Fault | RK45 1e-9 vs 1e-11: i_a / omega / torque | Euler × 1000 vs RK45 1e-11: i_a / omega / torque |
|---|---|---|---|
| startup | healthy | 9.6e-07 / 2.9e-08 / 9.6e-07 | 7.5e-07 / 1.4e-07 / 7.5e-07 |
| startup | outer_race | 4.1e-06 / 1.9e-07 / 4.1e-06 | 6.9e-06 / 5.2e-07 / 6.9e-06 |
| steady_state | healthy | 7.8e-15 / 5.6e-16 / 7.8e-15 | 1.9e-10 / 1.9e-13 / 1.9e-10 |
| steady_state | outer_race | 5.3e-06 / 2.0e-07 / 5.3e-06 | 8.4e-06 / 3.2e-07 / 8.4e-06 |
| load_step | healthy | 1.2e-06 / 2.9e-08 / 1.2e-06 | 2.4e-04 / 7.5e-06 / 2.4e-04 |
| load_step | outer_race | 6.4e-06 / 2.4e-07 / 6.4e-06 | 2.2e-04 / 7.4e-06 / 2.2e-04 |

Worst RK45 (1e-9) vs RK45 (1e-11): 6.4e-06 (< 0.0001: yes). **The RK45 reference is converged; R2 continues with it.**

