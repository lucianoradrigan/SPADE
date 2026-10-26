# P1 — FMI 3.0 and Model Exchange in FmuAdapter (the interfaces a PLECS 5.0 export produces)

Command: `python experiments/report_p1_fmu_me_fmi3.py `. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13.

Why: the reviewer asked that circuits designed in PLECS be the Gym environments of the Python algorithms. PLECS 5.0 exports FMI 2.0/3.0 FMUs; a **Model Exchange** export (PLECS Coder licence) runs standalone, a Co-Simulation export is tool-coupling (PLECS installed and licensed at runtime). Before P1 the adapter drove only FMI 2.0 Co-Simulation. **No PLECS is available here; no PLECS-exported FMU has been run.** The paths are exercised with OpenModelica builds of SPADE's own bucks.

## 1. Averaged buck (duty 0.5, 500 steps of 20 µs) vs the native buck (exact ZOH)

| Build / interface | NRMS v_out vs native | p50 / p99 per step |
|---|---|---|
| co_simulation (2.0, CoSimulation) | 2.9e-06 | 0.003 / 0.005 ms |
| model_exchange_rk4_x1 (2.0, ModelExchange) | 5.1e-07 | 0.044 / 0.086 ms |
| model_exchange_rk4_x10 (2.0, ModelExchange) | 5.0e-11 | 0.394 / 0.552 ms |

## 2. Switching buck (PWM inside the model)

| Build / interface | Steps run (of 500) | Events | p50 per step | v_out at 10 ms | Error |
|---|---|---|---|---|---|
| co_simulation | 24 | 0 | 0.05 ms | None | [solver] doStep at t = 0.00048 s failed: fmi2DoStep failed with status 4 (fatal). |
| model_exchange_rk4_x4 | 500 | 1450 | 1.22 ms | 23.7428 | — |
| model_exchange_rk4_x10 | 500 | 1450 | 1.52 ms | 23.7411 | — |
| model_exchange_rk4_x40 | 500 | 1451 | 3.47 ms | 23.741 | — |

Convergence in the substep count (switching instants located by bisection): max |x4 − x40| = 2.83e-03 V, max |x10 − x40| = 4.03e-03 V. The native **averaged** buck ends at 23.7529 V; the switching model's period-end sample differs by its ripple.

**Reading.** Model Exchange with SPADE's RK4 reproduces the averaged buck to 5e-11, and it **runs the switching buck that aborts in Co-Simulation (S-15)**, converging in the substep count. That is the interface in which a PLECS circuit with real switching would arrive. B3's pre-registered verdicts are not changed by this (they were measured on the Co-Simulation builds).

## 3. FMI 3.0 and 2.0 with real binaries: Modelica Association Reference FMUs

Source: Modelica Association Reference FMUs v0.0.42 · https://github.com/modelica/Reference-FMUs/releases/download/v0.0.42/Reference-FMUs.zip · SHA-256 b6b4a195b9c0f6bdd3459cd949cbb35fc21ee2049d6dab4cb7a3ba7da63b1e97 · Licence: BSD-2-Clause (LICENSE.txt). Run with the user's permission (2026-10-26); OpenModelica 1.27.1 does not export FMI 3.0.

| FMU | Interface | Check | Result |
|---|---|---|---|
| 3.0/Dahlquist | ModelExchange | x vs exp(-t), dt 0.1 | max error 3.1e-11 |
| 3.0/Dahlquist | CoSimulation | x vs the FMU's own explicit Euler 0.9^n (its CS solver) | max error 1.1e-16 |
| 3.0/Stair | ModelExchange | counter at t = 1..8 s (time events) | [2, 3, 4, 5, 6, 7, 8, 9] (expected [2, 3, 4, 5, 6, 7, 8, 9]) |
| 3.0/Stair | CoSimulation | counter at t = 1..8 s (time events) | [2, 3, 4, 5, 6, 7, 8, 9] (expected [2, 3, 4, 5, 6, 7, 8, 9]) |
| 2.0/Dahlquist | ModelExchange | x vs exp(-t), dt 0.1 | max error 3.1e-11 |
| 2.0/Dahlquist | CoSimulation | x vs the FMU's own explicit Euler 0.9^n (its CS solver) | max error 1.1e-16 |
| 2.0/Stair | ModelExchange | counter at t = 1..8 s (time events) | [2, 3, 4, 5, 6, 7, 8, 9] (expected [2, 3, 4, 5, 6, 7, 8, 9]) |
| 2.0/Stair | CoSimulation | counter at t = 1..8 s (time events) | [2, 3, 4, 5, 6, 7, 8, 9] (expected [2, 3, 4, 5, 6, 7, 8, 9]) |
| 3.0/Dahlquist | CoSimulation | dt 0.01 < fixedInternalStepSize 0.1 | [solver] doStep from t = 0 s stopped at 0 s instead of 0.01 s; the FMU declares fixedInternalStepSize = 0.1 s: use a communication step that is a multiple of it |
| 3.0/BouncingBall | ModelExchange | h vs fmpy CVode (rtol 1e-8), 3 s, state events | max error 1.0e-07, 11 events |

The Co-Simulation error of Dahlquist against exp(-t) (1.9e-2) is the FMU's own explicit Euler: SPADE reproduces 0.9^n to 1e-16. A communication step below an FMU's fixedInternalStepSize now fails with that explanation, and Plant Studio refuses such a Ts in Co-Simulation.
