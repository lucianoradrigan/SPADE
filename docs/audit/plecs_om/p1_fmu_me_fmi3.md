# P1 — FMI 3.0 and Model Exchange in FmuAdapter (the interfaces a PLECS 5.0 export produces)

Command: `python experiments/report_p1_fmu_me_fmi3.py `. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13.

Why: the reviewer asked that circuits designed in PLECS be the Gym environments of the Python algorithms. PLECS 5.0 exports FMI 2.0/3.0 FMUs; a **Model Exchange** export (PLECS Coder licence) runs standalone, a Co-Simulation export is tool-coupling (PLECS installed and licensed at runtime). Before P1 the adapter drove only FMI 2.0 Co-Simulation. **No PLECS is available here; no PLECS-exported FMU has been run.** The paths are exercised with OpenModelica builds of SPADE's own bucks.

## 1. Averaged buck (duty 0.5, 500 steps of 20 µs) vs the native buck (exact ZOH)

| Build / interface | NRMS v_out vs native | p50 / p99 per step |
|---|---|---|
| co_simulation (2.0, CoSimulation) | 2.9e-06 | 0.003 / 0.004 ms |
| model_exchange_rk4_x1 (2.0, ModelExchange) | 5.1e-07 | 0.044 / 0.064 ms |
| model_exchange_rk4_x10 (2.0, ModelExchange) | 5.0e-11 | 0.385 / 0.487 ms |

## 2. Switching buck (PWM inside the model)

| Build / interface | Steps run (of 500) | Events | p50 per step | v_out at 10 ms | Error |
|---|---|---|---|---|---|
| co_simulation | 24 | 0 | 0.05 ms | None | [solver] doStep at t = 0.00048 s failed: fmi2DoStep failed with status 4 (fatal). |
| model_exchange_rk4_x4 | 500 | 1450 | 1.21 ms | 23.7428 | — |
| model_exchange_rk4_x10 | 500 | 1450 | 1.52 ms | 23.7411 | — |
| model_exchange_rk4_x40 | 500 | 1451 | 3.43 ms | 23.741 | — |

Convergence in the substep count (switching instants located by bisection): max |x4 − x40| = 2.83e-03 V, max |x10 − x40| = 4.03e-03 V. The native **averaged** buck ends at 23.7529 V; the switching model's period-end sample differs by its ripple.

**Reading.** Model Exchange with SPADE's RK4 reproduces the averaged buck to 5e-11, and it **runs the switching buck that aborts in Co-Simulation (S-15)**, converging in the substep count. That is the interface in which a PLECS circuit with real switching would arrive. B3's pre-registered verdicts are not changed by this (they were measured on the Co-Simulation builds).

## 3. FMI 3.0

code path unit-tested with test doubles (tests/fmu/test_fmu_me_fmi3.py); no real FMI 3.0 binary executed. Not executed with a real binary: OpenModelica 1.27.1 does not export FMI 3.0; executing the Modelica Association Reference FMUs (BSD-2, v0.0.42) was not permitted in this session.
