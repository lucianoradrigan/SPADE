# SPADE's classical controllers vs. gym-electric-motor 3.0.3's own (V5)

TODO_CLAUDE_CODE.md V5. Date: 2026-10-20.

## How the GEM side was produced

The reference data were recorded once with GEM's own `gem_controllers` (3.0.3, from the unmodified clone) by `tests/data/gen_ref_data_v5.py` into `tests/data/ref_data_v5.npz`. `tests/control/test_gem_controller_comparison.py` replays them without importing GEM.

**Finding about GEM itself.** `gem_controllers` 3.0.3 only works with **NumPy < 2**. With NumPy 2, its DC operation-point stage receives empty arrays and raises `ValueError: The truth value of an empty array is ambiguous`, even on GEM's own default `Cont-SC-PermExDc-v0` environment. The generator therefore runs with `numpy<2`. This affects only the reference environment; driveflow does not use `gem_controllers`.

## The two sets of controllers are not the same tuning

The goal is an independent baseline, not equality.

| | GEM (`GemController.make`, defaults) | SPADE (`PICascadeController`) |
|---|---|---|
| Current loop | PI, magnitude optimum, EMF feedforward | PI, magnitude optimum, back-EMF feedforward |
| Speed loop | PI, symmetric optimum, a = 4 | PI, pole placement (ω_n = 80 rad/s, ζ = 0.9), optional setpoint weight b (C2) |
| Current cap | 80 % of the limit (`current_safety_margin` 0.2) | 80 % of the limit (`current_safety_margin` 0.8) |

**Plant (both sides).** `DcPermanentlyExcitedMotor` defaults, one-quadrant converter, `PolynomialStaticLoad(a=0.01, b=0.05, c=0, j_load=0.0025)`, 65 V supply, Euler at τ = 1e-4 s. This is the plant of `driveflow.validation.build_dc_system`.

## DC speed loop

| Response | Controller | Rise time | Overshoot | Settling (2 %) | Steady-state error | Peak current |
|---|---|---|---|---|---|---|
| Small step 150 → 152 rad/s (S-04) | GEM PI | 5.8 ms | **33.6 %** | 47.8 ms | < 1e-12 | — |
| | SPADE PI, b = 1 | 9.3 ms | 15.1 % | 64.8 ms | < 1e-11 | — |
| | SPADE PI, b = 0.7 | 16.2 ms | 3.1 % | 53.4 ms | < 1e-11 | — |
| Start-up 0 → 150 rad/s (current-limited) | GEM PI | 142.8 ms | 1.8 % | 175.2 ms | < 1e-8 | 164.3 A |
| | SPADE PI, b = 1 | 138.5 ms | 0.2 % | 171.5 ms | < 1e-9 | 168.0 A |
| | SPADE PI, b = 0.7 | 138.7 ms | 0.01 % | 178.5 ms | < 1e-8 | 168.0 A |

**Reading.**
- **Small steps.** The textbook symmetric optimum (a = 4) that GEM uses overshoots small steps about twice as much as SPADE's PI. The ≈15 % that S-04 documents is therefore not unusual for a cascade with a PI speed controller.
- **Large steps.** Both controllers are current-limited and almost identical.

## PMSM MTPA curve

**Comparison.** SPADE's closed-form MTPA (`pmsm_foc.mtpa_id_iq`), inverted for torque, was compared with GEM's own maximum-torque-per-current table (`PMSMOperationPointSelection._get_mtpc_lookup_table`) for the same motor (p = 3, L_d = 0.37 mH, L_q = 1.2 mH, R_s = 18 mΩ, ψ_p = 66 mWb, 400 A). The comparison covers the whole positive torque range of the table, 2.9–716 N·m, at 125 points.

**Result.** The maximum difference is **1.00 A in i_d and 0.91 A in i_q**. GEM searches i_d on a 2.0 A grid, so this is agreement within GEM's own resolution (half a grid step). The earlier check covered one point only (`test_matches_gems_own_torque_limit_at_nominal_current`). `test_closed_form_mtpa_matches_gems_table_over_the_whole_torque_range` now pins the whole curve.
