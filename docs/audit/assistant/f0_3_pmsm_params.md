# F0.3 — PMSM parameters (`Scenario.pmsm_motor_parameters`)

**Simulated evidence only.**

- **Command:** `python experiments/report_f0_3_pmsm_params.py`. Raw values: [`f0_3_pmsm_params.json`](f0_3_pmsm_params.json).
- **Field:** `pmsm_motor_parameters` (pmsm_v1 only), partial override merged over `pmsm_foc.MOTOR_PARAMS`; keys `p`, `l_d`, `l_q`, `r_s`, `psi_p`, and also `j_rotor` (GEM's `PermanentMagnetSynchronousMotor` accepts it, default 0.03883 kg·m²) and `u_supply` (`build_pmsm_system` accepts it, default 330 V).

## 1. Propagation (read in `control/classical/pmsm_foc.py`)

| Function | Receives the merged dict | Uses the global `MOTOR_PARAMS` inside? |
|---|---|---|
| `build_pmsm_system(mp, u_supply)` | yes (`motor_parameter=mp`) | no, only as the default argument |
| `DqCurrentController(system, mp)` | yes (gains and decoupling terms) | no |
| `mtpa_id_iq(i_s, mp)` | yes | no |
| `torque_of(i_d, i_q, mp)` (the record's `torque_nm`) | yes; it used to receive `MOTOR_PARAMS` explicitly in `runner._run_pmsm_scenario` | no |

Other globals (`U_SUPPLY`, `I_LIM`, `LOAD_INERTIA`, `TAU`) are defaults or constants of the bench; `u_supply` is overridable, the |i_s| ≤ 400 A bound of `pmsm_current_ref_a` stays.

**Change outside `Scenario`:** `mtpa_id_iq` divided by `l_d − l_q`, so a non-salient machine raised a division by zero. It now returns i_d = 0 for `l_d == l_q`, the formula's limit. The stock machine (l_d ≠ l_q) takes the original path.

## 2. Default vs explicit `MOTOR_PARAMS`

Bit-identical records (i_d, i_q, torque, rpm, u_d, u_q): FOC_MTPA yes, FOC_naive yes.

## 3–4. MTPA vs naive torque (mean of the last 5 ms of a 30 ms current step)

| Machine | i_s (A) | MTPA torque (N·m) | Naive torque (N·m) | MTPA / naive |
|---|---|---|---|---|
| non-salient (l_d = l_q = 0.8 mH) | 100 | 29.74 | 29.74 | 1.0000 |
| non-salient (l_d = l_q = 0.8 mH) | 200 | 59.66 | 59.66 | 1.0000 |
| non-salient (l_d = l_q = 0.8 mH) | 350 | 104.89 | 104.89 | 1.0000 |
| stock (l_d 0.37 mH, l_q 1.2 mH) | 100 | 42.05 | 29.77 | 1.4127 |
| stock (l_d 0.37 mH, l_q 1.2 mH) | 200 | 119.91 | 59.76 | 2.0068 |
| stock (l_d 0.37 mH, l_q 1.2 mH) | 350 | 308.40 | 105.18 | 2.9322 |

- Non-salient: MTPA and the naive policy give the same torque (ratio 1 to rounding; the test tolerance is 0.5 %).
- Stock salient machine: MTPA gains torque from the reluctance term, more at higher current.
- `l_d > l_q` is accepted with a `UserWarning` (MTPA assumes l_q ≥ l_d); see `tests/test_assistant_f0_pmsm_params.py`.

## Not executed

- A PMSM from a real datasheet (e.g. EMRAX 228): no ejecutado. Its flux and Ke conventions are unresolved (docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md §3); that is Phase 2 work.
- Field weakening or a speed loop: no ejecutado; the PMSM plant is a current-step bench near standstill.
