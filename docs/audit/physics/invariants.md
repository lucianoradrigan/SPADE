# B3: physics invariants (2026-11-16)

**Simulated.** Criteria and tolerances committed before any measurement: [`invariants_criteria.md`](invariants_criteria.md) (`ce24379`). Raw: [`invariants.json`](invariants.json). Tests: `tests/simulation/test_physics_invariants.py` (13 pass, 2 strict xfail S-21; about 30 s).

| Plant | I1 energy residual (≤ 2 %) | I2 voltage / torque error (≤ 1 %) | I3 S-21 case: final speed (≤ 1.02 × 120), min current (≥ −1 A) |
|---|---|---|---|
| `dc_perm_ex_v1` | -0.061 % ✓ | 2.8e-16 / 4.9e-13 ✓ | 120.0 rad/s (1.000×), min i 34.4 A ✓ |
| `dc_perm_ex_v2` | -0.016 % ✓ | 1.4e-16 / 5.1e-13 ✓ | 120.0 rad/s (1.000×), min i 34.4 A ✓ |
| `dc_pmg132_v1` | +61.743 % ✗ | 0.0e+00 / 4.0e-12 ✓ | 179.4 rad/s (1.495×), min i -161.2 A ✗ |
| `dc_pmg132_v2` | -0.039 % ✓ | 1.3e-16 / -1.2e-11 ✓ | 120.0 rad/s (1.000×), min i 0.0 A ✓ |

**I4 (PMSM, `pmsm_v1`):** the plant's torque state equals 1.5·p·(ψ_p·i_q + (L_d − L_q)·i_d·i_q) on 1200 steps (MTPA at 50/150/300 A); worst relative error 3.7e-16 ✓.

## Reading

- **Every v2 plant and `dc_perm_ex_v1` meet the four invariants.** No new finding (next free S-ID stays S-25).
- **`dc_pmg132_v1` violates I1 and I3: that is S-21.** Starting up to 150 rad/s, 62 % of the electrical input is not accounted for by losses, load and stored energy, and the motor ends at 179 rad/s instead of 150. In the S-21 case it ends at 179 rad/s for a 120 rad/s reference with the current at −161 A. Strict xfails citing S-21 (inventory rows 10–11). The same tests pass on `dc_pmg132_v2`.
- `dc_pmg132_v1` passes I2: the steady-state identities hold, but at the wrong speed (179 rad/s), which I2 as written does not check. Declared, not changed (the criteria were fixed before measuring).
- `dc_perm_ex_v1` is not affected because of its stock friction (94 % of rated torque at rated speed), as S-21 already says.
- I2 errors are at round-off level (1e-11 or less): in steady state the logged voltage, current and speed satisfy the model's equations exactly; it checks the bookkeeping of the logged columns, not a discretization error.

Nothing was changed in the plants, defaults or existing tests (D10).
