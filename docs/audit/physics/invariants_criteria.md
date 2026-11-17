# B3 physics invariants: criteria (written 2026-11-16, before any measurement)

All runs are simulated. Plants: `dc_perm_ex_v1`, `dc_perm_ex_v2`, `dc_pmg132_v1`, `dc_pmg132_v2` (DC, PI speed control) and `pmsm_v1` (FOC).

| # | Invariant | Case | Tolerance |
|---|---|---|---|
| I1 | Energy balance: ∫u·i dt = ∫R·i² dt + ∫T_L·ω dt + ΔE_kin + ΔE_mag | start-up 0 → 150 rad/s, 0.5 s, healthy, stock load | relative residual ≤ 2 % of ∫u·i dt |
| I2 | DC steady state: u = R·i + ψ·ω and ψ·i = T_L(ω) | last 20 % of a 1.0 s run at 150 rad/s, healthy | each ≤ 1 % (means over the window) |
| I3 | No energy created: a speed-controlled motor without extra load does not end above its reference | 120 rad/s, 0.7 s, no extra load (the S-21 case) | final speed ≤ 1.02 × ω_ref, and armature current ≥ −1 A |
| I4 | PMSM torque: T = 1.5·p·(ψ_p·i_q + (L_d − L_q)·i_d·i_q) | stock PMSM run | relative error ≤ 1e-6 (an identity of the model) |

Rules (instructions 2026-11-16, B3, D10): a violation in a v1 plant becomes a strict xfail citing S-21; a violation in a v2 plant (or elsewhere) becomes a new finding S-25 onward. No plant, default or existing test is changed. Tolerances are not changed after seeing results.
