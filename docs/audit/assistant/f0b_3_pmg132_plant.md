# 0b D3 — plant `dc_pmg132_v1` (PMG 132 datasheet at 60 V)

**Simulated evidence only.** User decision of 2026-11-02, applied as a **separate plant** so `dc_perm_ex_v1` (GEM's port, the plant of every trained model, dataset and regression reference) stays unchanged.

- **Command:** `python experiments/report_0b_3_pmg132_plant.py`. Raw values: [`f0b_3_pmg132_plant.json`](f0b_3_pmg132_plant.json). Datasheet reading: [`f0_7_s20_pmg132.md`](f0_7_s20_pmg132.md).
- **Same plant structure as dc_perm_ex_v1:** GEM `DcPermanentlyExcitedMotor`, one-quadrant converter, 65 V supply, PI or MPC, the same bearing-fault and vibration paths. Every DC-only `Scenario` field applies; overrides merge over the datasheet values.
- **Out of scope:** the diagnosis datasets and classifiers, the Gym environments and the GUI do not use it (decision A7: no GUI changes).

| Quantity | dc_pmg132_v1 | Datasheet (60 V) | dc_perm_ex_v1 (GEM) |
|---|---|---|---|
| r_a, l_a, J | 16 mΩ, 0.019 mH, 0.025 kg·m² | 16 mΩ, 0.019 mH, 0.025 kg·m² | same |
| ψ_e | 0.181 V·s/rad | 0.181 (P/(ω·I)) or 0.194 ((U − I·R)/ω) | 0.165 |
| Rated speed / torque / current / voltage | 300.5 rad/s, 20 N·m, 110 A, 60 V | 2870 min⁻¹, 20 N·m, 110 A, 60 V (S1) | 300 rad/s, 16 N·m, 97 A, 60 V |
| Limits current / torque / speed | 200 A, 38 N·m, 400 rad/s | 200 A, 38 N·m (S2 10 min); no speed limit given | 210 A, 38 N·m, 400 rad/s |
| Default load at rated speed | 0.40 N·m (2 % of rated torque, D1) | — | 15.01 N·m (94 % of rated torque) |

**Design check:** PI **ok** (τ_m/τ_a = 11.3; ψ_e is larger than GEM's, so τ_m is shorter); MPC **ok** (horizon/τ_a 1.68).

## Rated speed (2870 min⁻¹), load 0–100 % of rated torque

| Controller | Load (N·m) | Final speed (rad/s) | Steady error (rad/s) | Steady current (A) | Steady voltage (V) |
|---|---|---|---|---|---|
| PI | 0 | 300.54 | -0.000 | 2.2 | 54.4 |
| PI | 5 | 300.55 | 0.000 | 29.8 | 54.9 |
| PI | 10 | 300.55 | 0.000 | 57.5 | 55.3 |
| PI | 15 | 300.55 | 0.000 | 85.1 | 55.8 |
| PI | 20 | 300.55 | 0.000 | 112.7 | 56.2 |
| MPC | 20 | 300.39 | 0.157 | 112.7 | 56.2 |

At rated load the motor holds rated speed with ≈ 113 A (110 A rating plus the 2 % friction) and 56 V, inside the 160 A current cap (0.8 × 200 A) and the 65 V supply. The same load on dc_perm_ex_v1 cannot be held at 300 rad/s (F0.1: 188 A needed).
