# `dc_perm_ex_v2`, step 1 — pre-registration: convergence of the efficient discontinuous-conduction converter (variant B, variant A as fallback)

**Status:** committed before any code of variant B or A exists and before any measurement of them. The criteria are not changed after results are seen (`CLAUDE.md`). Any deviation is declared in the report.

- **Decision:** Luciano, 2026-11-04 / 2026-11-08: `dc_perm_ex_v2` with variant B, A as fallback, the convergence pre-registration first. Proposal: `s21_dc_perm_ex_v2_proposal.md` §3.
- Choices left to Claude Code are marked **[CC]**.
- **Out of this step:**
  - the `dc_perm_ex_v2` plant itself (proposal §2, step 2). It is built only on a variant that meets §4.
  - no default changes: `dc_perm_ex_v1`, `dc_converter_model=None/"gem"` and `"dcm"` with `dc_plant_substeps` stay byte-identical (S21-4).

## 1. What is compared

| Id | Integration | Status |
|---|---|---|
| **REF** | `dc_converter_model="dcm"`, `dc_plant_substeps=10000` | The converged reference (S21-2, amendment 1: NRMS(×1000 vs ×10000) < 1e-4 in all six cases, `s21_2_dcm_convergence_a1.md`). Recomputed in this step (the trajectories were not stored). |
| D1000 | `"dcm"` ×1000 | The setting the regenerated grids and the promoted PC / RPi5 models use. |
| **B** | New value `dc_converter_model="dcm_exact"` [CC] | Variant B (proposal §3.2), see below. |
| **A** | New value `dc_converter_model="dcm_selective"` [CC] | Variant A (proposal §3.1), only if B does not meet §4. |

**Variant B:**
- Each control period τ uses the exact electrical step with ω frozen at its start value: i(t) = i_ss + (i0 − i_ss)·e^(−t/τe), with i_ss = (u − e)/R_a, e = ψ_e·ω0 and τe = L_a/R_a.
- **Crossing:** if i_ss < 0 < i0, the analytic time t* = τe·ln((i0 − i_ss)/(−i_ss)) is computed. If t* < τ, the current is held at 0 for the rest of the period: blocked, the diode conducts no reverse current.
- **Blocked at the start** (i0 ≤ 0 and u ≤ e): i stays 0 for the whole period.
- **Mechanics:** advance over the period with the period-average electrical torque ψ_e·ī and the same load model as today.
- **Fault torque and vibration:** the same as the existing `"dcm"` path.

**Variant A:**
- Periods with a possible crossing (proposal §3.1 classification) use today's `"dcm"` with N = 1000 substeps.
- The other periods use the exact electrical step of B.

## 2. Cases

The six cases of S21-2 (`ai_n27_preregistration.md` §1.1), unchanged:
1. `dc_pmg132_v1`, 120 rad/s, no extra load, 1.0 s (the S-21 case);
2. stock motor with the 0b-D1 friction (`motor_parameter_overrides` = stock values, `dc_nominal_overrides` = stock ratings), 120 rad/s, 1.0 s;
3. outer_race, electrical severity ×2.0 (16 N·m), 80 rad/s, 2.5 s;
4. cage, ×2.0, 80 rad/s, 2.5 s;
5. control: healthy, 220 rad/s, 1.0 s;
6. control: healthy, 120 rad/s, load 5 N·m, 1.0 s.

The run configurations are taken from `experiments/report_s21_2_dcm_convergence_a1.py`, so the cases are exactly those of amendment 1.

## 3. Metric

- **NRMS**, as in R2 / S21-2: RMS of the difference over the run, divided by the range of the reference over the run. Computed for ω and for i_a, on the samples of every control period.
- **Cost:** CPU seconds per simulated second, measured on the same machine as D1000, one process per case, the machine otherwise idle (S-22 lesson).

## 4. Acceptance criterion (per variant)

A variant **MEETS THE CRITERION** if and only if all of these hold:

1. **Convergence:** NRMS(variant vs REF) < **1e-4** for ω and for i_a, in **each** of the six cases. This is the same threshold as S21-2 amendment 1, against the converged reference (G7).
2. **Control cases** (5, 6, where the current never reaches zero): NRMS(variant vs D1000) ≤ **1e-4** for ω and i_a (proposal §3.3).
3. **No negative current:** min i_a ≥ −1e-6 A in every case (precondition P2 of the AI-N27 regeneration).
4. **Cost [CC]:** the median over the six cases of the CPU per simulated second is at least **10× lower** than D1000's.
   - For B this is the reason to exist: proposal estimate ≈ 40–55×.
   - For A the proposal estimate is 3.4–50×.
   - A variant that is accurate but not 10× faster is reported as **accurate, not efficient enough**, and v2 is not built on it.

- **How it is applied:** point values decide. The verdict is per variant, with the failing checks listed.

## 5. Order and fallback

1. Implement B with unit tests:
   - the analytic crossing time against a closed-form case;
   - i ≥ 0 always;
   - the existing `"gem"` / `"dcm"` paths byte-identical (the existing tests unchanged, plus one new one).
2. Compute REF and D1000 for the six cases; then B. Report against §4.
3. **If B meets §4:** step 1 ends; dataset equivalence (§6) is run on B.
4. **If B does not meet §4:**
   - B's result is reported as it is;
   - A is implemented and evaluated with the same §4 (REF and D1000 reused).
   - B is not retuned: no other margin, no sub-periods, no change of the frozen-ω assumption.
5. **If A does not meet §4 either:** both results are reported. `dc_perm_ex_v2` waits for Luciano's decision. The fallback is "dcm ×1000 for batch only", as today.

## 6. Dataset equivalence (only for a variant that meets §4)

**Runs [CC]:** 10 runs of the 3-class dcm grid (`data/diagnosis_grid_dataset_dcm.parquet`), regenerated with the variant through the generator's own configuration (`build_jobs(3, 2.5, 7, …)`), the same jobs, only the converter value changes:
- 5 from the affected regime: the first acquisition in `build_jobs` order of each 3-class affected cell of `ai_n27_regen_preregistration.md` §4.2;
- 5 healthy, at the grid's speeds 80, 125, 150, 200 and 220 rad/s (the first healthy acquisition at each).

**Criterion:**
- **(a)** NRMS ≤ 1e-3 for `current_r`, `rpm`, `torque_nm` against the stored dcm ×1000 runs, per run. The stored grid is ×1000 and B is compared to it, not to REF. The bound is looser than §4 because ×1000 itself is 5.9e-5 from REF and the noise and vibration channels are added.
- **(b)** top-1 agreement ≥ **0.99** of the promoted PC (`pc_server_5class_w2048/2026-11-04_run01`, Keras) and RPi5 (`rpi5_resnet1d_se_5class_w2048/2026-11-04_run01`, float16 TFLite) on all windows of the 10 runs: stored ×1000 vs variant. The windows are built as in training, with the known classes only.

**If met:** the models trained on dcm ×1000 data are valid for v2. **If not:** reported; v2 can still be built, but those models would need the AI-N27-style warning on v2 data.

## 7. Cost of this step (estimate)

- **REF:** 4 cases × 1.0 s + 2 × 2.5 s at ≈ 1400 s CPU per simulated second ≈ 3.5 h of CPU, ≈ 40 min with 6 processes.
- **D1000:** ≈ 30 min of CPU. B: minutes.
- **Dataset equivalence:** 10 runs × ≈ 10–20 s with B. Classifier inference: minutes.
- **Code:** B ≈ 5–7 h; A, if needed, ≈ 4–6 h.

## 8. Report

`docs/audit/assistant/dc_perm_ex_v2_convergence.{md,json}`, with:
- the commands;
- the per-case NRMS table against REF and D1000;
- min i_a and the cost table;
- the verdict per variant and, if run, the dataset equivalence.

S-21 is updated in `docs/SIMULATION_STABILITY_REPORT.md`.
