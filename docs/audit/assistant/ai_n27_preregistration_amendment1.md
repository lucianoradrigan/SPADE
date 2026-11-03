# AI-N27 / S21-2 — pre-registration, amendment 1 (option A, Luciano, 2026-11-02)

**Status.** Committed **before** the new convergence level and before any classifier metric are measured. It amends [`ai_n27_preregistration.md`](ai_n27_preregistration.md); everything not named here stays as written there.

## 1. What happened under the original criterion (kept as recorded)

[`s21_2_dcm_convergence.md`](s21_2_dcm_convergence.md): the criterion NRMS(×100 vs ×1000) < 1e-4 for ω and i_a **was not met** in 2 of the 6 cases:

| Case | NRMS ×100 vs ×1000, ω | NRMS ×100 vs ×1000, i_a |
|---|---|---|
| 3_outer_race_x2_80 | 4.0e-05 | **5.0e-04** |
| 4_cage_x2_80 | 1.9e-05 | **2.0e-04** |

It was met in the other four (worst i_a 3.3e-05). The control passed (≤ 1.5e-13). As pre-registered, the impact study was not run. **This result stands; it is not re-read or overwritten.**

## 2. Amendment

1. **Convergence:** the same six cases also with **×10000** substeps. The criterion is now measured between the two finest levels: **NRMS(×1000 vs ×10000) < 1e-4 for ω and i_a in every case.** **The threshold 1e-4 does not change.** NRMS as before (RMS of the difference / range of the finer solution over the run).
2. **If it is not met at ×10000 either:** stop and report. The impact study is not run and no change of threshold is proposed.
3. **Impact versions:** **GEM1000** (`dc_converter_model="gem"`, `dc_plant_substeps=1000`) and **DCM1000** (`"dcm"`, 1000 substeps) replace GEM100 and DCM100. Set, windows, models, metrics and decision rule unchanged (§2 of the original).
4. **Two comparisons presented with the same weight** (Luciano, 2026-11-02):
   - **(a) GEM1000 vs DCM1000** — pre-registered; **decides the verdict** (isolates the converter model);
   - **(b) STORED (the training data, one Euler step) vs DCM1000** — the practical question: do the models change their mind between the data they were trained on and the corrected physics? It includes S-16 / S-18 and does not decide.
   Both by model, by label class and by regime (start-up t < 0.5 s, steady state t ≥ 0.5 s), and over all windows.
5. **Execution:** 10 worker processes. Regenerated runs stay in a scratch directory (not committed).
