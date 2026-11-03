# Proposal (not implemented): `dc_perm_ex_v2` with `dcm` by default, and an efficient `dcm`

**Status:** proposal requested by Luciano on 2026-11-03 (AI-N27 option 2). **Nothing here is implemented.** `dc_perm_ex_v1` (GEM converter) is not touched.

## 1. Cost basis (measured)

| Plant | CPU per simulated second | Source |
|---|---|---|
| Default (`dc_perm_ex_v1`, GEM, 1 Euler step per τ = 0.1 ms) | 3.0 s | healthy, 120 rad/s, 1.0 s (measured 2026-11-03) |
| `dcm` ×1 substep | 3.0 s | same case |
| `dcm` ×1000 (the converged setting, S21-2 amendment 1) | ≈ 195 s; ≈ 19 µs per substep | `s21_2_dcm_convergence_a1.json`, cases 1, 2, 5, 6 (1.0 s each) |
| `dcm` ×10000 | ≈ 1400 s | same |

- **Grid runs** (2.5 s with vibration and faults): ≈ 612 s of CPU each at ×1000 (cases 3–4), against ≈ 7–8 s with the default plant.
- **Consequence:** `dcm` ×1000 is usable for batch datasets (an 840-run grid ≈ 11.5 h with 10 processes) but not as an interactive default. A 2.5 s run in the GUI would take ≈ 8–10 min.

**Fraction of periods that can contain a zero crossing.** Measured on the impact study's DCM1000 noise-free runs (59 runs, mostly the affected regime):
- a period counts if the current at its start or its end is ≤ 0;
- mean 0.28, max 0.67, min 0.0.

Outside the braking regime the fraction is ≈ 0.

## 2. `dc_perm_ex_v2`

- **Definition:** same motor, load, supply, controller presets and fault/vibration models as `dc_perm_ex_v1`. Only the converter integration differs: `dc_converter_model` defaults to `"dcm"`, with the efficient integrator of §3 once its convergence is shown. Until then, `dcm` ×1000 for batch only.
- **Registration:**
  - a `plant_catalog` entry, with `describe_plant` saying "one-quadrant converter with discontinuous conduction (S-21 corrected)";
  - `Scenario` validation: `dc_converter_model` None resolves to `"gem"` for v1 and `"dcm"` for v2;
  - `"gem"` stays allowed on v2 only to reproduce the artifact, never as v2's default.
- **What changes elsewhere:**
  - About 20 modules list `dc_perm_ex_v1` explicitly: tuning, agents, envs, validation benches, run profiles, the assistant draft and objective, and the views. v2 is added where a DC plant is meant.
  - The classifiers trained on v1 data get the AI-N27 warning when applied to v2 data. The candidates of option 2 are the matching models for v2.
- **Defaults:** the GUI and assistant default plant stays v1 unless Luciano decides otherwise (S21-4 holds for v1).
- **v1 guarantee:** the existing byte-identity tests of v1 (`tests/test_s21_dcm_converter.py::test_explicit_gem_is_byte_identical_to_the_default`, and the reproduction precondition P1) stay unchanged.
- **Estimated effort:** ≈ 3–4 h. This covers the catalog, validation, the ~20 lists, tests and docs, without the integrator of §3.

## 3. Efficient `dcm`: subdivide only the periods with a possible zero crossing

### 3.1 Variant A (as requested): selective subdivision

At the start of each period τ, with i0, ω0, the commanded u = d·U_sup and e = ψ·ω0, the period is classified:
- **Possible crossing or blocked:** `u < e + m` and `i0 ≤ i_thr`, or `i0 ≤ 0`. Here m is a margin for the change of e within the period, ψ·|dω/dt|max·τ. The threshold i_thr is what the current can lose in one period with ω frozen: i_thr = (i0 − i_ss)·(1 − e^(−τ/τe)), with i_ss = (u − e)/R and τe = L/R (1.19 ms for the stock motor). The period is integrated with today's code, N substeps (N = 1000).
- **Otherwise (conducting, no crossing possible):** one step of the electrical equation, **exact with ω frozen** (exponential: i(τ) = i_ss + (i0 − i_ss)·e^(−τ/τe)). The mechanics advance with the period-average current.
  - **Not** a single Euler step: S-16 / B3 measured 2.6 % NRMS on i_a for one Euler step per τ.
  - A plain Euler step in the non-crossing periods would therefore not reproduce `dcm` ×1000. That is why ×1000 was needed in every period, not only at the crossings.
- **Expected cost per simulated second:** ≈ 3.6 s + f·192 s, where f is the fraction of crossing periods.
  - affected regime, f ≈ 0.28: ≈ 57 s (≈ 3.4× faster than ×1000);
  - no braking, f ≈ 0: ≈ 3.6 s (≈ 50×);
  - whole grid, f ≈ 0.05–0.1 (to be measured): ≈ 13–23 s (≈ 8–15×).

### 3.2 Variant B (alternative, cheaper): semi-analytic crossing

The same exact electrical step is used in every period. When i_ss < 0 and i0 > 0, the crossing time is analytic: t* = τe·ln((i0 − i_ss)/(−i_ss)). The current runs to t* and is then held at 0 (blocked), while the mechanics advance with zero torque.

- **Cost:** ≈ 1–2 right-hand-side evaluations per period, so ≈ 3.5–5 s per simulated second (≈ 40–55× faster than ×1000; ≈ 1.2–1.6× the default plant).
- **Risk:** the frozen-ω assumption within 0.1 ms, and the coupling error. Both are bounded by the convergence test below.

### 3.3 Validation (to pre-register before implementing)

- **Convergence:** the six cases of S21-2 (`ai_n27_preregistration.md` §1.1), with the same criterion as amendment 1: NRMS < 1e-4 for ω and i_a against `dcm` ×10000.
- **Control tolerance:** on the two artifact-free control cases, variant vs `dcm` ×1000 NRMS ≤ 1e-4.
- **Dataset equivalence:** 10 grid runs regenerated with the variant vs the `dcm` ×1000 grid, with classifier outputs compared on them (agreement ≥ 0.99 with the option-2 candidates). If met, models trained on `dcm` ×1000 data stay valid for v2.
- **Cost:** CPU time measured on the same cases.

### 3.4 Estimated effort

| Item | Effort |
|---|---|
| Variant A, code and unit tests | ≈ 4–6 h |
| Variant B, code and unit tests | ≈ 5–7 h |
| Pre-registered convergence (×10000 references for 6 cases ≈ 1.5 h of compute with 6 processes, plus the variants) | ≈ 3–4 h |
| `dc_perm_ex_v2` (§2) | ≈ 3–4 h |

- **Recommendation:** implement **B** first and **A** as the fallback if B does not meet the convergence criterion. B is ~10× cheaper than A in the braking regime, and both need the exact electrical step anyway.
- **If only A is wanted (as requested):** the total is ≈ 10–14 h.
