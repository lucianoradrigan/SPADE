# `dc_perm_ex_v2`, step 1 — amendment 1 to the convergence pre-registration: variant B with a linear back-EMF

**Status:** decided by Luciano on 2026-11-08 (decision E2 of `PLAN_CONSOLIDADO_2026-11-08.md`). It was committed **before any code of variant B or A existed and before any measurement**. Everything else in `dc_perm_ex_v2_convergence_preregistration.md` stands: cases, reference, metric, the 1e-4 threshold, the other §4 checks, the order and fallback, and the dataset equivalence.

## Why (an a-priori estimate, written before any code)

In the committed variant B, ω (and so e = ψ_e·ω) is frozen at its value at the start of each period. Within one period τ = 0.1 ms the back-EMF changes by ψ_e·ω̇·τ. During a start-up ω̇ is of order 10³ rad/s² (stock motor: ψ_e = 0.165 V·s/rad, L_a = 19 µH), and freezing it gives a current error per period of about:

ψ_e·ω̇·τ² / (2·L_a) ≈ 0.165 · 1000 · 1e-8 / 3.8e-5 ≈ 0.04 A

It accumulates over about τe/τ ≈ 12 periods, giving a bias of order 0.5 A against a current range of about 200 A. That is a NRMS of order 1e-3, above the 1e-4 threshold. This is an estimate, not a measurement.

## What changes

**Variant B (replaces the definition of §1):**

- **Back-EMF:** within each period it varies linearly, e(t) = e0 + ė·t. Here e0 = ψ_e·ω0, and ė = ψ_e·ω̇0, with ω̇0 the mechanical derivative at the start of the period (the system equation at the start state).
- **Conducting:** the electrical equation L_a·di/dt = u − R_a·i − e(t) has the closed-form solution
  i(t) = i_ss0 − (ė/R_a)·(t − τe) + (i0 − i_ss0 − ė·τe/R_a)·e^(−t/τe),
  with i_ss0 = (u − e0)/R_a and τe = L_a/R_a. Its period average is also closed-form.
- **Crossing:** if i(τ) < 0 with i0 > 0, the first zero t* in (0, τ) is located by bisection on the closed form to 1e-12·τ [CC]. After it the current is held at 0, blocked.
- **Blocked at the start** (i0 ≤ 0 and u ≤ e0): i stays 0 while u ≤ e(t). If e(t) falls below u inside the period, at t_on = (u − e0)/ė, conduction resumes from i = 0 with the same closed form.
- **Mechanics:** as in the committed text, the period-average electrical torque and the same load model.

**Variant A:** it uses the same exact step in its non-crossing periods, so it inherits the linear back-EMF.

**The frozen-ω version (B0):** it is **not** a candidate. It is not measured in this step either, so it cannot be selected after the results are seen.

## What does not change

- the threshold: NRMS < 1e-4 against dcm ×10000, ω and i_a, all six cases;
- the control check (≤ 1e-4 against ×1000), min i_a ≥ −1e-6 A and the 10× cost check;
- B first, A only if B does not meet §4. B is not retuned after measuring.
