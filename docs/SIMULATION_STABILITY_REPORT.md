# Simulation stability and controller performance report — SPADE (2026-10-18)

**Scope.** This report covers:
- the DC motor (`dc_perm_ex_v1`) with its PI cascade and linear MPC;
- the PMSM with its dq FOC current control and the MTPA demo;
- the VSC with the DPC neural controller;
- fault injection and synthetic signal generation (bearing-fault torque ripple and vibration Module B).

**How the numbers were produced.**
- Every number below comes from one of two sources:
  - `python experiments/run_stability_battery.py`, whose full output is in [`docs/audit/stability/results.md`](audit/stability/results.md) and `results.json`, with figures in the same folder;
  - the pytest files `tests/simulation/*`, `tests/control/*` and `tests/ai/test_monitoring_alerts.py`.
- Environment: Linux x86_64, Python 3.11.15, NumPy 2.4.6, TensorFlow 2.21.0, CPU only.
- **A PASS means the test was executed** and its assertion held, in the final run recorded in [`docs/audit/validation/final_pytest_output.txt`](audit/validation/final_pytest_output.txt).
- **XFAIL means a known, open defect**: a strict expected failure that turns red once the defect is fixed.

**How tolerances were chosen.** Each one has a stated source, never a universal default:
- the order of the integration method;
- eigenvalues of the linearized plant;
- an analytic steady state;
- the controller's own linear design model;
- the 2 % settling band used by the control criteria.

---

## 1. What was reviewed (code, not README)

| Module | Model / equations (verified in code) | Integration | Units |
|---|---|---|---|
| DC motor `DcPermanentlyExcitedMotor` + `PolynomialStaticLoad` | l_a di/dt = u − r_a i − ψ_e ω; J dω/dt = ψ_e i − (a·sgn ω + b ω + c ω²). The `a` term is smoothed near ω = 0 over τ_decay = 1 ms. | Forward Euler, τ = 1e-4 s, 1 substep | SI (A, V, rad/s, N·m) |
| PI cascade | Inner loop: magnitude optimum, k_p = l_a/2τ, k_i = r_a/2τ, back-EMF feedforward. Outer loop: pole placement at ω_n = 80 rad/s, ζ = 0.9. Conditional anti-windup. Current reference capped at 0.8 × 210 A. | — | — |
| Linear MPC | Forward-Euler model of [i, ω] at τ; 20-step horizon; L-BFGS-B with box bounds on u; soft penalty above 0.85 × i_max. Friction is **not** modelled, so the MPC has no integrator. | — | — |
| PMSM | GEM dq model: p = 3, l_d = 0.37 mH, l_q = 1.2 mH, r_s = 18 mΩ, ψ_p = 66 mWb. | Forward Euler, τ = 1e-4 s | SI |
| dq FOC | PI per axis (magnitude optimum), cross-coupling decoupling, integrator clamp. MTPA locus in closed form; it matches GEM's torque-limit formula at nominal current (existing test). | — | — |
| VSC | Identified discrete-time 2-state L-C model ADF/BDF (exact ZOH of L = 5.02 mH, R_f = 2 Ω, C = 15 µF, Ts = 100 µs; `sim/vsc_params.py`); load i = v_c/R. | Discrete map; there is no step to choose. | V, A |
| DPC | Keras network, receding horizon (first of 5 actions), trained at R = 8.0064 Ω only. | — | — |
| Fault injection | Torque ripple at the bearing order, tracked as an ODE phase state. Module B: modal vibration with impulse trains and background noise. | Euler (phase); Module B uses its own substeps (10 per τ) | N·m; vibration in **synthetic units** |

**Units and parameters.** Consistency was checked by recovering analytic steady states to 1e-4 relative or better:
- the open-loop DC speed, 191.341 rad/s;
- the closed-loop current balancing friction, 45.515 A;
- the voltage-limited speed;
- the PMSM short-circuit current ψ_p/l_d ≈ 178 A.

The vibration amplitude is in Module B's own synthetic unit, not g or m/s². This is already documented in AUDIT_ANSWERS Q04/Q29.

---

## 2. DC motor — numerical stability

| Test | Conditions | Metric | Expected criterion (source) | Actual result | Status |
|---|---|---|---|---|---|
| Bench = production plant | PI, 150 rad/s, 0.05 s | max \|Δi\|, \|Δrpm\| vs `run_scenario` | ≤ 1e-9 (identity) | equal | PASS |
| Time-step convergence | Euler h = τ/1, τ/2, τ/4, τ/10, τ/100 vs RK45 (rtol = atol = 1e-9), closed-loop PI 0.2 s | error ratio when h halves | 1.8–2.2 (first-order method → 2) | 2.019, 2.009 (ω); same for i | PASS |
| Production-step accuracy | h = τ = 1e-4 | max \|Δω\|, max \|Δi\| | < 0.1 % of ω_ref (10× inside the 2 % control band); < 2 % of i_max | 0.0496 rad/s (0.033 %); 3.45 A (1.6 %) | PASS |
| Euler stability region | Open loop, duty 0.5 | ρ(I + τA) at τ = 1e-4 | < 1 | 0.9929 | PASS |
| Numerical vs. physical instability | Open loop at 0.93× and 1.05× the predicted limit τ\* = 2/\|λ\|max = 2.582 ms. The plant itself is Hurwitz (λ = −774.7, −69.2 s⁻¹). | final ω | inside: = analytic 191.341 (1e-3); outside: diverges | 191.3; 2.5e7 → divergent exactly as predicted | PASS |
| Analytic steady state | Euler and RK45, open loop | ω_ss | = analytic (1e-4) | both 191.34 | PASS |
| Steady current balances friction | PI, 150 rad/s | i_ss | (a + bω)/ψ_e = 45.515 A (1e-6) | 45.51515 | PASS |
| Long simulation | 60 s (600 000 steps), PI | NaN/Inf; drift of ω and i between 1 s and 60 s | zero NaN/Inf; drift ≤ 1e-9 | 0 NaN/Inf; ω drift 0, i drift 0 (max \|ω − 150\| = 2.8e-14). Wall time 82 s, peak heap 29 MB. | PASS (20 s variant in pytest, `slow`) |
| Reproducibility | 3 repeated runs, same parameters and seed (PI + fault, MPC, VSC) | SHA-256 of all signals | identical | identical | PASS |
| Seed semantics | Same condition, seeds 1 and 2 | current equal? vibration different? | documents Q15 | current identical, vibration different | PASS (documents AI-N07 cause) |
| Extreme but legal inputs | ω_ref = 1000 rad/s (unreachable at 65 V); ω_ref < 0 (one-quadrant converter) | final ω | voltage-limited point (1e-3); no reversal | settles at the voltage-limited speed | PASS |
| Invalid inputs | NaN/inf references, l_a = 0, r_a < 0, J < 0, negative severity or noise, R ≤ 0, NaN magnitude, duration ≤ 0 | behavior | clear `ValueError` naming the field | `ValueError` (14 cases). **Before the fix: silent empty, NaN or non-physical output, see S-03.** | PASS (fixed) |
| Finite denormalized state with a fault | `build_plant` with 2 faults, 50 steps | `state * limits` finite | all finite | finite. **Before the fix: NaN in the phase slot (S-06).** | PASS (fixed) |

Figure: [`dc_convergence.png`](audit/stability/dc_convergence.png). The error lies exactly on a slope-1 line in log-log.

## 3. PMSM / FOC — numerical stability

| Test | Conditions | Metric | Expected criterion (source) | Actual result | Status |
|---|---|---|---|---|---|
| Physical damping | dq matrix eigenvalues at ω = 0…314 rad/s | Re(λ) | < 0 | < 0 at every speed | PASS |
| Euler stability limit | ρ(I + τA_dq) = 1, bisection | ω_max | must be compared with the nominal speed 314.2 rad/s | **265.8 rad/s, inside the nominal range** | PASS (the test asserts the finding, S-01) |
| Default solver limit (S-01 fix) | `EulerSolver(nsteps=3)`, ρ(I + (τ/3)A_dq) = 1 | ω_max | > 1.4 × nominal | **≈460 rad/s** | PASS |
| Physics at 300 rad/s | RK45, zero voltage, speed held | tail \|i_dq\| | = analytic short-circuit current 178.24 A (0.2 %) | 178.24 / 178.24 | PASS |
| Euler at 300 rad/s | Same run, Euler at τ = 1e-4 | tail \|i_dq\| | divergence expected (ρ = 1.00087) | 1166–5698 A, growing | PASS (asserts the numerical divergence) |
| Euler well inside its region | 100 rad/s, 0.6 s | tail \|i_dq\| | analytic 177.16 A (1 %) | 177.2 | PASS |
| Closed-loop demo regime | MTPA current steps i_s = 100/200/380 A, Euler vs RK45 | max \|Δ\| | < 0.5 % of i_s (asserted for 200 A) | 0.069 / 0.145 / 0.467 A | PASS |
| Reproducibility | MTPA vs. naive cloud, twice | equality, finiteness | identical, finite | identical | PASS |

Figure: [`pmsm_euler_vs_rk45_300rads.png`](audit/stability/pmsm_euler_vs_rk45_300rads.png).

## 4. VSC + DPC — numerical behavior

Since the second pass, every DPC scenario uses the **v4** checkpoint (`configs/dpc_trained_v4_r_range.weights.h5`, S-05). The values below were re-measured with it (`docs/audit/stability/results.md`). v3 values are given where they are the point of comparison.

| Test | Conditions | Metric | Expected criterion (source) | Actual result | Status |
|---|---|---|---|---|---|
| Reproducibility | Same seed, twice | arrays equal | identical | identical | PASS |
| Boundedness over the GUI range | R = 4, 8.0064, 20 Ω; 0.2 s | finite; max \|v_c\| | < 3 × \|v_ref\| = 150 V | 56.0 / 63.3 / 62.0 V (v3: 42.8 / 63.3 / 89.7 V) | PASS |
| Physical instability (correctly simulated) | R = 2, 3 Ω (open-loop ρ = 2.35, 1.25) | divergence | **expected**: physics, not software (existing `test_dpc_robustness_grid.py`) | diverges to inf | PASS (existing) |
| Nominal-load tracking | R = 8.0064 Ω | settled RMSE | v4: 2.52 ± 0.15 V (v3 baseline was 1.18 V) | 2.525 V | PASS |
| Long run | 2 s at R = 8.0064 Ω | RMSE first vs. last quarter | no degradation | 2.547 V → 2.562 V (v3: 1.175 → 1.172 V) | executed (battery) |
| Open-circuit plant | R = 1e9 | open-loop ρ | < 1 (so any divergence is the controller's) | 0.977 | PASS |
| Tracking away from 8 Ω | R = 4, 20, 1000 Ω | settled RMSE | ≤ 10 % of \|v_ref\| = 5 V | **3.10 / 2.89 / 1.90 V** (v3: 17.96 / 14.69 / 111.0 V) | **PASS (S-05 fixed)** |

Figure: [`vsc_rmse_vs_r.png`](audit/stability/vsc_rmse_vs_r.png).

### 4.1 S-05 fix: DPC fine-tuned over a range of loads

`experiments/finetune_dpc_r_range.py` applies the DAgger-style recipe of `finetune_dpc_closed_loop.py` over a grid:
- each round rolls the current controller out in closed loop at R ∈ {3.5, 4, 5, 6.5, 8.0064, 11, 15, 20, 30, 50, 100, 300, 1000} Ω × |v_ref| ∈ {25, 50, 75, 100} V × 2 start phases;
- it adds the visited states to `Data4train.mat`'s rows and continues training on the model-based loss;
- the reference frequency stays at 50 Hz, which `dpc_loss` assumes.

Three rounds from v3 (150 epochs each, LR 5e-4, 2.9 h on one CPU core) gave **v4**. Evidence: [`dpc_r_range_finetune_v4.json`](audit/stability/dpc_r_range_finetune_v4.json).

Settled RMSE at |v_ref| = 50 V, 2000 steps:

| R (Ω) | 3.5 | 4 | 6 | **8.0064** | 9 | 13 | 20 | 25 | 70 | 200 | 1000 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 20.5 | 18.0 | 8.5 | **1.18** | 4.0 | 10.2 | 14.7 | 13.9 | 23.9 | 49.1 | 111.0 | 547.8 |
| v4 | 8.0 | 3.1 | 2.5 | **2.52** | 2.7 | 2.2 | 2.9 | 3.6 | 2.2 | 1.6 | 1.9 | 14.8 |

The other axes of the dashboard's robustness grid, at R = 8.0064 Ω:

| Condition | v3 | v4 |
|---|---|---|
| \|v_ref\| = 10 V | 1.75 V | 4.24 V |
| \|v_ref\| = 150 V | 8.0 V | 8.6 V |
| ω = 50 rad/s | 3.4 V | 3.6 V |
| ω = 700 rad/s | 3.9 V | 4.3 V |

v4 is the default because its worst case over the dashboard's load range (4–20 Ω) is 3.1 V, against 18 V for v3. **Trade-off:** v4 is worse at the single point v3 was trained for, and at small references.

A further round from v4 (LR 2e-4, adding 10 V and 150 V references) diverged at 20–25 Ω and was discarded: [`dpc_r_range_finetune_v5.json`](audit/stability/dpc_r_range_finetune_v5.json). v3 stays in `configs/` for comparison.

### 4.2 Cross-check against DPC4PowerElectronics (TODO V0–V3, 2026-10-20)

**V0. The identified model is a physical L-C filter.** ADF/BDF are the exact zero-order-hold discretization, at Ts = 100 µs, of:

```
di_f/dt = (v_o − v_c − R_f·i_f)/L
dv_c/dt = (i_f − i_load)/C
```

with L = 5.02 mH, R_f = 2 Ω and C = 15 µF. `sim/vsc_params.discretize_zoh` reproduces both matrices to float32 precision, and `tests/simulation/test_vsc_params.py` pins it. The plant is therefore an LC filter, not "LCL" as seven earlier mentions said; those mentions are corrected. Ts = 100 µs is now confirmed, not inferred: the identified C scales with the assumed Ts. Because the matrices are now parametric, the plant can be re-discretized for a degraded capacitor (A4).

**V1. Parameters read from the original Simulink models.**

`experiments/extract_dpc4pe_slx_params.py` reads `CCS_MPC.slx` and `DPCvalidation.slx` as zip archives; MATLAB is not needed. The table and evidence are in [`docs/audit/upstream/dpc4pe_parameters.md`](audit/upstream/dpc4pe_parameters.md).

`CCS_MPC.slx`'s `InitFcn` sets:
- L = 5.02e-3, Rl = 2, C = 15e-6;
- Ts = 1e-4, wref = 2π·50;
- Vref = 50, Vdc = 200.

L, R_f, C, Ts, the reference and the identified ADF/BDF all match SPADE. The ADF/BDF match is to within 4.3e-8.

Four findings come out of the comparison (S-09 to S-12, §6):
- **S-09:** the converter voltage limit is missing from SPADE;
- **S-10:** the data-collection controller is a 5-step MPC, not a one-step one;
- **S-11:** most of the original training rows are at 40 Ω;
- **S-12:** the validation model uses R_f = 0.

**V3. Fidelity of the training port.** `control/dpc/train.py` matches `Main.txt` in architecture, initialization, data split, loss, learning rate and batching. Two minor details of MATLAB's `adamupdate` are not reproduced: the epoch is used as the step counter, and ε = 1e-8. The v1 epoch count is not recorded.

No MATLAB network trained by `Main.txt` on this data could be recovered for comparison:
- **`trainedNet1nDPCv4_1.mat`** (from the upstream git history) is a 7 → 80 → 80 → 2 `dlnetwork`, an architecture different from `Main.txt`.
- **The InitFcn's commented-out 15 → 16 → 16 → 10 network** has a holdout DPC loss of 1325, against 1.15 for SPADE v1, so it is not a converged `Main.txt` result.

Evidence: [`dpc4pe_parameters.md`](audit/upstream/dpc4pe_parameters.md) §V3 and [`matlab_network_comparison.json`](audit/upstream/matlab_network_comparison.json).

**V2. The upstream MPC as a baseline.** `control/mpc/vsc_ccs_mpc.py` ports `CCS_MPC.slx`'s active controller (`MATLAB Function4`): horizon 5, Q = diag(2, 1), closed-form solution, optional ±134 V clamp. Two parts are adapted to SPADE's plant:
- the one-step delay compensation is dropped, because SPADE's averaged plant applies v_o in the same step;
- the extrapolated references are replaced by the exact ones, aligned like the DPC's.

On SPADE's plant, over R = 3.5 Ω – 5 kΩ and |v_ref| = 10–150 V ([`dpc_vs_mpc.md`](audit/stability/dpc_vs_mpc.md), [figure](audit/stability/dpc_vs_mpc.png)):

| Point | MPC N=5 | DPC v3 | DPC v4 (default) |
|---|---|---|---|
| 8.0064 Ω, 50 V | 0.14 V | 1.18 V | 2.52 V |
| 8.0064 Ω, 10 V | 0.03 V | 1.75 V | 4.24 V |
| 4 Ω, 50 V | 0.27 V | 17.96 V | 3.10 V |
| 1 kΩ, 50 V | < 0.01 V | 111 V | 1.90 V |

**What the MPC shows.** It tracks below 1 V everywhere in the grid. With the upstream ±134 V clamp, it can no longer track the points S-09 predicts are unreachable: 150 V at every load, and 120 V at 10 Ω or less. The MPC is the realistic target for C1: the DPC's error is at least an order of magnitude above it at every grid point.

**v4 outside its training range.** At 5 kΩ, above its 1 kΩ training range, v4 settles at 14.7 V at 50 V.

**CPU per step.** The MPC takes ≈ 21 µs per step and the DPC 1.5 ms. The DPC figure is an eager Keras call, so it measures framework overhead and is not an embedded-cost estimate.

### 4.4 C1: DPC accuracy at the nominal point without losing v4's range (2026-10-20)

**Approach.** Option 1 of TODO C1, no retraining. `ScheduledDpcController` uses v3 in a region of (R, |v_ref|) and v4 everywhere else, with 5 % hysteresis on the region's boundaries:
- R = 7.25–8.75 Ω with |v_ref| ≤ 62.5 V;
- |v_ref| ≤ 10 V with R = 5–12 Ω.

**How the region was chosen.** It came from a design grid alone. A first rule was revised after its load-step run, and the revision is declared in [`dpc_operating_point_selection.md`](audit/stability/dpc_operating_point_selection.md).

| | v3 | v4 (default) | Scheduled | C1 target |
|---|---|---|---|---|
| 8.0064 Ω, 50 V | 1.18 V | 2.52 V | **1.18 V** | ≤ 1.3 V |
| 8.0064 Ω, 10 V | 1.75 V | 4.24 V | **1.75 V** | ≤ 2.0 V |
| 4 / 20 / 1000 Ω, 50 V | 18.0 / 14.7 / 111 V | 3.10 / 2.89 / 1.90 V | **3.10 / 2.89 / 1.90 V** | ≤ 3.5 V |

**Results.**
- **Targets:** all met.
- **Divergence:** none at any of the 49 evaluated points.
- **Held-out grid:** the scheduled controller is never worse than v4 (R = 3.5–1000 Ω × 15–60 V), and better at 8 of the 44 points (7.75 and 8.25 Ω at every magnitude).
- **Run-time switching:** in the load-step run 8 → 20 → 8 Ω and the setpoint-step run 50 → 75 → 50 V, it switches twice per run without a larger transient than v4 alone.

**What is not covered.** Off the grid, at 9.5 Ω and 15 V, v3 is still better than v4, and the rule does not capture it. The MPC of §V2 remains about 10× better than either network.

**Status.** Implemented and tested (`tests/control/test_dpc_scheduled.py`). **Default since 2026-10-20** (confirmed by the user): `datagen/runner.py` runs every DPC scenario with it. `test_dpc_robustness_grid.py` was re-measured for it, with the same ≥ 1.4× margin: 1.18 V at the nominal point (v4 alone: 2.52 V) and 1.75 V at 10 V (v4 alone: 4.24 V).

### 4.3 Cross-check against gym-electric-motor 3.0.3 (TODO V4, 2026-10-20)

**How the references were made.** `tests/data/gen_ref_data_v4.py` was run once against the unmodified GEM 3.0.3 clone and recorded four trajectories in `tests/data/ref_data_v4.npz`. `tests/simulation/test_gem_equivalence.py` replays them without importing GEM:

- **Same plant, same solver.** The permanently excited DC motor (two-quadrant converter, 4 voltage steps, 2000 steps) and the PMSM (B6 bridge, dq voltage steps, rotor held at 400 rad/s) were each run with Euler and with solve_ivp (RK45, rtol = atol = 1e-10). SPADE reproduces all four GEM trajectories **bit for bit**.
- **DC motor: SPADE's Euler vs. GEM's RK45.** The maximum error is 0.19 % of peak speed and 2.6 % of peak current; the current error sits on the voltage steps.
- **S-01, checked with GEM's own code.** GEM's single-step Euler diverges at 400 rad/s: |i_s| reaches 3·10⁵ A, against ≤ 450 A for RK45.
- **Induction motors.** `sim/motors/squirrel_cage_induction_motor.py` and the other induction models are used only by `experiments/_archive/`. They were not cross-checked.

**S-13: the S-01 fix gives stability, not accuracy, at high speed.** The PMSM ran open loop on the same dq voltage steps, and Euler's error was measured against RK45 (RMS error of i_s, relative to RMS i_s):

| Speed (rad/s) | Euler ×1 | Euler ×3 (default) | Euler ×10 | Euler ×30 |
|---|---|---|---|---|
| 100 | 7.8 % | 2.4 % | 0.7 % | 0.2 % |
| 200 | 20 % | 5.4 % | 1.5 % | 0.5 % |
| 314 (nominal) | 525 % | **24.6 %** | 5.3 % | 1.6 % |
| 400 | diverges | **48 %** | 7.6 % | 2.2 % |

**Why.** Forward Euler adds numerical growth of about (hω_el)²/2 per substep to the lightly damped dq oscillation. Three substeps keep it below the physical damping r_s/L, so the solution stays stable, but the effective damping falls to roughly half, and the transients after each voltage step are wrong by tens of percent.

**What is not affected.** The PMSM page's closed-loop current steps run near standstill (j_load = 5), and Euler there still matches RK45 within 0.5 %. Open-loop or high-speed use of `build_pmsm_system` is affected.

**Status. Fixed on 2026-10-20** (confirmed by the user): `EULER_NSTEPS = 10`. The error at 314 rad/s is now 5.3 %, pinned below 6 % by `test_default_pmsm_solver_is_close_to_rk45_at_nominal_speed`. The stability limit rises to ≈ 841 rad/s.

## 5. Controller performance (dynamic stability)

**Metrics used.**
- For speed loops: rise time (10–90 %), settling time (±2 %), overshoot, steady-state error, RMSE and max \|e\|.
- For disturbances: maximum dip, recovery into ±0.5 %, and residual offset.
- For the VSC, a tracking problem with a rotating reference, only the settled RMSE is meaningful; step metrics are not applied to it.

| Test | Conditions | Metric | Expected criterion (source) | Actual result | Status |
|---|---|---|---|---|---|
| PI small step vs. design model | 150 → 152 rad/s (unsaturated; peak i 92 A < 168 A) | rise / settling / overshoot | ±10 % / ±10 % / ±1.5 pt of the linear model (inner loop 60× faster; 0.1 ms delay is 1 % of the rise time) | rise 9.3 vs 9.71 ms; settling 64.8 vs 65.4 ms; overshoot **15.1 % vs 14.65 %** | PASS |
| PI steady-state error | same | mean \|e\| over the last 10 % | < 1e-6 (integral action) | 2.6e-12 | PASS |
| PI load disturbance | +10 N·m at 0.5 s, 150 rad/s | speed dip | linear model ±5 % | 1.804 vs 1.777 rad/s (+1.5 %) | PASS |
| PI disturbance rejection | same | final error; Δi | 0; Δi = 10/ψ_e (1e-4) | 2.8e-14 rad/s; Δi = 60.61 A | PASS |
| PI large step (current-limited) | 0 → 150 rad/s | rise time vs. physical bound (accelerating at the 168 A cap) | ≥ bound and ≤ 1.05 × bound | 0.1385 s = bound 0.1385 s | PASS |
| PI large step | same | overshoot; current | < 2 %; ≤ 168 A | 0.18 %; 168.0 A | PASS |
| PI parameter drift | r_a +50 %, l_a ×2, ψ_e −20 %, J_rotor ×3 (controller tuned on the nominal plant) | stable, settles < 0.9 s, overshoot < 5 %, final error < 0.01 | — | worst case J×3: settling 0.479 s, overshoot 0.93 % | PASS |
| Torque mode | τ_ref = 10 N·m | i_ss | 10/ψ_e (1 %) | ok | PASS |
| MPC large step | 0 → 150 rad/s | overshoot, settling, peak current | < 2 %; settles; < 1.1 × 0.85 × 210 A | 0 %; 0.159 s; 178.8 A | PASS (`slow`) |
| MPC steady-state offset | 150 rad/s | SSE | < 0.2 % of ref (10× inside the 2 % band; no integrator, so the offset is expected) | 0.058 rad/s (0.039 %) | PASS (`slow`) |
| MPC under load step | +10 N·m | final offset | < 0.2 % | 0.136 rad/s (0.09 %) | PASS (`slow`) |
| MPC parameter drift | same 4 drifts | settling, SSE | settles, SSE < 0.2 % | all settle; SSE 0.054–0.060 rad/s | executed (battery) |

Figures: [`dc_small_step_vs_linear.png`](audit/stability/dc_small_step_vs_linear.png) and [`dc_load_step.png`](audit/stability/dc_load_step.png).

**PI vs. MPC under identical conditions** (`results.md`):
- MPC is about 7 % faster on large steps and dips 13× less on the load step.
- But it keeps a residual offset (0.06–0.14 rad/s), and it costs about 65–75× more CPU (55–104 s vs. 0.8–1.4 s of wall time per run, under tracemalloc).
- PI has zero offset and is cheap.
- Both are stable over every tested drift.

### 5.1 C3: CPU cost of the MPC (2026-10-20)

**The bottleneck.** The MPC's cost was the per-step L-BFGS-B call: a Python callback loop of up to 50 iterations. The condensed matrices and the warm start were already in place.

**The option.** `MpcController(qp_solver="exact")` solves each step in three stages:
1. the closed-form optimum, when no voltage bound and no current limit is active;
2. otherwise, an exact piecewise-QP. For a fixed set of horizon steps over the current limit, the soft penalty is quadratic, so each pass is a bounded least-squares solve (scipy BVLS), iterated until that set stops changing;
3. L-BFGS-B only as a last resort: 5 of 9000 steps in the hardest run.

**Default since 2026-10-20** (confirmed by the user): `"exact"`. `qp_solver="lbfgsb"` keeps the original solver, bit-identical to the controller before C3. Evidence: [`mpc_exact_solver.json`](audit/stability/mpc_exact_solver.json), from `python experiments/benchmark_mpc_solver.py`, single-threaded BLAS.

| Run | L-BFGS-B | Exact | Speed-up | max \|Δω\| | max \|Δi\| (RMS) |
|---|---|---|---|---|---|
| Start-up 0 → 150 rad/s, 0.6 s | 54.6 s | 1.24 s | **44×** | 0.86 rad/s | 136 A (6.1 A) |
| Steps 100 → 200 → 150 rad/s, 0.9 s | 79.1 s | 9.6 s | **8.2×** | 8.5 rad/s | 277 A (64 A) |
| Load step 10 N·m, 0.8 s | 72.2 s | 1.31 s | **55×** | 0.86 rad/s | 136 A (5.3 A) |
| Torque 10 N·m, 0.3 s | 22.2 s | 0.12 s | **190×** | 0 | 0 |

**MPC vs. PI.** The PI runs the 0.6 s start-up in 0.16 s. With the exact solver the MPC costs about **7.6×** the PI, against about 330× with L-BFGS-B in this measurement.

**Why the trajectories differ.**
- **The exact solver reaches the optimum and L-BFGS-B does not.** With its 50-iteration cap, L-BFGS-B stops short of the optimum: at steady state its current dithers by ±0.7 A, and it exceeds the 178.5 A soft limit (178.8 A). The exact solver holds 178.5 A and a constant steady-state current.
- **Large instantaneous differences sit at one moment.** The 136 A difference is at the instant the current leaves the limit, where the exact solver releases it slightly earlier. The time to reach 90 % is 0.1433 s against 0.1442 s, and the settled tracking error is the same (RMS 0.058 rad/s).
- **The speed-steps run falls in a physically infeasible regime.** Its last segment asks a one-quadrant converter to brake from 200 to 150 rad/s, which it cannot do (S-07). Both solvers fail to track there (RMS error 49.7 rad/s with L-BFGS-B, 43.3 rad/s exact), and their trajectories diverge.

**Target.** The C3 target was at least 3× faster; the exact solver achieves 8–190×.


---

## 6. Findings

| ID | Severity | Type | Finding | Evidence | Action |
|---|---|---|---|---|---|
| **S-01** | High → **fixed** | Numerical instability (method, not physics) | One forward-Euler step per τ = 1e-4 s is unstable on the PMSM's open-loop dq dynamics above **265.8 rad/s**, below the motor's nominal 314 rad/s. The physical system is damped there: RK45 matches the analytic current. | §3; `test_single_step_euler_diverges_at_300_rad_s_although_the_physics_is_stable` | **Fixed (second pass):** `pmsm_foc.default_ode_solver()` = `EulerSolver(nsteps=EULER_NSTEPS=3)` is now the default of `build_pmsm_system` and the open-loop bench. Its stability limit is **≈460 rad/s (1.47× nominal)**. `test_default_solver_is_stable_with_margin_above_the_nominal_speed` and `test_default_solver_reaches_the_analytic_short_circuit_current_at_high_speed[300, 314]` check it; the closed-loop demo still matches RK45 within 0.5 % (`test_closed_loop_current_step_euler_matches_rk45`). The cost is 3 right-hand-side evaluations per control step in the PMSM demo only. **Stable is not accurate:** at 314–400 rad/s open loop, the 3-substep Euler is 25–48 % off RK45 (S-13, §4.3). |
| S-02 | Info | Numerical vs. physical | DC motor: the Euler limit τ\* = 2.58 ms is 26× the production step, and the error at τ = 1e-4 is 0.033 % of speed. | §2 | none needed |
| **S-03** | Medium | Invalid input handling | `Scenario` accepted inputs that produce meaningless runs without any error. Duration ≤ 0 gave an empty list. A NaN reference or l_a = 0 gave all-NaN output. r_a < 0 gave 1.5e17 rpm. R = 0 raised a bare `ZeroDivisionError`. Negative severities and noise were accepted. | probe, 2026-10-18; `TestInvalidInputsAreRejected` (14 cases fail before, pass after) | **Fixed**: `Scenario._validate_numeric_fields`. Legal-but-unstable inputs are still accepted on purpose: R < R\*, unreachable or negative setpoints. |
| **S-04** | Low | Documentation / design | The PI docstring said "ζ = 0.9 → essentially no overshoot". The PI zero gives **≈15 %** overshoot on small steps; large steps hide it through current saturation. | §5, linear model vs. simulation | **Fixed** (docstring). **Option added (C2, 2026-10-20):** `PICascadeController(setpoint_weight=b)` weights the reference in the proportional term only. Default b = 1 is unchanged, operation for operation. On the 2 rad/s step: b = 0.7 gives 3.1 % overshoot (rise 16.2 ms), b = 0.5 gives 0.4 % (rise 24.1 ms), against 15.1 % (9.3 ms) for b = 1. The linear model predicts each value within 0.5 points. The load-step response is unchanged. Exposed as `Scenario.pi_setpoint_weight` and as an advanced option on the DC Motor page. `TestPiSetpointWeighting`. |
| **S-05** | Medium → **fixed** | Controller generalization (learned) | v3 tracked well only at its single training load: RMSE 18 V at 4 Ω, 15 V at 20 Ω, 111 V at 1 kΩ, against 1.18 V at 8 Ω. | §4 and §4.1 | **Fixed (second pass)**: v4 (§4.1) is the default. It tracks within 5 V at every stable load tested from 4 Ω to 1 kΩ. The cost is the nominal point (2.52 V instead of 1.18 V) and \|v_ref\| = 10 V (4.2 V instead of 1.75 V). `test_tracking_within_10pct_of_reference_wherever_the_plant_is_stable` is no longer xfail, and the thresholds of `test_dpc_robustness_grid.py` were re-measured for v4. |
| **S-06** | Low | Invalid value in state vector | The bearing-fault phase state had limit = inf, so `state * limits` gave 0·inf = NaN in that slot and a `RuntimeWarning` on every step of every faulted run. | warnings in the baseline run; `test_denormalized_state_vector_is_finite_with_a_fault_injected` | **Fixed**: unit limit for phase states. Trajectories are unchanged (all existing byte-identical datagen tests pass). |
| S-07 | Info | Physical behavior, not a bug | One-quadrant converter: negative speed references are not tracked (the motor coasts), and references above ≈ 394 rad/s saturate at the 65 V supply. | §2 | none (physics) |
| S-08 | Info | Physical instability, correctly simulated | VSC with R < R\* = 3.3707 Ω diverges with or without a controller. | existing `test_dpc_robustness_grid.py` | none; alarm threshold fixed in AI-N03 |
| **S-09** | Medium | Model scope (missing actuator limit) | SPADE never limits the converter voltage v_o. The upstream converter has V_dc = 200 V, and its active MPC output is clamped at ±134 V per axis (≈ 2·V_dc/3). The steady-state v_o needed for \|v_ref\| = 150 V is 149–232 V for every load, and 151 V at 8 Ω for 120 V. The 120 V and 150 V points of the dashboard range and of `test_dpc_robustness_grid.py` therefore describe an idealized converter with unlimited voltage. | `docs/audit/upstream/dpc4pe_parameters.md` S-09 | **Mitigated (2026-10-20).** `Scenario.vsc_voltage_limit_v` adds an optional per-axis clamp (134 V reproduces the original converter), and the VSC page has a toggle for it. Without the limit, the page warns when a settled run needed more than ±134 V. VSC records now carry the applied `v_o_real` / `v_o_imag` and `v_o_saturated`. The default stays unlimited, so no existing result changes (`tests/test_vsc_voltage_limit.py`). |
| **S-10** | Low | Upstream documentation vs. code | Upstream says the DPC learns from a one-step explicit MPC. The only active controller in `CCS_MPC.slx` is an unconstrained 5-step condensed MPC (Q = diag(2, 1)) with delay compensation and extrapolated references. `Data4train` logs its delay-compensated state and the references 3–7 steps ahead. SPADE's closed loop feeds the measured state and the references 0–4 steps ahead. | `dpc4pe_parameters.md` S-10; `CCS_MPC.slx` `MATLAB Function4` | **Open (documented).** V2 ports this controller. The input convention is not changed. |
| S-11 | Info | Training data | `Data4train.mat` holds exactly two loads: 8.0064 Ω (= 10 Ω ∥ 40 Ω) and 39.9984 Ω, 5000 rows each. `Main.txt`'s training rows 1–6000 are 5000 rows at 40 Ω and 1000 at 8 Ω; the holdout is all 8 Ω. | `dpc4pe_parameters.json` `data4train` | none; relevant to C1 |
| S-12 | Info | Upstream inconsistency | `DPCvalidation.slx`'s filter inductor has R = 0, while `CCS_MPC.slx` and the identified model have 2 Ω. The validation model has no `InitFcn`, so its other parameters are not in the repository. | `dpc4pe_parameters.md` S-12 | none in SPADE (it validates on the identified model) |
| **S-13** | Medium → **fixed** | Numerical accuracy (method) | PMSM default solver (Euler, 3 substeps): stable up to ≈460 rad/s, but in open-loop transients its RMS i_s error against RK45 is 24.6 % at 314 rad/s and 48 % at 400 rad/s. At 100 rad/s it is 2.4 %. | §4.3; `test_gem_equivalence.py` | **Fixed (2026-10-20):** `EULER_NSTEPS = 10`, 5.3 % at 314 rad/s, stable to ≈ 841 rad/s. |
| **S-14** | Medium | Controller robustness (learned) vs. plant degradation | The DPC closed loop diverges when the filter capacitor degrades, and the threshold depends strongly on the load: it diverges for C ≤ ×0.84 at 4 Ω, ≤ ×0.67 at 5 Ω and ≤ ×0.42 at the nominal 8.0064 Ω; at 10, 15 and 20 Ω it stays stable down to ×0.40. Divergence is monotone in C at every load. Found by E10b (C ×0.4 at 8 Ω had no window to identify). | `python experiments/measure_dpc_capacitor_divergence.py` → `docs/audit/stability/dpc_capacitor_divergence.md`; `tests/test_findings_e10.py::test_s14_dpc_loop_vs_capacitor_at_the_nominal_load` | **Documented (2026-10-21).** Relevant to A4: capacitor-degradation scenarios near low loads end in divergence, not in a measurable degraded steady state. No controller or plant default changed. |
| **S-15** | Medium | Numerical (external solver), OpenModelica FMU | The switching FMUs (`BuckSwitching`, `DcMotorSwitching`; OpenModelica 1.27.1, MSL 4.1.0, co-simulation with CVODE via `--fmiFlags=s:cvode`) abort in `fmi2DoStep` with CVODE error −27 (`CV_TOO_CLOSE`), and the steps at which they survive depend on the build. FMUs built on 2026-10-24 (4 builds, including one from the previous model source, all with the same pattern) fail at the PWM period Ts for both models and at every Ts/k tried for the buck; the DC motor runs 2 ms only at Ts/2 or Ts/4 for some duties. FMUs built on 2026-10-22 from the same buck model and the same image digest ran at Ts and Ts/5 and failed at Ts/2, Ts/4, Ts/10 and Ts/20. That difference is not explained: the 2026-10-22 FMUs were overwritten, and only their SHA-256 and results remain. Within one FMU, runs are deterministic. Not tried: other OpenModelica versions; other solvers or model changes for the delivered FMUs. | `python experiments/report_b1_switching_cvode.py` → `docs/audit/plecs_om/b1_switching_cvode.json` (and `…_build_2026-10-22.json`); `tests/fmu/test_om_fmus.py::test_switching_fmus_simulate_10_ms_at_the_pwm_period` (strict xfail) | **Open (2026-10-24).** B3 evaluates them as built. The models are not tuned to avoid the failure. **Workaround found (P1, 2026-10-26):** the same switching buck built as FMI 2.0 **Model Exchange** and integrated by SPADE (RK4 with event location) runs 10 ms at Ts and converges in the substep count (`docs/audit/plecs_om/p1_fmu_me_fmi3.md`). B3's verdicts are unchanged (measured on the Co-Simulation builds). |
| S-16 | Info | Numerical accuracy (method), native DC motor | Open loop, the native `dc_perm_ex_v1` plant (one forward-Euler step per τ = 100 µs, armature time constant 1.19 ms) differs from the same plant integrated with 100 Euler substeps by 2.6 % NRMS of the i_a range (omega 0.11 %), over the B3 sequence of duty steps and PRBS with converter-branch changes. The OpenModelica FMU of the same equations matches the fine integration to 2e-5. Consistent with §2 "Production-step accuracy" (1.6 % of i_max, closed loop). | `docs/audit/plecs_om/b3_om_fmu_spike.md` (criterion 4 diagnostic; [post-hoc analysis](audit/plecs_om/b3_om_fmu_spike.md#post-hoc-analysis-not-pre-registered--added-2026-10-25), not pre-registered); `python experiments/report_b3_om_fmu_spike.py` | **Open (2026-10-25).** It makes DcMotorAveraged formally NO-GO in B3 (criterion 4 ≤ 1 %); the verdict stands. Impact on diagnosis (R2, pre-registered, NAT vs a converged RK45 reference): **affects diagnosis** under the registered rule, see S-18. Native plant not changed (§9). |
| S-17 | Low | Numerical method (time argument), `sim/solvers.EulerSolver` | With `nsteps > 1`, `_integrate_nsteps` starts its substep clock at the **end** time t of the step (`current_t = t`) and evaluates f at t + k·h/n, k = 1…n: one control period later than the interval [t − τ, t] it integrates over. Inherited from gym-electric-motor. It matters only when the right-hand side depends on t: the substep solution converges, but to the system shifted by one period. Measured in R2.1 on the DC plant with the validation benches' time-triggered `LoadTorqueStep` (4 N·m at 1.0 s): Euler × 10/100/1000 stop at 2.4e-4 NRMS of RK45 (no convergence), while Euler × 1000 with the step at 1.0 s matches RK45 with the step one period earlier to 2.2e-6. Production paths with `nsteps > 1` (PMSM, `EULER_NSTEPS = 10`, `PolynomialStaticLoad`) and the diagnosis plant (`BearingFaultLoad`: the fault phase is a state, not t) have time-independent right-hand sides, so they are not affected. | `python experiments/report_s16_discretization_sensitivity.py convergence` → `docs/audit/plecs_om/r2_convergence.{md,json}`; `… load_step_offset` → `r2_load_step_offset.json`; `tests/test_r2_dc_plant_integration.py::test_s17_euler_substeps_evaluate_inside_the_step` (strict xfail) | **Open (2026-10-25).** Not fixed: changing the solver needs the user's confirmation (§9 of the round). It stopped R2 at R2.1 (no Euler reference reached 1e-4 in the load-step scenario). |
| S-18 | Medium | Numerical accuracy (method) → diagnosis, native DC motor | S-16's single Euler step changes what the diagnosis pipeline sees. Pre-registered R2 (`docs/audit/plecs_om/r2_preregistration.md`): 60 runs per version (3 speeds × 2 severities × 5 classes × 2 seeds, 2.5 s, generator's noise rule), NAT = production plant, REF = RK45 1e-9 (converged within 6.4e-6 of 1e-11). Fails the rule on (1) the current_r fault-band amplitudes: worst 16.5 %, all bands above 5 % at the 2nd and 3rd harmonics of noisy runs (noise-free worst 2.4 %, fundamentals worst 1.3 %); (2) start-up top-1 agreement: PC 0.917, RPi5 0.925, ESP32 0.969, with PC and RPi5 2.5 points more accurate on REF; (3) ESP32 steady state 0.977, all on ball/cage windows (known-label 1.000). PC and RPi5 in steady state pass their own conditions (0.987 / 1.000, \|Δ acc\| ≤ 0.006). The largest signal differences (current NRMS up to 0.46) are cage/ball runs at 100 rad/s where i_a keeps crossing zero and the converter switches branch. | `python experiments/report_s16_discretization_sensitivity.py sensitivity --workers 10` → `docs/audit/plecs_om/r2_s16_sensitivity.{md,json}`, `r2_figures/`; `tests/test_r2_sensitivity_regression.py` (slow) | **Open (2026-10-25).** User decision (2026-10-25): regenerate the datasets with the reference. **Done (R3):** `data/diagnosis_grid_dataset_5class_rk45.parquet`, `data/diagnosis_grid_dataset_rk45.parquet` (`docs/audit/plecs_om/r3_datasets_rk45.md`); originals kept. **Retrained (R4, candidates):** PC/RPi5/ESP32 on the RK45 datasets, same recipe; no improvement on either plant (start-up slightly worse for PC; one training seed each), see `docs/audit/plecs_om/r4_retrained_rk45.md`. Nothing promoted, no default changed. **User decision (2026-10-26): keep the promoted models; S-18 documented, plant default unchanged.** |
| S-19 | Info | Physical operating range of the training grid (one-quadrant converter + fault ripple) | In the classifier training grids, 78 of 525 runs (5-class) and 21 of 315 (3-class) do not hold the speed reference within 5 % in steady state: high-severity fault runs at 80–125 rad/s, where the one-quadrant converter cannot brake against the fault ripple and the mean speed rises up to +72 %. Same with Euler and with RK45 (not a discretization effect). Their `grid_speed` column is the reference, not the actual speed. | `docs/audit/plecs_om/r3_speed_tracking.json`, `r3_datasets_rk45.md` | **Documented (2026-10-25).** No change; relevant to any per-speed analysis of the grid datasets. |

## 7. Limitations of this report

- The PMSM model is checked only open-loop and in the standstill current-step demo. There is no closed speed loop for the PMSM in the code base, so none was tested.
- The vibration (Module B) signals are checked for reproducibility and finiteness here. Their physical calibration is covered by the earlier reports (`docs/audit/vibration_generalization_report.txt`): only one axis is calibrated, and it does not generalize across bearings.
- Wall time and memory were measured under `tracemalloc` on one 12-core Linux machine. Use them for relative comparison only.
- Not executed on Windows or macOS (see `docs/VALIDATION_SUMMARY.md`).
