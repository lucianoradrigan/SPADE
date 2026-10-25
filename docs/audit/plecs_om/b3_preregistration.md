# B3 — OpenModelica FMU feasibility (GO / NO-GO): pre-registration

**Status of this file.** It is committed **before** any B3 measurement. The criteria and targets are
the user's (round PLECS/OM, §5 B3) and are not changed after results are seen. Deviations, if any,
go into the results report, marked as such.

## Observations made BEFORE this registration (declared, not B3 results)

While building and testing B1/B2 (2026-10-22 to 2026-10-24), three things were seen:
1. **BuckAveraged:** under a constant duty 0.5 for 10 ms through `FmuAdapter`, it matches the native buck within 0.43 mV, two runs are bit-identical, and doStep takes about 2.4 µs (p50).
2. **BuckSwitching and DcMotorSwitching** are built with CVODE, as required. They abort in `fmi2DoStep` with CVODE error −27 (`CV_TOO_CLOSE`), and which communication steps they survive depends on the build. This is finding S-15; the evidence is in `b1_modelica_plants.md`.
   - **The FMUs B3 uses (built 2026-10-24):** the buck fails at Ts and at every Ts/k tried. The DC motor fails at Ts and runs 2 ms only at Ts/2 or Ts/4, for some duties.
   - **A build of 2026-10-22** ran both models at Ts and Ts/5.
   - **Two earlier notes in this round were each true of one build only:** "fails at every step tried" and "runs at Ts".
   - Whether their outputs are right was **not** looked at: that is criterion 5.
3. **Diagnostic only, on modified models or flags that are not used here:**
   - A half-period PWM offset made them run at the sub-steps, but the buck's cycle average became ≈ 2·d·V_in.
   - A build with OpenModelica's default solver gave correct averages only at communication steps ≪ Ts.

This finding is registered as **S-15** in B1. B3 evaluates the FMUs exactly as `scripts/fmu/build_fmus.sh` builds them (CVODE, `--fmiFlags=s:cvode`); no model or flag is changed for B3.

## Set-up

- **FMUs:** `data/fmu/{BuckAveraged,BuckSwitching,DcMotorAveraged,DcMotorSwitching}.fmu` as built in B1 (`data/fmu/manifest.json`, `.mo` SHA-256 `ec5e4545…`). Their SHA-256 are in `data/fmu/manifest.json`, copied into the results.
- **Driver:** `driveflow.integrations.fmu.adapter.FmuAdapter`. The communication step is the native control step: Buck 20 µs (`sim/buck_system.TAU`), DC motor 100 µs (`datagen/runner.TAU`).
- **Native references:**
  - Buck: `sim/buck_system.BuckSystem` (exact ZOH of the averaged model).
  - DC motor: `datagen.runner.build_plant` (GEM-equivalent DcMotorSystem, Euler at 100 µs), driven open loop with the same duty sequence.
  - The native converter is one-quadrant (`sim/converters.py:391-393`): it applies 65 V · duty while i_a ≥ 0, and the supply voltage when i_a < 0 (freewheeling). The averaged FMU has no converter, so the B3 driver applies **the same rule**, using the FMU's own i_a at the start of each step: v_a = 65 · duty, or 65 when i_a < 0.
  - The number of steps where that branch is taken is reported for both plants.
  - DcMotorSwitching contains its own chopper (`ChopperStepDown`), so it receives the duty directly.
- **Action sequences (fixed):**
  - **Buck:** duty levels 0, 0.3, 0.6 and 0.45 for 250 steps each (5 ms), then PRBS7 (`integrations/plecs/excitation.py`, seed 5, D_min 0.1, D_max 0.8, T_hold 1 ms) for 1000 steps (20 ms). 2000 steps in all.
  - **DC motor:** from rest, duty levels 0.4, 0.7 and 0.55 for 1000 steps each (0.1 s), then PRBS7 (seed 9, D_min 0.3, D_max 0.8, T_hold 20 ms) for 3000 steps (0.3 s). 6000 steps in all.
  - A second, different sequence is used only for criterion 1: the same levels with the PRBS seed 77.
- **Switching models:**
  - Criteria 1, 2, 3, 6 and 7 are evaluated at the communication step Ts, like the averaged models.
  - A sample taken at a period boundary is not a cycle average. So, for criterion 5, each period is also run as 4 communication steps of Ts/4, the sub-step that ran most often in B1. The 5 samples of each period (both boundaries included) are averaged with the trapezoid rule and compared with the native value at the end of the period. The ripple is reported as max − min of those 5 samples, a lower bound.
  - **Expected:** with the FMUs B3 uses, the switching models are expected not to simulate at Ts. The rules below then give NO-GO. This expectation is written down before the run.

## Criteria (per model)

| # | Criterion | GO target |
|---|---|---|
| 1 | The action affects the output | Two different sequences give different outputs; the same sequence gives identical outputs |
| 2 | Simulated time monotone and step counter consistent | No exception; t after n steps = n·Ts within 1e-12 relative |
| 3 | Reproducible reset | Two episodes with the same seed and actions are bit-identical (else ≤ 1e-9 relative, declared) |
| 4 | Averaged models vs native, same sequence | RMS(FMU − native) ≤ 1 % of the native signal's range (max − min), on v_out and i_L (Buck), omega and i_a (DC motor) |
| 5 | Switching models vs native, values averaged per switching cycle | ≤ 3 % RMS of the native range; ripple also reported |
| 6 | Latency per doStep | Averaged: p50 ≤ 0.5 ms. Switching: reported; "GO for evaluation only" if p50 ≤ 20 ms |
| 7 | Long-episode stability | 10 episodes of 1 s simulated per model (PRBS7 seeds 11…20, the DC levels above, the Buck levels above), no solver failure and no NaN |

**Also reported, no target:**
- The DC-motor averaged FMU vs the gym-electric-motor 3.0.3 reference trajectory of V4 (`tests/data/ref_data_v4.npz`, `dc_solve_ivp`, RK45 at 1e-10), with the same RMS / range metric on omega and i_a.
  - Set-up as in `tests/data/gen_ref_data_v4.py`: 2000 steps of 100 µs and actions 0.3 / 0.8 / −0.4 / 0.6.
  - V4 uses a two-quadrant converter at 420 V, so the FMU receives v_a = 420 · action.
  - The FMU's load parameters are set to V4's: loadA 0.05, loadB 0.02, loadC 0.001, Jload 0.01.
- Wall time per episode.

**Verdict per model:**
- **GO:** every applicable criterion passes.
- **GO for evaluation only:** a switching model that passes everything except the training latency.
- **NO-GO:** fails 1, 2, 3, 4 or 7. For a switching model, also when criterion 5 cannot be evaluated because the FMU does not simulate.

**If criterion 4 fails:**
1. The model is not adjusted by looking at the difference.
2. The parameters are re-checked against the native code, and what was found is documented.
3. If the difference persists, it is reported to the user as a possible finding of the native plant.

**After B3:** work stops and the verdicts are sent to the user. B4 waits for the user's OK.

**Artifacts:**
- script `experiments/report_b3_om_fmu_spike.py`;
- results `docs/audit/plecs_om/b3_om_fmu_spike.{md,json}`;
- figures native vs FMU under `docs/audit/plecs_om/b3_figures/`.
