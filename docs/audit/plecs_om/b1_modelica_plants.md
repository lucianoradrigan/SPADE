# B1 — Native plants in Modelica, exported as FMI 2.0 Co-Simulation FMUs (round PLECS/OM)

**Source:** `modelica/SPADE/package.mo`.
**Build:** `scripts/fmu/build_fmus.sh`, which runs `scripts/fmu/build_fmus.mos` on the pinned OpenModelica image (B0).
**Output:** `data/fmu/<Model>.fmu` and `data/fmu/manifest.json`. Both are git-ignored (D8).

## Models

| Model | Inputs | Outputs | Native counterpart |
|---|---|---|---|
| `BuckAveraged` | `duty` | `v_out`, `i_L` | `sim/buck_system.BuckSystem` (`buck_v1`) |
| `BuckSwitching` | `duty` | `v_out`, `i_L` | the same; MSL `SignalPWM` at 50 kHz plus a `SinglePhase2Level` leg with ideal switches |
| `DcMotorAveraged` | `v_a` | `omega`, `i_a`, `torque` | `datagen/runner.build_plant` (`dc_perm_ex_v1`), without its converter |
| `DcMotorSwitching` | `duty` | `omega`, `i_a`, `torque` | the same, with a `ChopperStepDown` from 65 V and PWM at 10 kHz |

## Parameters and their sources

Every value is the native default. The source is in each parameter's description string in the `.mo`.

| Parameter | Value | Native source |
|---|---|---|
| Vin, L, C, rL, R | 48 V, 220 µH, 220 µF, 0.05 Ω, 6 Ω | `sim/buck_system.py:41-45` (`BuckParams`) |
| Buck PWM frequency | 50 kHz | 1/TAU, `sim/buck_system.py:36` |
| Ra, La, psi_e, Jr | 16 mΩ, 19 µH, 0.165 Wb, 0.025 kg·m² | `sim/motors/dc_permanently_excited_motor.py:50-53` |
| wNominal, IaNominal | 300 rad/s, 97 A | `dc_permanently_excited_motor.py:55` |
| Jload, loadA, loadB, loadC | 0.0025 kg·m², 0.01 N·m, 0.05 N·m·s/rad, 0 | `datagen/runner.py:127` |
| Static-friction linearisation | below a·τ/J, τ = 1 ms | `sim/loads.py:430, 444-447, 470-473` |
| Supply, PWM frequency (DC motor) | 65 V, 10 kHz | `datagen/runner.py:141` and 1/TAU (`:71`) |

**Modelling notes:**
- **The MSL machine is used without losses.** `DC_PermanentMagnet` gets `VaNominal = psi_e·wNominal + Ra·IaNominal = 51.052 V`. With that value its flux constant is exactly psi_e = 0.165. Brush, core, friction and stray losses keep their MSL defaults of zero, and `alpha20a = 0`.
- **`torque` is `airGapDC.tauElectrical`**, the electromagnetic torque ψ_e·i_a. This is the native `torque_nm`.
- **The averaged DC model has no converter.** The one-quadrant rule of the native converter (65 V·duty, or 65 V when i_a < 0; `sim/converters.py:391-393`) is applied by whoever drives the FMU. In B3 that is the driver.
- **`DcMotorSwitching` returns the armature current through the chopper's `dc_n2`.** Grounding `pin_an` as well over-determines the circuit; the first build failed on exactly that equation imbalance.
- **Load parameters exposed.** `loadA/loadB/loadC` are top-level parameters, so that B3 can set the load of the V4 gym-electric-motor reference. Their defaults are the native values.

## Build

Toolchain: OpenModelica 1.27.1, MSL 4.1.0+maint.om, `buildModelFMU(…, version="2.0", fmuType="cs", platforms={"static"})` with `--fmiFlags=s:cvode`. The command, run from the B1 branch:

```
SPADE_DATA_DIR=../SPADE-dev/data scripts/fmu/build_fmus.sh
```

Build of 2026-10-24 (manifest commit `e5f9631`, `.mo` SHA-256 `ec5e4545c2a99398…`). That commit is an earlier state of this commit, before it was amended; the `.mo` SHA-256 identifies the source. The four FMUs take 11.6 s in all (wall time, Docker included).

| FMU | Bytes | SHA-256 |
|---|---|---|
| `BuckAveraged.fmu` | 1,351,818 | `48d52b6ac081cd7a…` |
| `BuckSwitching.fmu` | 1,515,276 | `6883e29d8a8da410…` |
| `DcMotorAveraged.fmu` | 1,452,446 | `1c7111be7ac42f4f…` |
| `DcMotorSwitching.fmu` | 1,608,019 | `363065ba45258646…` |

FMU bytes differ from build to build: each build gets a new GUID. The behaviour of 4 rebuilds was identical (see S-15).

`check_models.mos` checks each model with `checkModel`. Results are in `b1_modelica_plants.json`.

## Tests (`tests/fmu/test_om_fmus.py`, marker `fmu`)

| Test | What it checks |
|---|---|
| `test_fmu_interface` | FMI 2.0 co-simulation, and inputs and outputs with their causality (all 4 models) |
| `test_manifest_hashes_match` | each FMU's SHA-256, and the SHA-256 of the `.mo` it was built from |
| `test_averaged_fmus_simulate_10_ms` | 10 ms at a constant input, finite outputs |
| `test_switching_fmus_simulate_10_ms_at_the_pwm_period` | the same for the switching models, at communication step Ts. Strict xfail S-15 (inventory rows 7–8) |

The tests skip, rather than fail, when fmpy or the FMUs are missing. Their correctness against the native plants is evaluated in B3, not here.

## Finding S-15 (open): switching FMUs and CVODE

Reproduce with `python experiments/report_b1_switching_cvode.py --out docs/audit/plecs_om/b1_switching_cvode.json`.

**Set-up:** each switching FMU is run for 2 ms at a constant duty, with a communication step of Ts/k, where Ts is the PWM period. 2 ms is 100 periods for the buck and 20 for the DC motor.

**FMUs built on 2026-10-24 from this commit** (`b1_switching_cvode.json`):

| Model | Duty | Ts/1 | Ts/2 | Ts/4 | Ts/5 | Ts/10 | Ts/20 |
|---|---|---|---|---|---|---|---|
| BuckSwitching | 0.3 | fails at 0.48 ms | fails at 0.33 ms | fails at 0.175 ms | fails at 0.016 ms | fails at 0.018 ms | fails at 0.019 ms |
| BuckSwitching | 0.5 | fails at 0.48 ms | fails at 0.33 ms | fails at 0.175 ms | fails at 0.016 ms | fails at 0.018 ms | fails at 0.019 ms |
| BuckSwitching | 0.8 | fails at 0.48 ms | fails at 0.33 ms | fails at 0.175 ms | fails at 0.016 ms | fails at 0.018 ms | fails at 0.019 ms |
| DcMotorSwitching | 0.3 | fails at 0.8 ms | runs | runs | fails at 0.48 ms | fails at 0.39 ms | fails at 0.195 ms |
| DcMotorSwitching | 0.5 | fails at 0.8 ms | fails at 0.8 ms | runs | fails at 0.48 ms | fails at 0.39 ms | fails at 0.195 ms |
| DcMotorSwitching | 0.8 | fails at 0.8 ms | fails at 0.8 ms | fails at 0.7 ms | fails at 0.48 ms | fails at 0.39 ms | fails at 0.195 ms |

**FMUs built on 2026-10-22** (`b1_switching_cvode_build_2026-10-22.json`):
- Buck model: the same as today's.
- DC motor model: the same, except that the load coefficients were not yet top-level parameters.
- Same image digest and the same build script logic.
- They were later overwritten; only the SHA-256 of the switching FMUs and the results remain.

| Model | Duty | Ts/1 | Ts/2 | Ts/4 | Ts/5 | Ts/10 | Ts/20 |
|---|---|---|---|---|---|---|---|
| BuckSwitching | 0.3 | runs | fails at 0.24 ms | fails at 0.165 ms | runs | fails at 0.008 ms | fails at 0.009 ms |
| BuckSwitching | 0.5 | runs | fails at 0.05 ms | fails at 0.165 ms | runs | fails at 0.008 ms | fails at 0.009 ms |
| BuckSwitching | 0.8 | runs | fails at 0.05 ms | fails at 0.05 ms | runs | fails at 0.008 ms | fails at 0.009 ms |
| DcMotorSwitching | 0.3 | runs | fails at 0.3 ms | fails at 0.325 ms | runs | fails at 0.24 ms | fails at 0.245 ms |
| DcMotorSwitching | 0.5 | runs | fails at 0.3 ms | fails at 0.325 ms | runs | fails at 0.24 ms | fails at 0.245 ms |
| DcMotorSwitching | 0.8 | runs | fails at 0.3 ms | fails at 0.325 ms | runs | fails at 0.24 ms | fails at 0.245 ms |

**What is established:**
- **The failure:** the error is `fmi2DoStep` → fmi2Fatal. The runtime prints `##CVODE## -27 error occurred at time = …`, and −27 is `CV_TOO_CLOSE`. The FMU's logger callback only receives the unformatted template, so the code comes from the runtime's stdout.
- **Determinism:** within one FMU, runs are deterministic. Two runs gave identical outcomes.
- **Today's builds are consistent:** four builds of 2026-10-24 gave the same pattern, and one of them was built from the 2026-10-22 model source.
  - In today's builds, the buck fails at every step tried, including Ts.
  - The DC motor runs only at Ts/2 or Ts/4, and only for some duties.
- **The 2026-10-22 build behaved differently:** it ran at Ts and Ts/5. That difference is **not explained**.
- **Earlier notes in this round were wrong** in opposite directions:
  - "fails at every communication step tried";
  - "runs at Ts".

  Both were true only of a particular build. The B3 pre-registration states this.

**Not tried:**
- other OpenModelica versions;
- other solvers;
- any change to the models.

For the delivered FMUs, those would be tuning the model to the tool.

**Consequence for B3:** the switching FMUs are evaluated as built. With these builds, they are expected not to simulate at Ts.

## Not executed

- **Windows and macOS FMUs:** not built here. Reason: the builds run in a Linux container. B8's CI is meant to cover Windows (§8 excludes forcing macOS).
