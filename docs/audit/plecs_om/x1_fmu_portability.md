# X1 — Do SPADE's OpenModelica FMUs work on Windows, macOS and Linux?

CI run: https://github.com/lucianoradrigan/spade/actions/runs/40442231526 (workflow `.github/workflows/fmu-portability.yml`, all jobs green). Per OS: `python scripts/fmu/compile_fmu_from_source.py data/fmu/variants/<M>_fmi2_me.fmu out/<M>_me.fmu --drop-existing-binaries [--static-runtime --generator "MinGW Makefiles" on Windows]`, then `python experiments/report_x1_fmu_portability.py --buck-me out/BuckAveraged_me.fmu --switching-me out/BuckSwitching_me.fmu --out out/x1_<OS>.json`. Local row: `python experiments/report_x1_fmu_portability.py --buck-me data/fmu/variants/BuckAveraged_fmi2_me.fmu --out <scratch>` (pinned Docker build, no switching run).

The pinned OpenModelica build (Docker) produces Linux binaries only. OpenModelica FMUs ship their C sources and a CMakeLists.txt, so the Model Exchange variants were compiled on each OS with that OS's compiler and checked there.

## 1. B3 pre-registered criteria, BuckAveraged Model Exchange (RK4 ×10 substeps) vs the native buck (exact ZOH)

| OS | CPU | Crit 1/2/3/4/7 | NRMS v_out / i_L | Crit 6: p50 per step (≤ 0.5 ms) | B3 verdict | Info: RK4 ×1 p50 / NRMS v_out |
|---|---|---|---|---|---|---|
| Linux | AMD EPYC 7763 64-Core Processor | ✓/✓/✓/✓/✓ | 3.2e-11 / 5.6e-11 | 0.840 ms | **NO-GO** (fails 6) | 0.096 ms / 3.2e-07 |
| Windows | AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD | ✓/✓/✓/✓/✓ | 3.2e-11 / 5.6e-11 | 0.719 ms | **NO-GO** (fails 6) | 0.082 ms / 3.2e-07 |
| macOS | Apple M1 (Virtual) | ✓/✓/✓/✓/✓ | 3.2e-11 / 5.6e-11 | 0.381 ms | **GO** | 0.041 ms / 3.2e-07 |
| Linux (local, Ryzen 5 5600X) | AMD Ryzen 5 5600X 6-Core Processor | ✓/✓/✓/✓/✓ | 3.2e-11 / 5.6e-11 | 0.388 ms | **GO** | 0.044 ms / 3.2e-07 |

**Reading.** The numerical results are identical on the three OSes (NRMS 3.2e-11 / 5.6e-11, same as the Linux Docker build). Functional criteria 1–4 and 7 pass everywhere. Criterion 6 is a per-step latency and depends on the machine: it passes on the macOS runner (Apple M1) and on the local Ryzen 5 5600X, and fails on the shared ubuntu-latest (EPYC 7763, 0.84 ms) and windows-latest runners (0.72 ms). **B3's pre-registered verdict is reported unchanged: NO-GO on those two runners because of criterion 6.**

Declared deviation (2026-10-26, after the first CI run, before this one): the CI exit code uses criteria 1–4 and 7 only, since criterion 6 measures the runner's CPU. The RK4 ×1 column is post-hoc information, not a criterion: one substep is ~9× faster (≤ 0.1 ms on every runner) at NRMS 3e-7, far inside criterion 4's 1 %.

## 2. Modelica Association Reference FMUs v0.0.42 (FMI 3.0 and 2.0, their own binaries)

| OS | Results |
|---|---|
| Linux | 3.0/Dahlquist ME pass, 3.0/Dahlquist CS pass, 3.0/Stair ME pass, 3.0/BouncingBall ME pass, 2.0/Dahlquist ME pass, 2.0/Dahlquist CS pass, 2.0/Stair ME pass |
| Windows | 3.0/Dahlquist ME pass, 3.0/Dahlquist CS pass, 3.0/Stair ME pass, 3.0/BouncingBall ME pass, 2.0/Dahlquist ME pass, 2.0/Dahlquist CS pass, 2.0/Stair ME pass |
| macOS | 3.0/Dahlquist ME pass, 3.0/Dahlquist CS pass, 3.0/Stair ME pass, 3.0/BouncingBall ME pass, 2.0/Dahlquist  skipped, 2.0/Stair  skipped |
| Linux (local, Ryzen 5 5600X) | 3.0/Dahlquist ME pass, 3.0/Dahlquist CS pass, 3.0/Stair ME pass, 3.0/BouncingBall ME pass, 2.0/Dahlquist ME pass, 2.0/Dahlquist CS pass, 2.0/Stair ME pass |

On macOS the FMI 2.0 archive ships no darwin64 binary for these models, so they are skipped (`supported()` reports why); FMI 3.0 (aarch64-darwin) passes.

## 3. BuckSwitching Model Exchange (reported, no verdict: no switching model is distributed)

| OS | ×4 steps / p50 | ×40 steps / p50 | max abs(×4 − ×40) |
|---|---|---|---|
| Linux | 500 / 2.61 ms | 500 / 7.42 ms | 2.83e-03 V |
| Windows | 500 / 2.23 ms | 500 / 6.53 ms | 2.83e-03 V |
| macOS | 500 / 1.45 ms | 500 / 4.03 ms | 2.83e-03 V |
| Linux (local, Ryzen 5 5600X) | no ejecutado: local run without --switching-me | | |

## 4. Windows build

OpenModelica's `RUNTIME_DEPENDENCIES_LEVEL=all` fails on Windows/MinGW (first CI run 40434104485: `cmake --install` cannot resolve the api-ms-/ext-ms- API-set DLLs). `--static-runtime` links libgcc and libwinpthread into the DLL; `objdump -p` on both DLLs lists only KERNEL32.dll and the Universal CRT (api-ms-win-crt-*), present on Windows 10/11. So the FMU runs on a Windows machine without MinGW.

## Not done

- Adding the win64/darwin64 FMUs to a release (step 3): needs the user's confirmation (publishing).
- No ejecutado: a Windows or macOS machine outside GitHub Actions; Intel macOS (macos-latest is arm64).
