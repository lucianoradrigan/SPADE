# Screenshots — PLECS Plant Studio, OpenModelica FMU engine (B5), 2026-10-25

Plant Studio → Create Plant → Use as a Gymnasium plant. Five states of the "Environment readiness" column, light and dark (`plant_studio_fmu_<state>_<theme>.png`, 1440 px wide).

## How they were made

- One local Streamlit server per (state, theme), from `feat/closure-round2`, each with its own `SPADE_WORKSPACE` and `SPADE_FMU_DIR`.
- The theme was set through the browser's colour scheme: the app follows it, and both themes are defined in `.streamlit/config.toml`.
- Capture used Playwright (Chromium, headless).
- The FMUs are the local builds (BuckAveraged `48d52b6a…`, DcMotorAveraged `1c7111be…`).
- Every verification shown was really run, through the `fmu_verification` job and the worker.

| State | File | How it was reached | Stage shown |
|---|---|---|---|
| Not configured | `plant_studio_fmu_not_configured_*` | Nothing imported | 2. Engine |
| Not supported | `plant_studio_fmu_not_supported_*` | DcMotorAveraged (B3 NO-GO) imported and specified: verdict, link to the report, "Reference use only", Publishability "Not supported" | 3. Circuit |
| Needs attention | `plant_studio_fmu_needs_attention_*` | BuckAveraged verified, then the FMU file changed (one extra file in the archive): "Stale — Revalidate", Publish disabled | 6. Validate & publish |
| Failed | `plant_studio_fmu_failed_*` | BuckAveraged with Ts = 0.5 s: the verification fails 4 of 13 checks (the real result of that configuration) | 6. Validate & publish |
| Validated | `plant_studio_fmu_validated_*` | BuckAveraged, Ts = 20 µs: 13/13 checks, published as `SPADE/FmuOmBuck-v0` | 6. Validate & publish |

The page shows only the visible part of Streamlit's scrolling main area, from the stage selector down. The rest of each stage is covered by the AppTests in `tests/fmu/test_plant_studio_fmu.py`.

Not measured: colour contrast, mobile widths, keyboard navigation.
