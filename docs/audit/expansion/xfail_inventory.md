# xfail inventory (T4, round 2026-10-22; updated in T8: 6 xfails after T6)

- **Commands:**
  - `grep -rn "xfail" tests conftest.py pyproject.toml`
  - `pytest -rx -q`: the full suite on `bca4396` gives 1131 passed and 5 xfailed (`round_2026-10-22_baseline.txt`).
- **Guard:** `tests/test_xfail_inventory.py` fails if any xfail lacks `strict=True` or a finding ID (`AI-N…` / `S-…`) in its reason. It also fails if the cited ID is not defined in `docs/AI_TECHNICAL_AUDIT.md` or `docs/SIMULATION_STABILITY_REPORT.md`.

| # | Test | Strict | Finding | What it asserts (the property SPADE does not have yet) | Status of the defect |
|---|---|---|---|---|---|
| 1 | `tests/test_findings_e10.py::test_ai_n15_pinn_capacitance_unbiased_with_noise` | yes | AI-N15 | PINN capacitance estimate unbiased at 1–5 % noise | Open (−1.3 % to −9.1 % mean error) |
| 2 | `tests/test_findings_e10.py::test_ai_n16_pinn_interval_has_nominal_coverage` | yes | AI-N16 | PINN 5–95 % bootstrap interval covers the truth ≥ 90 % | Open (52–78 %) |
| 3 | `tests/test_findings_e14.py::test_ai_n18_gui_detector_flags_most_faulted_windows` | yes | AI-N18 | run01 residual detector flags ≥ 90 % of faulted windows | Cause addressed in run02, not adopted (AI-N21); run01 still fails |
| 4 | `tests/test_findings_e14.py::test_ai_n20_edge_detectors_usable` | yes | AI-N20 | Distilled RPi5/ESP32 residual detectors usable | Open |
| 5 | `tests/test_findings_e14.py::test_ai_n21_gated_detector_meets_the_false_alarm_criterion` | yes | AI-N21 | Gated (run02) residual detector ≤ 1 % false alarms at 100 % noise | Not fixed: superseded by AI-N22 (E14 closed, negative) |
| 6 | `tests/test_findings_e14.py::test_ai_n22_noise_aware_detector_meets_the_false_alarm_criterion` | yes | AI-N22 | Noise-aware (T6) detector meets criterion 3 | Closed as a negative result; added in T6 |
| 7–8 | `tests/fmu/test_om_fmus.py::test_switching_fmus_simulate_10_ms_at_the_pwm_period[BuckSwitching, DcMotorSwitching]` | yes | S-15 | OpenModelica switching FMUs simulate 10 ms at the PWM period | Open; added in B1 (round PLECS/OM, 2026-10-24). Skipped when the FMUs are not built. The outcome depends on the build, so an XPASS on another build fails the suite on purpose |

- **Non-compliant xfails:** none. No finding had to be assigned or created, and no xfail was converted to a normal test: none of the five defects is fixed.
- **Other skip mechanisms:** none. There are no `pytest.xfail()` calls inside tests and no conftest-level xfail.
