# T1: findings registered from E10a and E10b (round 2026-10-21)

Decision D5 of `DECISIONES_EXPANSION_2026-10-21.md`. Branch `feat/E10-findings` from `dev` (acebd43). IDs: the next free ones after AI-N14 and S-13. Raw: [`t1_findings.json`](t1_findings.json).

| ID | Type | Severity | Status | Finding | Evidence (command) | Tests |
|---|---|---|---|---|---|---|
| AI-N15 | AI | Medium | open | The VSC PINN estimates C systematically low: mean error -1.3/-3.0/-4.5/-9.1 % at 0/1/2/5 % noise (worst -24.3 %). | `python experiments/report_pinn_identification.py` | `tests/test_findings_e10.py::test_ai_n15_pinn_capacitance_unbiased_with_noise (strict xfail)`; `tests/test_findings_e10.py::test_ai_n15_bias_is_recorded_as_negative_and_growing_with_noise` |
| AI-N16 | AI | Medium | open | The PINN 5-95 % bootstrap interval is not calibrated: C coverage 0/52/70/78 % at 0/1/2/5 % noise vs 90 % nominal. GUI now shows 'Coverage not validated.' | `python experiments/report_pinn_identification.py` | `tests/test_findings_e10.py::test_ai_n16_pinn_interval_has_nominal_coverage (strict xfail)`; `tests/test_findings_e10.py::test_ai_n16_gui_says_coverage_not_validated` |
| AI-N17 | AI | High | open | The promoted dc_motor PC classifier (pc_server_5class_w2048/2026-10-20_run01) has no OOD detector; the A1 detector flagged 13 % of ball windows (AI-N13). Not integrated by decision D6. | `python experiments/report_ood_detector.py (A1); metrics.json of the promoted run` | `tests/test_findings_e10.py::test_ai_n17_promoted_classifier_has_no_ood_detector` |
| S-14 | Simulation | Medium | documented | DPC closed loop diverges as C degrades; threshold depends on load: C<=x0.84 at 4 ohm, <=x0.67 at 5 ohm, <=x0.42 at 8.0064 ohm; stable to x0.40 at 10-20 ohm. | `python experiments/measure_dpc_capacitor_divergence.py` | `tests/test_findings_e10.py::test_s14_dpc_loop_vs_capacitor_at_the_nominal_load[0.5-False]`; `tests/test_findings_e10.py::test_s14_dpc_loop_vs_capacitor_at_the_nominal_load[0.4-True]` |

Changes:

- **Findings written up** in `docs/AI_TECHNICAL_AUDIT.md`: new §9, and three rows in the summary table of §6.
- **S-14** added to the findings table of `docs/SIMULATION_STABILITY_REPORT.md` (§6).
- **New script** `experiments/measure_dpc_capacitor_divergence.py`, which writes `docs/audit/stability/dpc_capacitor_divergence.{md,json}`.
  - It scans C from x0.70 down to x0.40 in steps of 0.01, on the stable loads of R_GRID (≥ 4 ohm).
  - Loads that already diverge at x0.70 are scanned upward to x1.00.
  - Runtime about 9 min.
- **GUI:** the VSC Parameter Identification tab shows "Coverage not validated.", with the measured coverage and, for the PINN, the C bias.
- **No default, model or threshold changed.**
