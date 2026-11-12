# V1 completion (M6.2, 2026-11-12): rotor inertia on the DC Motor and PMSM / FOC pages

The inventory (`inventory_2026-11-12.md`) found that every V1 capability of the Phase 0 simulator was already on the pages (round B) **except the rotor inertia `j_rotor`**. `Scenario` supports it on both plants (`motor_parameter_overrides`, `pmsm_motor_parameters`). Everything shown is **simulated**.

| Page | New control | Initial value (= stock, unchanged) | Reaches |
|---|---|---|---|
| DC Motor → Parameters | Rotor inertia j_rotor (kg·m²) | 0.02500 (`DcPermanentlyExcitedMotor`) | `Scenario.motor_parameter_overrides["j_rotor"]` |
| PMSM / FOC → Motor characteristics | Rotor inertia j_rotor (kg·m²) | 0.03883 (`PermanentMagnetSynchronousMotor`) | the native FOC generator's motor parameters; validated by `Scenario.pmsm_motor_parameters` |

- **No default changed.** At the stock value the control sends nothing, so the stock configuration, its hash and its run are exactly those from before the control existed.
- **Validation:** Scenario's own message is shown with `st.error`, including S-24 (a voltage limit above the supply), never an exception. The widget refuses values below 1e-5 kg·m².
- **Tests:** `tests/test_v1_rotor_inertia.py` (6):
  - the page opens at the stock value and sends nothing;
  - a changed value reaches the Scenario (DC) or the generator (PMSM);
  - an invalid value shows the message;
  - Scenario validates `j_rotor`.
- **Existing V1 tests:** `tests/test_v1_gui_phase0.py` (9) still passes.
- **Screenshots:** `docs/screenshots/frontend/v1/before_*.png` (dev `849721c`, fresh clone) and `after_*.png` (this branch).
