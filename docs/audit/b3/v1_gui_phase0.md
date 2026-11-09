# V1 (round B, T6, 2026-11-09): Phase 0 simulator options in the GUI

**Simulated.** Branch `feat/V1-gui-phase0`. Tests: `tests/test_v1_gui_phase0.py` (9, AppTest), `PYTHONPATH=src:. python -m pytest -q tests/test_v1_gui_phase0.py`.

## DC Motor page: Parameters tab → "Advanced: plant, load, supply, limits"

| Control | Scenario field | Stock value (not sent) |
|---|---|---|
| Plant mode | `plant_config_id`: `dc_perm_ex_v2` (default) or `dc_perm_ex_v1` ("GEM-faithful mode") | `dc_perm_ex_v2` (decision E3) |
| Extra load torque (N·m) | `load_torque_nm` | 0 → not sent |
| Supply voltage (V) | `dc_supply_v` | 65 V → not sent |
| Limits i, u, ω, torque | `dc_limit_overrides` (only the changed keys) | GEM's `DcPermanentlyExcitedMotor.limits` |
| Nominal values i, u, ω, torque | `dc_nominal_overrides` (only the changed keys) | GEM's `nominal_values` |

- **Choosing v1** shows the S-21 warning (`simulation_adapter.S21_V1_WARNING`).
- **The operating-envelope chart** uses the overridden limits.
- **An invalid value** shows `Scenario`'s own `ValueError` message as an error, and nothing runs.

## PMSM / FOC page

- **New control:** DC-link voltage `u_supply`, stock 330 V; it is passed to the FOC cloud generator.
- **The page's motor parameters** (p, l_d, l_q, r_s, ψ_p, u_supply) are checked by `Scenario(plant_config_id="pmsm_v1", pmsm_motor_parameters=…)`. The page shows its error messages and its warnings, for example "l_d > l_q".
- **At the stock supply,** the configuration hash of the page is the same as before (the key is absent).
- **`j_rotor` is not offered:** this page runs current steps with a fixed large load inertia (`pmsm_foc.run_current_step`), so the rotor inertia would not change the result.

## No default changes (tests)

- **With every control at its stock value:**
  - the Scenario has `plant_config_id="dc_perm_ex_v2"` and no new field;
  - the resulting DataFrame equals, frame for frame, the run built with the page's pre-V1 keywords (`test_with_stock_options_the_run_equals_the_run_without_them`).
- **Each control reaches the Scenario** (`test_each_option_reaches_the_scenario`), and v1 is used when chosen.

## Finding S-24 (Low, open, queued)

- **What:** `Scenario` compares the voltage limit with the supply only when both are given explicitly, so `dc_limit_overrides={'u': 100}` is accepted with the stock 65 V supply.
- **Not changed here:** it would change what `Scenario` accepts, which is Luciano's decision.
- **Help text:** the page's text says the check applies when the supply is changed. **Update (round C, 2026-11-09):** S-24 is corrected on `dev` (decision C6), and the help text now says a limit above the supply (the given one, or 65 V) is rejected.

## Not covered

- The **Advanced Flow** mode of the DC Motor page keeps its options and v2. The new controls are in the single-run mode only.
- **Visual check:** not executed. Playwright is not installed on this machine, and installing it would add a dependency.
