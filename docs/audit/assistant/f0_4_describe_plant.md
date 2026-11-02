# F0.4 — `describe_plant()`: what each native plant lets you configure

- **Command:** `python experiments/report_f0_4_describe_plant.py`. Full output: [`f0_4_describe_plant.json`](f0_4_describe_plant.json).
- **Source of truth:** `configs/plant_catalog.yaml` (units, conventions, ranges, importance, descriptions). Defaults are read from the code, not from the YAML.
- **Guard:** `tests/test_assistant_f0_plant_catalog.py`:
  - every `Scenario` field is described or marked `not_applicable` for each of the 4 plants;
  - every described field exists;
  - every range is probed against `Scenario.__post_init__` just inside and just outside each bound (also each key of the dict fields).
  - A mutation check (three ranges altered in memory) was detected by the test each time.

Range notation: `[` inclusive, `(` exclusive; "any finite" = only finiteness is checked; "—" = no numeric range (string, dict or unchecked field).

## `dc_perm_ex_v1` — Permanently excited DC motor (GEM, Heinzmann PMG 132 parameters), one-quadrant converter

Controllers: PI, MPC.

| Field | Type | Unit | Range | Default | Importance | Convention / description |
|---|---|---|---|---|---|---|
| `scenario_id` | str | — | — | — | required | Unique run name; becomes dataset_id and source_file in the export. |
| `controller_type` | str | — | — | PI | required | Controller of the run; each plant accepts only its own controllers. Allowed: PI, MPC. |
| `omega_ref_rad_s` | float | rad/s | any finite | 150.0 | important | Speed reference of the speed PI/MPC. The one-quadrant converter cannot track negative references or brake. Convention: mechanical shaft speed. |
| `fault_type` | str | — | — | — | important | None (healthy), outer_race, inner_race, ball, cage, or a custom label with fault_order_override. |
| `electrical_severity` | float | N*m | [0, ∞) | 8.0 | optional | Electrical fault severity; only acts when a fault is active. Convention: amplitude of the bearing-fault torque ripple (MCSA path). |
| `mechanical_severity` | float | vibration model units | [0, ∞) | — | optional | Mechanical fault severity; default is the calibrated value of the fault type. Convention: impulse amplitude of the synthetic vibration model (Module B). |
| `seed` | int | — | — | 0 | optional | Reproducibility seed (DC: plant randomness, vibration and noise; VSC: the reference's start phase; PMSM and buck: recorded, deterministic runs). |
| `duration_s` | float | s | (0, ∞) | 1.0 | important | Simulated time of the run. |
| `plant_config_id` | str | — | — | dc_perm_ex_v1 | required | Which plant to simulate. |
| `motor_parameter_overrides` | dict | — | — | — | important | Partial override of the DC motor's electrical and mechanical parameters. |
| ↳ `r_a` | float | ohm | (0, ∞) | 0.016 | | armature circuit resistance (two-terminal, brushes included as given by the datasheet); temperature not stated by GEM: prefer the cold value and say so |
| ↳ `l_a` | float | H | (0, ∞) | 1.9e-05 | | armature circuit inductance |
| ↳ `psi_e` | float | V*s/rad | (0, ∞) | 0.165 | | flux constant = back-EMF constant Ke in V*s/rad = torque constant Kt in N*m/A (SI) |
| ↳ `j_rotor` | float | kg*m^2 | (0, ∞) | 0.025 | | rotor inertia only; the load adds j_load = 0.0025 |
| `electrical_noise_pct` | float | % | [0, ∞) | 0.0 | optional | Sensor noise on the current, added after the run. Convention: white noise on current_r, as % of the run's own standard deviation. |
| `mechanical_noise_pct` | float | % | [0, ∞) | 0.0 | optional | Sensor noise on the vibration channels, added after the run. Convention: white noise on each acceleration axis, as % of that axis's standard deviation. |
| `torque_ref_nm` | float | N*m | any finite | — | optional | Torque-control mode (bypasses the speed loop); omega_ref_rad_s is then ignored. |
| `fault_order_override` | float | multiple of shaft speed | (0, ∞) | — | optional | Characteristic fault frequency as an order; makes fault_type a free label. |
| `extra_faults` | list | — | — | — | optional | Additional simultaneous faults, as (fault_type, order_override) pairs. |
| `pi_setpoint_weight` | float | — | [0, 1] | — | optional | Setpoint weight b of the speed PI's proportional term (PI only). |
| `dc_plant_solver` | str | — | — | — | optional | Reference integration of the DC plant (studies only; None = one Euler step per period). Allowed: euler, rk45. |
| `dc_plant_substeps` | int | — | [1, 10000] | — | optional | Euler substeps per control period (with dc_plant_solver euler or unset). |
| `load_torque_nm` | float | N*m | [0, ∞) | — | important | Constant load torque; '20 % load' must be converted with the motor's rated torque by the caller. Convention: absolute passive load torque (Coulomb-type, opposes rotation), added to the default friction constant a (0.01 for the stock motor; 1 % of rated torque for another motor with known ratings). |
| `dc_supply_v` | float | V | (0, ∞) | — | important | Supply voltage. Convention: DC link of the one-quadrant converter. |
| `dc_limit_overrides` | dict | — | — | — | important | Motor limits: normalisation and the controllers' current cap (PI 0.8 x i, MPC 0.85 x i). u must not exceed dc_supply_v. |
| ↳ `omega` | float | rad/s | (0, ∞) | 400 | | mechanical speed |
| ↳ `torque` | float | N*m | (0, ∞) | 38.0 | | — |
| ↳ `i` | float | A | (0, ∞) | 210 | | armature current; for a datasheet, the peak (short-time) current, not the continuous one |
| ↳ `u` | float | V | (0, ∞) | 60 | | armature voltage |
| `dc_nominal_overrides` | dict | — | — | — | optional | Rated values (continuous duty, S1); the tuning's J_effort uses the rated current. |
| ↳ `omega` | float | rad/s | (0, ∞) | 300 | | rated mechanical speed |
| ↳ `torque` | float | N*m | (0, ∞) | 16.0 | | rated (continuous) torque |
| ↳ `i` | float | A | (0, ∞) | 97 | | rated continuous current (S1) |
| ↳ `u` | float | V | (0, ∞) | 60 | | rated armature voltage |

Not applicable here: `load_resistance_ohm`, `reference_magnitude_v`, `reference_omega_rad_s`, `pmsm_current_ref_a`, `vsc_voltage_limit_v`, `vsc_capacitor_scale`, `vsc_esr_ohm`, `buck_vout_ref_v`, `buck_vin_v`, `buck_load_step_ohm`, `buck_load_step_time_s`, `buck_parameter_overrides`, `plecs_model_vars`, `plecs_time_limit_s`, `plecs_output_points`, `pmsm_motor_parameters`.

**Stock motor:** parameters {'r_a': 0.016, 'l_a': 1.9e-05, 'psi_e': 0.165, 'j_rotor': 0.025}; limits {'omega': 400, 'torque': 38.0, 'i': 210, 'u': 60}; nominal {'omega': 300, 'torque': 16.0, 'i': 97, 'u': 60}; supply 65 V; load {'a': 0.01, 'b': 0.05, 'c': 0.0, 'j_load': 0.0025}; j_total 0.0275 kg·m²; control period 0.0001 s.

**Design check:** PI **ok** (τ_a 1.19 ms, τ_m 16.2 ms, τ_m/τ_a 13.6, bandwidth ratio 62.5); MPC **ok** (horizon/τ_a 1.68).

## `dc_pmg132_v1` — Permanently excited DC motor with the Perm-Motor PMG 132 datasheet values at 60 V (S-20), one-quadrant converter

Controllers: PI, MPC.

| Field | Type | Unit | Range | Default | Importance | Convention / description |
|---|---|---|---|---|---|---|
| `scenario_id` | str | — | — | — | required | Unique run name; becomes dataset_id and source_file in the export. |
| `controller_type` | str | — | — | PI | required | Controller of the run; each plant accepts only its own controllers. Allowed: PI, MPC. |
| `omega_ref_rad_s` | float | rad/s | any finite | 150.0 | important | Speed reference of the speed PI/MPC. The one-quadrant converter cannot track negative references or brake. Convention: mechanical shaft speed. |
| `fault_type` | str | — | — | — | important | None (healthy), outer_race, inner_race, ball, cage, or a custom label with fault_order_override. |
| `electrical_severity` | float | N*m | [0, ∞) | 8.0 | optional | Electrical fault severity; only acts when a fault is active. Convention: amplitude of the bearing-fault torque ripple (MCSA path). |
| `mechanical_severity` | float | vibration model units | [0, ∞) | — | optional | Mechanical fault severity; default is the calibrated value of the fault type. Convention: impulse amplitude of the synthetic vibration model (Module B). |
| `seed` | int | — | — | 0 | optional | Reproducibility seed (DC: plant randomness, vibration and noise; VSC: the reference's start phase; PMSM and buck: recorded, deterministic runs). |
| `duration_s` | float | s | (0, ∞) | 1.0 | important | Simulated time of the run. |
| `plant_config_id` | str | — | — | — | required | Which plant to simulate. |
| `motor_parameter_overrides` | dict | — | — | — | important | Partial override of the DC motor's electrical and mechanical parameters. |
| ↳ `r_a` | float | ohm | (0, ∞) | 0.016 | | armature circuit resistance (two-terminal, brushes included as given by the datasheet); temperature not stated by GEM: prefer the cold value and say so |
| ↳ `l_a` | float | H | (0, ∞) | 1.9e-05 | | armature circuit inductance |
| ↳ `psi_e` | float | V*s/rad | (0, ∞) | 0.181 | | flux constant = back-EMF constant Ke in V*s/rad = torque constant Kt in N*m/A (SI) |
| ↳ `j_rotor` | float | kg*m^2 | (0, ∞) | 0.025 | | rotor inertia only; the load adds j_load = 0.0025 |
| `electrical_noise_pct` | float | % | [0, ∞) | 0.0 | optional | Sensor noise on the current, added after the run. Convention: white noise on current_r, as % of the run's own standard deviation. |
| `mechanical_noise_pct` | float | % | [0, ∞) | 0.0 | optional | Sensor noise on the vibration channels, added after the run. Convention: white noise on each acceleration axis, as % of that axis's standard deviation. |
| `torque_ref_nm` | float | N*m | any finite | — | optional | Torque-control mode (bypasses the speed loop); omega_ref_rad_s is then ignored. |
| `fault_order_override` | float | multiple of shaft speed | (0, ∞) | — | optional | Characteristic fault frequency as an order; makes fault_type a free label. |
| `extra_faults` | list | — | — | — | optional | Additional simultaneous faults, as (fault_type, order_override) pairs. |
| `pi_setpoint_weight` | float | — | [0, 1] | — | optional | Setpoint weight b of the speed PI's proportional term (PI only). |
| `dc_plant_solver` | str | — | — | — | optional | Reference integration of the DC plant (studies only; None = one Euler step per period). Allowed: euler, rk45. |
| `dc_plant_substeps` | int | — | [1, 10000] | — | optional | Euler substeps per control period (with dc_plant_solver euler or unset). |
| `load_torque_nm` | float | N*m | [0, ∞) | — | important | Constant load torque; '20 % load' must be converted with the motor's rated torque by the caller. Convention: absolute passive load torque (Coulomb-type, opposes rotation), added to the default friction constant a (0.01 for the stock motor; 1 % of rated torque for another motor with known ratings). |
| `dc_supply_v` | float | V | (0, ∞) | — | important | Supply voltage. Convention: DC link of the one-quadrant converter. |
| `dc_limit_overrides` | dict | — | — | — | important | Motor limits: normalisation and the controllers' current cap (PI 0.8 x i, MPC 0.85 x i). u must not exceed dc_supply_v. |
| ↳ `omega` | float | rad/s | (0, ∞) | 400.0 | | mechanical speed |
| ↳ `torque` | float | N*m | (0, ∞) | 38.0 | | — |
| ↳ `i` | float | A | (0, ∞) | 200.0 | | armature current; for a datasheet, the peak (short-time) current, not the continuous one |
| ↳ `u` | float | V | (0, ∞) | 60.0 | | armature voltage |
| `dc_nominal_overrides` | dict | — | — | — | optional | Rated values (continuous duty, S1); the tuning's J_effort uses the rated current. |
| ↳ `omega` | float | rad/s | (0, ∞) | 300.54569719342356 | | rated mechanical speed |
| ↳ `torque` | float | N*m | (0, ∞) | 20.0 | | rated (continuous) torque |
| ↳ `i` | float | A | (0, ∞) | 110.0 | | rated continuous current (S1) |
| ↳ `u` | float | V | (0, ∞) | 60.0 | | rated armature voltage |

Not applicable here: `load_resistance_ohm`, `reference_magnitude_v`, `reference_omega_rad_s`, `pmsm_current_ref_a`, `vsc_voltage_limit_v`, `vsc_capacitor_scale`, `vsc_esr_ohm`, `buck_vout_ref_v`, `buck_vin_v`, `buck_load_step_ohm`, `buck_load_step_time_s`, `buck_parameter_overrides`, `plecs_model_vars`, `plecs_time_limit_s`, `plecs_output_points`, `pmsm_motor_parameters`.

**Stock motor:** parameters {'r_a': 0.016, 'l_a': 1.9e-05, 'psi_e': 0.181, 'j_rotor': 0.025}; limits {'omega': 400.0, 'torque': 38.0, 'i': 200.0, 'u': 60.0}; nominal {'omega': 300.54569719342356, 'torque': 20.0, 'i': 110.0, 'u': 60.0}; supply 65 V; load {'a': 0.2, 'b': 0.0006654562080497367, 'c': 0.0, 'j_load': 0.0025}; j_total 0.0275 kg·m²; control period 0.0001 s.

**Design check:** PI **ok** (τ_a 1.19 ms, τ_m 13.4 ms, τ_m/τ_a 11.3, bandwidth ratio 62.5); MPC **ok** (horizon/τ_a 1.68).

## `pmsm_v1` — Salient PMSM, dq current control near standstill (current-step bench)

Controllers: FOC_MTPA, FOC_naive.

| Field | Type | Unit | Range | Default | Importance | Convention / description |
|---|---|---|---|---|---|---|
| `scenario_id` | str | — | — | — | required | Unique run name; becomes dataset_id and source_file in the export. |
| `controller_type` | str | — | — | — | required | Controller of the run; each plant accepts only its own controllers. Allowed: FOC_MTPA, FOC_naive. |
| `seed` | int | — | — | 0 | optional | Reproducibility seed (DC: plant randomness, vibration and noise; VSC: the reference's start phase; PMSM and buck: recorded, deterministic runs). |
| `duration_s` | float | s | (0, ∞) | 1.0 | important | Simulated time of the run. |
| `plant_config_id` | str | — | — | — | required | Which plant to simulate. |
| `pmsm_current_ref_a` | float | A | [-400, 400] | — | required | Stator current reference, split into (i_d, i_q) by the policy. Convention: signed stator current magnitude \|i_s\| (amplitude-invariant dq, peak per phase); sign = sign of i_q. |
| `pmsm_motor_parameters` | dict | — | — | — | important | Partial override of the PMSM. MTPA assumes l_q >= l_d. |
| ↳ `p` | int | — | [1, ∞) | 3 | | pole PAIRS, not poles |
| ↳ `l_d` | float | H | (0, ∞) | 0.00037 | | d-axis inductance per phase (star equivalent); a line-to-line value is about 2x |
| ↳ `l_q` | float | H | (0, ∞) | 0.0012 | | q-axis inductance per phase (star equivalent) |
| ↳ `r_s` | float | ohm | (0, ∞) | 0.018 | | stator resistance per phase (star equivalent) = R_line-line / 2; cold value |
| ↳ `psi_p` | float | V*s | (0, ∞) | 0.066 | | permanent-magnet flux linkage, PEAK per phase (torque = 1.5*p*psi_p*i_q with amplitude-invariant dq) |
| ↳ `j_rotor` | float | kg*m^2 | (0, ∞) | 0.03883 | | rotor inertia; the bench adds a 5 kg*m^2 load inertia |
| ↳ `u_supply` | float | V | (0, ∞) | 330.0 | | DC link of the B6 bridge |

Not applicable here: `omega_ref_rad_s`, `fault_type`, `electrical_severity`, `mechanical_severity`, `motor_parameter_overrides`, `electrical_noise_pct`, `mechanical_noise_pct`, `torque_ref_nm`, `fault_order_override`, `extra_faults`, `load_resistance_ohm`, `reference_magnitude_v`, `reference_omega_rad_s`, `pi_setpoint_weight`, `vsc_voltage_limit_v`, `vsc_capacitor_scale`, `vsc_esr_ohm`, `buck_vout_ref_v`, `buck_vin_v`, `buck_load_step_ohm`, `buck_load_step_time_s`, `buck_parameter_overrides`, `plecs_model_vars`, `plecs_time_limit_s`, `plecs_output_points`, `dc_plant_solver`, `dc_plant_substeps`, `load_torque_nm`, `dc_supply_v`, `dc_limit_overrides`, `dc_nominal_overrides`.

## `vsc_dpc_v1` — Voltage source converter with L-C filter, DPC neural controller

Controllers: DPC.

| Field | Type | Unit | Range | Default | Importance | Convention / description |
|---|---|---|---|---|---|---|
| `scenario_id` | str | — | — | — | required | Unique run name; becomes dataset_id and source_file in the export. |
| `controller_type` | str | — | — | — | required | Controller of the run; each plant accepts only its own controllers. Allowed: DPC. |
| `seed` | int | — | — | 0 | optional | Reproducibility seed (DC: plant randomness, vibration and noise; VSC: the reference's start phase; PMSM and buck: recorded, deterministic runs). |
| `duration_s` | float | s | (0, ∞) | 1.0 | important | Simulated time of the run. |
| `plant_config_id` | str | — | — | — | required | Which plant to simulate. |
| `load_resistance_ohm` | float | ohm | (0, ∞) | — | important | Output load. VSC below R* = 3.37 ohm is open-loop unstable (physics, not rejected). Convention: resistive load across the output capacitor. |
| `reference_magnitude_v` | float | V | [0, ∞) | — | important | Reference voltage magnitude; the network was trained at 50 V. Convention: magnitude of the rotating voltage reference (complex phasor). |
| `reference_omega_rad_s` | float | rad/s | any finite | — | optional | Reference frequency; the network was trained at 50 Hz. Convention: electrical angular frequency of the reference. |
| `vsc_voltage_limit_v` | float | V | (0, ∞) | — | optional | Converter voltage limit; 134 V reproduces the original converter (S-09). None: unlimited. Convention: per-axis clamp of the converter voltage command. |
| `vsc_capacitor_scale` | float | fraction of 15 uF | (0, ∞) | — | optional | Synthetic capacitor degradation (effective C); synthetic-only fault. |
| `vsc_esr_ohm` | float | ohm | [0, ∞) | — | optional | Synthetic series ESR of the filter capacitor. |

Not applicable here: `omega_ref_rad_s`, `fault_type`, `electrical_severity`, `mechanical_severity`, `motor_parameter_overrides`, `electrical_noise_pct`, `mechanical_noise_pct`, `torque_ref_nm`, `fault_order_override`, `extra_faults`, `pi_setpoint_weight`, `pmsm_current_ref_a`, `buck_vout_ref_v`, `buck_vin_v`, `buck_load_step_ohm`, `buck_load_step_time_s`, `buck_parameter_overrides`, `plecs_model_vars`, `plecs_time_limit_s`, `plecs_output_points`, `dc_plant_solver`, `dc_plant_substeps`, `load_torque_nm`, `dc_supply_v`, `dc_limit_overrides`, `dc_nominal_overrides`, `pmsm_motor_parameters`.

## `buck_v1` — Averaged synchronous buck converter, PI cascade

Controllers: PI.

| Field | Type | Unit | Range | Default | Importance | Convention / description |
|---|---|---|---|---|---|---|
| `scenario_id` | str | — | — | — | required | Unique run name; becomes dataset_id and source_file in the export. |
| `controller_type` | str | — | — | PI | required | Controller of the run; each plant accepts only its own controllers. Allowed: PI. |
| `seed` | int | — | — | 0 | optional | Reproducibility seed (DC: plant randomness, vibration and noise; VSC: the reference's start phase; PMSM and buck: recorded, deterministic runs). |
| `duration_s` | float | s | (0, ∞) | 1.0 | important | Simulated time of the run. |
| `plant_config_id` | str | — | — | — | required | Which plant to simulate. |
| `load_resistance_ohm` | float | ohm | (0, ∞) | — | important | Output load. VSC below R* = 3.37 ohm is open-loop unstable (physics, not rejected). Convention: resistive load across the output capacitor. |
| `buck_vout_ref_v` | float | V | [0, 48) | — | important | Output voltage reference. (upper bound is buck_vin_v (48 V by default): a buck cannot step up) |
| `buck_vin_v` | float | V | (24, ∞) | — | important | Input voltage. (must exceed buck_vout_ref_v (24 V by default); on its own only > 0 is required) |
| `buck_load_step_ohm` | float | ohm | (0, ∞) | — | optional | Load after the step; None: no step. |
| `buck_load_step_time_s` | float | s | [0, ∞) | — | optional | Time of the load step. |
| `buck_parameter_overrides` | dict | — | — | — | optional | Partial override of the buck's components. |
| ↳ `l_h` | float | H | (0, ∞) | 0.00022 | | filter inductance |
| ↳ `c_f` | float | F | (0, ∞) | 0.00022 | | output capacitance |
| ↳ `r_l_ohm` | float | ohm | [0, ∞) | 0.05 | | inductor series resistance |
| ↳ `i_max_a` | float | A | (0, ∞) | 15.0 | | inductor current limit of the controller |

Not applicable here: `omega_ref_rad_s`, `fault_type`, `electrical_severity`, `mechanical_severity`, `motor_parameter_overrides`, `electrical_noise_pct`, `mechanical_noise_pct`, `torque_ref_nm`, `fault_order_override`, `extra_faults`, `reference_magnitude_v`, `reference_omega_rad_s`, `pi_setpoint_weight`, `pmsm_current_ref_a`, `vsc_voltage_limit_v`, `vsc_capacitor_scale`, `vsc_esr_ohm`, `plecs_model_vars`, `plecs_time_limit_s`, `plecs_output_points`, `dc_plant_solver`, `dc_plant_substeps`, `load_torque_nm`, `dc_supply_v`, `dc_limit_overrides`, `dc_nominal_overrides`, `pmsm_motor_parameters`.
