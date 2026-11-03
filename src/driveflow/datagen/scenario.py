"""Scenario definitions: one Scenario = one simulation run's parameters (load, controller, seed,
fault type -- per INSTRUCTIONS.md A.4). Kept deliberately small/explicit rather than a generic
config-object soup, since Macro-fase A only needs to vary a handful of things (Sec. A.4).
"""

import math
import numbers
from dataclasses import dataclass

#: Per-fault-type Module B impulse amplitude that best matches real Paderborn separability
#: (AUC), found by grid search in experiments/verify_vibration_separability_auc.py -- see
#: docs/patch2_retiro_modulo_C.md Sec. 4's result table. Only outer_race/inner_race were
#: validated against real data (KAt-DataCenter has no clean single-defect ball/cage code, see
#: docs/patch3_mejora_modulo_B.md's Paso 0); ball/cage reuse inner_race's (more conservative,
#: harder-to-detect) value rather than guessing a third number with no evidence behind it.
CALIBRATED_MECHANICAL_SEVERITY = {
    "outer_race": 0.05,
    "inner_race": 0.02,
    "ball": 0.02,
    "cage": 0.02,
}


@dataclass(frozen=True)
class Scenario:
    """One dataset-generation run.

    Args:
        scenario_id: Unique name for this run (becomes dataset_id/source_file in the export).
        controller_type: "PI" or "MPC" (both against plant_config_id="dc_perm_ex_v1" -- the same
            DcMotorSystem, genuinely interchangeable controllers for it, see
            control/mpc/controller.py) or "DPC" (Macro-fase B, against
            plant_config_id="vsc_dpc_v1" -- a Voltage Source Converter, NOT the DC motor; see
            runner.py's module docstring on why DPC is two unrelated plants away from PI/MPC, not
            a controller swap on the same system -- docs/patch5_alcance_macrofase_B.md,
            reconfirmed with a real MPC in hand by docs/patch10_implementacion_mpc.md).
        omega_ref_rad_s: Target speed for the whole run (PI/dc_perm_ex_v1 only; ignored for DPC
            scenarios, which have no rotating machinery -- see runner.py).
        fault_type: None (healthy) or one of bearing_frequencies.FAULT_FREQUENCY_FUNCS' keys, or
            any other string naming a user-defined fault when fault_order_override is also set.
            PI/dc_perm_ex_v1 only; ignored for DPC scenarios (no bearing in a VSC).
        fault_order_override: dc_perm_ex_v1 only -- characteristic frequency as a multiple of
            shaft speed, bypassing bearing_frequencies.fault_order()'s geometry-based lookup for
            both the electrical (BearingFaultLoad) and mechanical (FaultImpulseGenerator) paths.
            None (default) uses fault_type's built-in geometry-derived order, the original
            behavior; fault_type must then be one of the 4 recognized keys. When set, fault_type
            becomes a free-form label (dashboard's "Custom fault types") -- e.g. a gear-mesh
            order, a different bearing's known BPFO, unbalance (order=1), misalignment (order=2).
            Load-zone amplitude modulation (see LoadZoneModulator) never applies to a custom
            order -- that needs defect-position physics this project has no model for on an
            arbitrary user-defined fault, so amplitude is constant instead (same treatment as the
            built-in outer_race/cage types already get).
        extra_faults: dc_perm_ex_v1 only -- optional list of (fault_type, order_override) pairs
            for ADDITIONAL simultaneous faults, layered on top of the primary fault_type above --
            dashboard's "Combine with additional fault types", for exploring compound/mixed-fault
            scenarios (e.g. outer_race + a custom gear-mesh order together). Each extra shares
            electrical_severity/mechanical_severity with the primary (no separate per-fault
            severity -- this is about which fault frequencies are simultaneously present, not
            per-fault amplitude tuning). order_override works the same as the primary's own
            field: None resolves via the geometry-based lookup (fault_type must then be a
            recognized key), given bypasses it (fault_type can then be any label). None/empty
            (default): unchanged single-fault behavior.
        electrical_severity: BearingFaultLoad's torque-ripple amplitude (Nm) -- the electrical/
            MCSA path (datagen/fault_injection.py). Independent of mechanical_severity: they
            drive two physically distinct excitation mechanisms with unrelated natural scales
            (Nm of torque ripple vs. Module B's own impulse-amplitude units) -- a single shared
            "severity" was tried first and found to conflate the two (a value realistic for one
            path saturated or under-drove the other), see docs/patch2_retiro_modulo_C.md Sec. 4.
        mechanical_severity: VibrationSynthesizer.set_fault's impulse amplitude (Module B, own
            units). Defaults to the per-fault-type CALIBRATED_MECHANICAL_SEVERITY value when not
            given explicitly, matching real Paderborn separability (AUC) rather than an
            arbitrary round number.
        seed: Simulation seed. For PI/dc_perm_ex_v1, seeds SCMLSystem's randomness. For
            DPC/vsc_dpc_v1 (a deterministic plant+controller, no randomness to seed), reused
            instead to pick the reference's starting phase (seed maps to a phase in [0, 2*pi) --
            see runner.py) so different seeds still give distinguishable, reproducible runs.
        duration_s: Run length.
        plant_config_id: "dc_perm_ex_v1" (default, PI) or "vsc_dpc_v1" (DPC) -- must match
            controller_type (validated below); each controller_type only has one plant defined.
        motor_parameter_overrides: dc_perm_ex_v1 only -- optional partial override dict for
            DcPermanentlyExcitedMotor's motor_parameter (r_a, l_a, psi_e, j_rotor), merged over
            the class's own defaults by GEM's own ElectricMotor constructor (not re-implemented
            here). None (default) uses GEM's stock motor unmodified.
        electrical_noise_pct: dc_perm_ex_v1 only -- white Gaussian noise added to current_r after
            the run, sized as this percentage of current_r's own std dev over the run (0 = none,
            the default -- output matches the raw simulation exactly). Unlike electrical_severity
            (a specific fault's torque-ripple amplitude), this is generic sensor-style noise:
            applied regardless of fault_type, always-on measurement jitter rather than a physical
            fault mechanism.
        mechanical_noise_pct: dc_perm_ex_v1 only -- same idea as electrical_noise_pct, applied
            independently to acc_x/acc_y/acc_z (each sized as a percentage of that axis's own std
            dev), on top of whatever Module B already produced (background noise +, if fault_type
            is set, fault impulses).
        load_resistance_ohm: vsc_dpc_v1 only -- the resistive load R in i_load = v_c/R, fed to
            BOTH the plant (VscSystem) and the DPC network itself (DpcController.control() puts
            r_ohm directly into the network's input row -- see control/dpc/controller.py). None
            (default) uses runner.py's _VSC_R_OHM (8.0064 Ohm), the constant value found in
            Data4train.mat's own "r" column across all 10000 rows -- the DPC network has only
            ever seen this one R at training time. Setting a different value is a genuine
            robustness probe (does tracking degrade when the real load doesn't match what the
            network was trained against?), not a validated operating point -- there is no data
            confirming how well the network generalizes away from 8.0064 Ohm.
        reference_magnitude_v: vsc_dpc_v1 only -- |v_ref|, the rotating reference's constant
            magnitude (Volts). None (default) uses RotatingReference's own default (50.0V),
            verified directly against Data4train.mat's holdout split -- see
            control/dpc/reference.py's module docstring. Same off-distribution caveat as
            load_resistance_ohm: the network has only ever seen 50.0V during training.
        reference_omega_rad_s: vsc_dpc_v1 only -- the rotating reference's angular frequency
            (rad/s). None (default) uses RotatingReference's own default (2*pi*50, grid
            frequency) -- same off-distribution caveat as the other two VSC/DPC-only fields
            above.
        torque_ref_nm: dc_perm_ex_v1 only -- if set, PICascadeController runs in torque-control
            mode (PICascadeController.control_torque(): bypasses the outer speed loop, feeds the
            inner current loop i_ref = torque_ref_nm/psi_e directly) instead of the default speed
            control. None (default) uses omega_ref_rad_s as a speed setpoint via the full cascade,
            the existing/original behavior. When set, omega_ref_rad_s has no effect (there is no
            speed reference to track in this mode) -- mutually exclusive, like fault_type/
            electrical_severity being ignored for DPC scenarios.
        pi_setpoint_weight: dc_perm_ex_v1 with controller_type="PI" only -- setpoint weight b in
            [0, 1] of the speed loop's proportional term (PICascadeController, S-04). None
            (default) keeps the controller's own default b = 1, the original behavior.
        pmsm_current_ref_a: pmsm_v1 only (required there) -- the stator current magnitude
            reference |i_s| in A, signed: its sign is the sign of i_q (motoring / braking torque).
            controller_type "FOC_MTPA" splits it into (i_d, i_q) on the MTPA locus
            (pmsm_foc.mtpa_id_iq); "FOC_naive" uses i_d = 0, i_q = i_s. Same two policies, same
            dq current controller and the same plant (pmsm_foc.build_pmsm_system, large load
            inertia, near standstill) as the PMSM page. |i_s| <= 400 A. No fault model exists for
            this plant: every record is labelled "normal".
        vsc_voltage_limit_v: vsc_dpc_v1 only -- optional per-axis clamp of the converter voltage
            command v_o (V). None (default): no limit, the original behavior. 134 V reproduces the
            original converter (V_dc = 200 V, sim.vsc_params.UPSTREAM_CONVERTER_LIMIT_V; S-09).
        vsc_capacitor_scale / vsc_esr_ohm: vsc_dpc_v1 only -- synthetic filter-capacitor
            degradation (sim/vsc_faults.py): effective C as a fraction of the nominal 15 uF, and a
            series ESR in ohm. None (default) keeps the nominal identified plant bit for bit.
            SYNTHETIC-ONLY: no real reference data exists for this fault (A4).
        buck_vout_ref_v / buck_vin_v / buck_load_step_ohm / buck_load_step_time_s /
            buck_parameter_overrides: buck_v1 only (expansion task E6; sim/buck_system.py, averaged
            synchronous buck under control/classical/buck_pi.BuckVoltageController).
            buck_vout_ref_v is the output-voltage reference (default 24 V, must be below V_in);
            buck_vin_v overrides V_in (default 48 V); load_resistance_ohm, shared with the VSC,
            is the initial load (default 6 ohm); buck_load_step_ohm switches the load to that value
            at buck_load_step_time_s (default: half of duration_s); buck_parameter_overrides sets
            any of l_h, c_f, r_l_ohm, i_max_a. No fault model: every record is labelled "normal".
        dc_converter_model: DC plants only (S-21, decision S21-2). None or "gem" = GEM's averaged
            one-quadrant converter, decided once per period (the default, byte-identical). "dcm" =
            discontinuous conduction: the converter is re-evaluated in each of dc_plant_substeps
            substeps (default runner.DCM_DEFAULT_SUBSTEPS) and the current stays at zero when the
            diode blocks (sim/dc_dcm.py). A study option; it does not change any default.
        dc_plant_solver / dc_plant_substeps: dc_perm_ex_v1 only (round PLECS/OM task R2, decision
            G3; finding S-16) -- how the plant is integrated over each control period TAU = 100 us.
            The controller still acts every TAU and its action (including the converter branch) is
            held for the whole period; only the plant integration changes. dc_plant_solver "euler"
            (forward Euler, dc_plant_substeps substeps per TAU) or "rk45" (scipy solve_ivp RK45 at
            rtol = atol = 1e-9, the fine solver of the validation benches; dc_plant_substeps must
            then be unset). None for both (default) is the production plant: one Euler step per
            TAU, bit for bit. Not a default anywhere; for reference studies only.
        load_torque_nm: dc_perm_ex_v1 only (round Assistant, task F0.1, decisions A1/A2) -- a
            constant passive load torque in N*m (absolute, >= 0), added to the constant term of
            the stock load: PolynomialStaticLoad's a becomes 0.01 + load_torque_nm. Like a, it
            always opposes the rotation (Coulomb-type, it cannot drive the shaft) and is smoothed
            linearly for |omega| below a*tau_decay/j_total so the integration stays stable at
            standstill. It applies in speed control and with torque_ref_nm (torque control) alike.
            One level per run; a ramp within a run is not modelled. "20 % load" is not a Scenario
            concept: the caller converts it with the motor's rated torque. None (default) leaves
            the default load untouched, bit for bit. Default load (round Assistant 0b, D1): the stock
            friction (a = 0.01, b = 0.05) for the stock motor; when motor_parameter_overrides is set
            AND dc_nominal_overrides gives 'torque' and 'omega', a friction scaled to that motor
            instead, a + b*omega_rated = 2 % of its rated torque (half constant, half viscous;
            runner.dc_default_load_parameter).
        dc_supply_v / dc_limit_overrides / dc_nominal_overrides: dc_perm_ex_v1 only (round
            Assistant, task F0.2). dc_supply_v is the DC-link voltage of the one-quadrant converter
            (IdealVoltageSupply; default 65 V). dc_limit_overrides and dc_nominal_overrides
            override the motor's limits (default omega=400 rad/s, torque=38 N*m, i=210 A, u=60 V)
            and nominal values (default omega=300, torque=16, i=97, u=60), keys among omega,
            torque, i, u, merged over the defaults. Limits normalise the state and set the
            controllers' current cap (PI: 0.8 x i, MPC: 0.85 x i); the simulator does not clip
            states with them. GEM keeps an explicit i limit as given (it derives u/r_a only for a
            missing or zero i), and it does NOT derive i from changed motor parameters: with
            motor_parameter_overrides alone the current limit stays at the stock 210 A. A u limit
            above dc_supply_v is rejected. None (default) for all three keeps the stock motor and
            the 65 V supply, bit for bit.
        plant_config_id="dc_pmg132_v1" (round Assistant 0b, D3, user decision 2026-11-02): the same
            DC-motor plant, controllers (PI, MPC), fault and vibration paths as dc_perm_ex_v1, with the
            motor of the Perm-Motor PMG 132 datasheet at 60 V (S-20): r_a 16 mOhm, l_a 0.019 mH,
            J 0.025 kg*m^2, psi_e 0.181 V*s/rad (P/(omega*I)); rated 110 A, 20 N*m, 2870 min^-1, 60 V;
            limits 200 A and 38 N*m (short time, S2 10 min). Default load: 2 % of the rated torque at
            rated speed (D1). Every dc_perm_ex_v1-only field above also applies to it; overrides are
            merged over the datasheet values. Not used by the diagnosis datasets, the Gym
            environments or the GUI.
        pmsm_motor_parameters: pmsm_v1 only (round Assistant, task F0.3) -- partial override of the
            PMSM merged over control/classical/pmsm_foc.MOTOR_PARAMS: p (pole pairs, int >= 1), l_d,
            l_q (H, dq inductances per phase), r_s (ohm, stator resistance per phase), psi_p (V*s,
            permanent-magnet flux linkage, peak per phase), j_rotor (kg*m^2, GEM default 0.03883)
            and u_supply (V, DC link, default 330). The merged dict reaches the plant, the dq
            current controller, the MTPA split and the reported torque. MTPA assumes l_q >= l_d
            (salient interior-magnet machine); l_d > l_q is accepted with a warning. The |i_s| <=
            400 A bound of pmsm_current_ref_a stays. None (default): the stock PMSM, bit for bit.
    """

    scenario_id: str
    controller_type: str = "PI"
    omega_ref_rad_s: float = 150.0
    fault_type: str = None
    electrical_severity: float = 8.0
    mechanical_severity: float = None
    seed: int = 0
    duration_s: float = 1.0
    plant_config_id: str = "dc_perm_ex_v1"
    motor_parameter_overrides: dict = None
    electrical_noise_pct: float = 0.0
    mechanical_noise_pct: float = 0.0
    torque_ref_nm: float = None
    fault_order_override: float = None
    extra_faults: list = None
    load_resistance_ohm: float = None
    reference_magnitude_v: float = None
    reference_omega_rad_s: float = None
    pi_setpoint_weight: float = None
    pmsm_current_ref_a: float = None
    vsc_voltage_limit_v: float = None
    vsc_capacitor_scale: float = None
    vsc_esr_ohm: float = None
    buck_vout_ref_v: float = None
    buck_vin_v: float = None
    buck_load_step_ohm: float = None
    buck_load_step_time_s: float = None
    buck_parameter_overrides: dict = None
    plecs_model_vars: dict = None
    plecs_time_limit_s: float = None
    plecs_output_points: int = None
    dc_plant_solver: str = None
    dc_plant_substeps: int = None
    load_torque_nm: float = None
    dc_supply_v: float = None
    dc_limit_overrides: dict = None
    dc_nominal_overrides: dict = None
    pmsm_motor_parameters: dict = None
    dc_converter_model: str = None

    _VALID_PAIRS = {("PI", "dc_perm_ex_v1"), ("MPC", "dc_perm_ex_v1"), ("PI", "dc_pmg132_v1"), ("MPC", "dc_pmg132_v1"), ("DPC", "vsc_dpc_v1"), ("FOC_MTPA", "pmsm_v1"), ("FOC_naive", "pmsm_v1"),
                    ("PI", "buck_v1")}
    _BUCK_PARAMETERS = ("l_h", "c_f", "r_l_ohm", "i_max_a")
    #: |i_s| limit of the PMSM (control/classical/pmsm_foc.py I_LIM).
    _PMSM_CURRENT_LIMIT_A = 400.0

    #: plant_config_id prefix of a PLECS batch profile (task E2); its controller lives in the circuit.
    PLECS_PREFIX = "plecs:"
    PLECS_CONTROLLER = "circuit"

    @property
    def is_plecs(self) -> bool:
        return isinstance(self.plant_config_id, str) and self.plant_config_id.startswith(self.PLECS_PREFIX)

    def __post_init__(self):
        if self.is_plecs or self.controller_type == self.PLECS_CONTROLLER:
            if not (self.is_plecs and self.controller_type == self.PLECS_CONTROLLER):
                raise NotImplementedError(f"PLECS plants ('{self.PLECS_PREFIX}<profile_id>') run with controller_type={self.PLECS_CONTROLLER!r} (the control lives inside the circuit) "
                                          f"and nothing else does; got ({self.controller_type!r}, {self.plant_config_id!r})")
        elif (self.controller_type, self.plant_config_id) not in self._VALID_PAIRS:
            raise NotImplementedError(
                f"(controller_type={self.controller_type!r}, plant_config_id={self.plant_config_id!r}) "
                f"is not a supported pairing -- valid pairs are {sorted(self._VALID_PAIRS)}. "
                "PI and MPC both target the DC motor plant (dc_perm_ex_v1); DPC targets a "
                "separate VSC plant (vsc_dpc_v1) -- see runner.py's module docstring."
            )
        if self.mechanical_severity is None:
            default = CALIBRATED_MECHANICAL_SEVERITY.get(self.fault_type, 0.0)
            object.__setattr__(self, "mechanical_severity", default)  # frozen dataclass
        self._validate_numeric_fields()

    #: Motor parameters that must be strictly positive for the DC model to be physical (a zero
    #: l_a divides by zero in the electrical ODE; a negative r_a or j_rotor makes the plant
    #: unstable/meaningless -- measured: r_a < 0 drove the motor to 1.5e17 rpm with no error).
    _POSITIVE_MOTOR_PARAMETERS = ("r_a", "l_a", "psi_e", "j_rotor")

    def _validate_numeric_fields(self):
        """Reject inputs that cannot describe a physical run, instead of silently returning an
        empty record list or NaN/inf trajectories (docs/SIMULATION_STABILITY_REPORT.md, S-03).
        Physically legal but unstable operating points (e.g. VSC load R below R* ~= 3.37 ohm, or a
        speed reference the one-quadrant converter cannot reach) are NOT rejected: those are
        real behaviours the platform exists to show."""

        def finite(name, value, minimum=None, strict=False):
            if value is None:
                return
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"Scenario.{name} must be a finite number, got {value!r}")
            if minimum is not None and (value <= minimum if strict else value < minimum):
                raise ValueError(f"Scenario.{name} must be {'>' if strict else '>='} {minimum}, got {value!r}")

        finite("duration_s", self.duration_s, 0.0, strict=True)
        finite("omega_ref_rad_s", self.omega_ref_rad_s)
        finite("electrical_severity", self.electrical_severity, 0.0)
        finite("mechanical_severity", self.mechanical_severity, 0.0)
        finite("electrical_noise_pct", self.electrical_noise_pct, 0.0)
        finite("mechanical_noise_pct", self.mechanical_noise_pct, 0.0)
        finite("torque_ref_nm", self.torque_ref_nm)
        finite("fault_order_override", self.fault_order_override, 0.0, strict=True)
        finite("load_resistance_ohm", self.load_resistance_ohm, 0.0, strict=True)
        finite("reference_magnitude_v", self.reference_magnitude_v, 0.0)
        finite("reference_omega_rad_s", self.reference_omega_rad_s)
        finite("pi_setpoint_weight", self.pi_setpoint_weight, 0.0)
        finite("pmsm_current_ref_a", self.pmsm_current_ref_a)
        finite("vsc_voltage_limit_v", self.vsc_voltage_limit_v, 0.0, strict=True)
        finite("vsc_capacitor_scale", self.vsc_capacitor_scale, 0.0, strict=True)
        finite("vsc_esr_ohm", self.vsc_esr_ohm, 0.0)
        finite("buck_vout_ref_v", self.buck_vout_ref_v, 0.0)
        finite("buck_vin_v", self.buck_vin_v, 0.0, strict=True)
        finite("buck_load_step_ohm", self.buck_load_step_ohm, 0.0, strict=True)
        finite("buck_load_step_time_s", self.buck_load_step_time_s, 0.0)
        for key, value in (self.buck_parameter_overrides or {}).items():
            if key not in self._BUCK_PARAMETERS:
                raise ValueError(f"Scenario.buck_parameter_overrides: unknown parameter {key!r}; expected one of {self._BUCK_PARAMETERS}")
            finite(f"buck_parameter_overrides[{key!r}]", value, 0.0, strict=key != "r_l_ohm")
        plecs_fields = {"plecs_model_vars": self.plecs_model_vars, "plecs_time_limit_s": self.plecs_time_limit_s, "plecs_output_points": self.plecs_output_points}
        if self.is_plecs:
            from driveflow.integrations.plecs.batch import check_profile_id

            check_profile_id(self.plant_config_id[len(self.PLECS_PREFIX):])
            finite("plecs_time_limit_s", self.plecs_time_limit_s, 0.0, strict=True)
            if self.plecs_output_points is not None and (isinstance(self.plecs_output_points, bool) or not isinstance(self.plecs_output_points, int) or self.plecs_output_points < 2):
                raise ValueError(f"Scenario.plecs_output_points must be an int >= 2, got {self.plecs_output_points!r}")
            for key, value in (self.plecs_model_vars or {}).items():
                finite(f"plecs_model_vars[{key!r}]", value)
        elif any(v is not None for v in plecs_fields.values()):
            raise ValueError(f"Scenario.{next(k for k, v in plecs_fields.items() if v is not None)} applies only to PLECS plants ('{self.PLECS_PREFIX}<profile_id>')")
        if self.plant_config_id == "buck_v1":
            vin = self.buck_vin_v if self.buck_vin_v is not None else 48.0
            vref = self.buck_vout_ref_v if self.buck_vout_ref_v is not None else 24.0
            if vref >= vin:
                raise ValueError(f"Scenario.buck_vout_ref_v must be below V_in ({vin:g} V) for a buck converter, got {vref!r}")
        if self.plant_config_id == "pmsm_v1":
            if self.pmsm_current_ref_a is None:
                raise ValueError("Scenario.pmsm_current_ref_a is required for plant_config_id='pmsm_v1'")
            if abs(self.pmsm_current_ref_a) > self._PMSM_CURRENT_LIMIT_A:
                raise ValueError(f"Scenario.pmsm_current_ref_a must satisfy |i_s| <= {self._PMSM_CURRENT_LIMIT_A} A, got {self.pmsm_current_ref_a!r}")
        if self.pi_setpoint_weight is not None and self.pi_setpoint_weight > 1.0:
            raise ValueError(f"Scenario.pi_setpoint_weight must be <= 1, got {self.pi_setpoint_weight!r}")
        for key, value in (self.motor_parameter_overrides or {}).items():
            finite(f"motor_parameter_overrides[{key!r}]", value, 0.0, strict=key in self._POSITIVE_MOTOR_PARAMETERS)
        for i, (_ft, order) in enumerate(self.extra_faults or []):
            finite(f"extra_faults[{i}] order_override", order, 0.0, strict=True)
        self._validate_dc_plant_integration()
        self._validate_dc_motor_fields()
        self._validate_pmsm_motor_parameters()

    #: Keys of pmsm_motor_parameters (pmsm_foc.MOTOR_PARAMS, plus j_rotor and u_supply).
    PMSM_PARAMETER_KEYS = ("p", "l_d", "l_q", "r_s", "psi_p", "j_rotor", "u_supply")

    def _validate_pmsm_motor_parameters(self):
        mp = self.pmsm_motor_parameters
        if mp is None:
            return
        if self.plant_config_id != "pmsm_v1":
            raise ValueError(f"Scenario.pmsm_motor_parameters applies only to plant_config_id='pmsm_v1', got {self.plant_config_id!r}")
        if not isinstance(mp, dict):
            raise ValueError(f"Scenario.pmsm_motor_parameters must be a dict with keys among {self.PMSM_PARAMETER_KEYS}, got {mp!r}")
        for key, value in mp.items():
            if key not in self.PMSM_PARAMETER_KEYS:
                raise ValueError(f"Scenario.pmsm_motor_parameters: unknown key {key!r}; expected one of {self.PMSM_PARAMETER_KEYS}")
            if key == "p":
                if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                    raise ValueError(f"Scenario.pmsm_motor_parameters['p'] (pole pairs) must be an int >= 1, got {value!r}")
            elif isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Scenario.pmsm_motor_parameters[{key!r}] must be a finite number > 0, got {value!r}")
        from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS

        merged = {**MOTOR_PARAMS, **mp}
        if merged["l_d"] > merged["l_q"]:
            import warnings

            warnings.warn(f"Scenario.pmsm_motor_parameters: l_d ({merged['l_d']:g} H) > l_q ({merged['l_q']:g} H). The MTPA policy assumes l_q >= l_d "
                          "(salient interior-magnet machine) and may not be meaningful for this motor.", UserWarning, stacklevel=3)

    #: The DC motor plants: GEM's port (dc_perm_ex_v1) and the PMG 132 datasheet motor (dc_pmg132_v1, round Assistant 0b D3).
    DC_PLANTS = ("dc_perm_ex_v1", "dc_pmg132_v1")
    #: Fields of round Assistant phase 0 that only the DC motor plants accept.
    _DC_MOTOR_ONLY_FIELDS = ("load_torque_nm", "dc_supply_v", "dc_limit_overrides", "dc_nominal_overrides")
    #: Keys of dc_limit_overrides / dc_nominal_overrides (DcPermanentlyExcitedMotor limits).
    DC_LIMIT_KEYS = ("omega", "torque", "i", "u")

    def _validate_dc_motor_fields(self):
        """Round Assistant, phase 0: optional DC-motor fields. Each must be finite and in range,
        and is rejected on any other plant (same rule as the VSC-only fields)."""
        used = [name for name in self._DC_MOTOR_ONLY_FIELDS if getattr(self, name) is not None]
        if not used:
            return
        if self.plant_config_id not in self.DC_PLANTS:
            raise ValueError(f"Scenario.{used[0]} applies only to plant_config_id='dc_perm_ex_v1' or 'dc_pmg132_v1' (the DC motor plants), got {self.plant_config_id!r}")

        def real(value) -> bool:
            return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)

        value = self.load_torque_nm
        if value is not None and (not real(value) or value < 0):
            raise ValueError(f"Scenario.load_torque_nm must be a finite number >= 0 (N*m), got {value!r}")
        if self.dc_supply_v is not None and (not real(self.dc_supply_v) or self.dc_supply_v <= 0):
            raise ValueError(f"Scenario.dc_supply_v must be a finite number > 0 (V), got {self.dc_supply_v!r}")
        for name in ("dc_limit_overrides", "dc_nominal_overrides"):
            overrides = getattr(self, name)
            if overrides is None:
                continue
            if not isinstance(overrides, dict):
                raise ValueError(f"Scenario.{name} must be a dict with keys among {self.DC_LIMIT_KEYS}, got {overrides!r}")
            for key, v in overrides.items():
                if key not in self.DC_LIMIT_KEYS:
                    raise ValueError(f"Scenario.{name}: unknown key {key!r}; expected one of {self.DC_LIMIT_KEYS}")
                if not real(v) or v <= 0:
                    raise ValueError(f"Scenario.{name}[{key!r}] must be a finite number > 0, got {v!r}")
        u_limit = (self.dc_limit_overrides or {}).get("u")
        if u_limit is not None and self.dc_supply_v is not None and u_limit > self.dc_supply_v:
            raise ValueError(f"Scenario.dc_limit_overrides['u'] ({u_limit!r} V) must not exceed dc_supply_v ({self.dc_supply_v!r} V)")

    #: dc_plant_solver values (see the class docstring); DC_PLANT_MAX_SUBSTEPS bounds the cost.
    DC_PLANT_SOLVERS = ("euler", "rk45")
    DC_PLANT_MAX_SUBSTEPS = 10_000

    #: dc_converter_model values (S-21, decision S21-2): "gem" = GEM's averaged one-quadrant converter
    #: (the default, None); "dcm" = discontinuous conduction, re-evaluated inside the period (sim/dc_dcm.py).
    DC_CONVERTER_MODELS = ("gem", "dcm")

    def _validate_dc_plant_integration(self):
        solver, n = self.dc_plant_solver, self.dc_plant_substeps
        model = self.dc_converter_model
        if model is not None:
            if self.plant_config_id not in self.DC_PLANTS:
                raise ValueError("Scenario.dc_converter_model applies only to plant_config_id='dc_perm_ex_v1' or 'dc_pmg132_v1' (the DC motor plants)")
            if model not in self.DC_CONVERTER_MODELS:
                raise ValueError(f"Scenario.dc_converter_model must be one of {self.DC_CONVERTER_MODELS}, got {model!r}")
            if model == "dcm" and solver == "rk45":
                raise ValueError("Scenario.dc_converter_model='dcm' integrates with its own substeps (dc_plant_substeps); it does not combine with dc_plant_solver='rk45'")
        if solver is None and n is None:
            return
        if self.plant_config_id not in self.DC_PLANTS:
            raise ValueError("Scenario.dc_plant_solver / dc_plant_substeps apply only to plant_config_id='dc_perm_ex_v1' or 'dc_pmg132_v1' (the DC motor plants)")
        if solver is not None and solver not in self.DC_PLANT_SOLVERS:
            raise ValueError(f"Scenario.dc_plant_solver must be one of {self.DC_PLANT_SOLVERS}, got {solver!r}")
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, numbers.Integral) or not 1 <= n <= self.DC_PLANT_MAX_SUBSTEPS:
                raise ValueError(f"Scenario.dc_plant_substeps must be an int in [1, {self.DC_PLANT_MAX_SUBSTEPS}], got {n!r}")
            if solver == "rk45":
                raise ValueError("Scenario.dc_plant_substeps applies only to dc_plant_solver='euler' (RK45 chooses its own steps)")

    @property
    def label(self) -> str:
        names = ([self.fault_type] if self.fault_type is not None else []) + [ft for ft, _ in (self.extra_faults or [])]
        names = list(dict.fromkeys(names))  # de-dupe while preserving order -- a saved compound
        # custom fault type (dashboard's "Custom fault types") reuses its own single name for
        # every one of its component orders when split into (primary, extra_faults) at generate
        # time, so this collapses "combo+combo+combo" back into the one clean name.
        return "+".join(names) if names else "normal"


def healthy_and_faulted_grid(
    base_scenario_id: str,
    fault_types=("outer_race", "inner_race"),
    electrical_severity: float = 8.0,
    mechanical_severity: float = None,
    seeds=(0,),
    **scenario_kwargs,
) -> list:
    """Builds the "normal + at least 2 fault types" grid A.5's closing criterion asks for, varying
    seed. One Scenario per (condition, seed) combination.

    Args:
        mechanical_severity: If None (default), each fault_type uses its own
            CALIBRATED_MECHANICAL_SEVERITY value instead of one shared number.
    """
    scenarios = []
    for seed in seeds:
        scenarios.append(Scenario(scenario_id=f"{base_scenario_id}_normal_seed{seed}", fault_type=None, seed=seed, **scenario_kwargs))
        for fault_type in fault_types:
            scenarios.append(
                Scenario(
                    scenario_id=f"{base_scenario_id}_{fault_type}_seed{seed}",
                    fault_type=fault_type,
                    electrical_severity=electrical_severity,
                    mechanical_severity=mechanical_severity,
                    seed=seed,
                    **scenario_kwargs,
                )
            )
    return scenarios
