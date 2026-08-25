"""Scenario definitions: one Scenario = one simulation run's parameters (load, controller, seed,
fault type -- per INSTRUCTIONS.md A.4). Kept deliberately small/explicit rather than a generic
config-object soup, since Macro-fase A only needs to vary a handful of things (Sec. A.4).
"""

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
        controller_type: "PI" (Macro-fase A, against plant_config_id="dc_perm_ex_v1") or "DPC"
            (Macro-fase B, against plant_config_id="vsc_dpc_v1" -- a Voltage Source Converter,
            NOT the DC motor; see runner.py's module docstring on why these are two unrelated
            plants, not a controller swap on the same system). "MPC" is not implemented yet.
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

    _VALID_PAIRS = {("PI", "dc_perm_ex_v1"), ("DPC", "vsc_dpc_v1")}

    def __post_init__(self):
        if (self.controller_type, self.plant_config_id) not in self._VALID_PAIRS:
            raise NotImplementedError(
                f"(controller_type={self.controller_type!r}, plant_config_id={self.plant_config_id!r}) "
                f"is not a supported pairing -- valid pairs are {sorted(self._VALID_PAIRS)}. "
                "Each controller_type here targets exactly one plant (DPC controls a VSC, not "
                "the DC motor PI/MPC target -- see runner.py's module docstring)."
            )
        if self.mechanical_severity is None:
            default = CALIBRATED_MECHANICAL_SEVERITY.get(self.fault_type, 0.0)
            object.__setattr__(self, "mechanical_severity", default)  # frozen dataclass

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
