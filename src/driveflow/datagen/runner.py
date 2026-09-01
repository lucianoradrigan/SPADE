"""Orchestrates one Scenario end-to-end -> a list of per-timestep records matching
export_parquet.py's schema. Two unrelated plant/controller pairs, dispatched by
scenario.plant_config_id (see scenario.py's docstring for the valid pairings):

- "dc_perm_ex_v1" / PICascadeController or MpcController (scenario.controller_type: "PI" or
  "MPC" -- see _DC_MOTOR_CONTROLLERS below and docs/patch10_implementacion_mpc.md): SCMLSystem +
  {PI,MPC} + BearingFaultLoad (electrical/MCSA fault path) + VibrationSynthesizer (mechanical
  fault path). Mirrors the loop sketched in docs/addendum_vibracion_v1.md Sec. 4, updated per Patch 3
  (docs/patch3_mejora_modulo_B.md): VibrationSynthesizer.step() no longer takes i_d/i_q/torque --
  Module B's excitation was found to have no exploitable dependence on those signals::

      for k in range(n_steps):
          action = controller.control(state, reference)
          state = scml_system.simulate(action)
          omega = extract_omega(state, scml_system)
          acc_x, acc_y, acc_z = vibration_synth.step(omega)
          record = {**state_to_dict(state), "acc_x": acc_x, ...}

- "vsc_dpc_v1" / DpcController: VscSystem + DpcController + RotatingReference (Macro-fase B.2,
  docs/macro_fase_B2_dpc_deployment.md). NOT the DC motor, no vibration/bearing physics -- a
  Voltage Source Converter has no motor, no mechanical side, nothing for Module B to attach to.
  This is why it is a genuinely separate code path here (`_run_vsc_scenario`), not a second
  controller branch inside the DC-motor loop above: the two loop bodies don't share a state
  representation, a controller interface, or which physical quantities are meaningful to record.
  Columns that don't apply to a VSC (rpm, torque_nm, acc_x/y/z, bpfo_hz, ...) are NaN in its
  records, same pattern as current_s/current_t already being NaN for the DC motor's single
  current channel.
"""

from pathlib import Path

import numpy as np
import yaml

from driveflow.control.classical import PICascadeController
from driveflow.control.dpc.controller import DpcController
from driveflow.control.mpc import MpcController
from driveflow.control.dpc.reference import RotatingReference
from driveflow.datagen.fault_injection import BearingFaultLoad
from driveflow.datagen.scenario import Scenario
from driveflow.sim import (
    ContOneQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PolynomialStaticLoad,
)
from driveflow.sim.vibration import (
    KAT_DATACENTER_6203_GEOMETRY,
    Mode,
    ModalAxis,
    ModalVibrationModel,
    VibrationSynthesizer,
)
from driveflow.sim.vibration import bearing_frequencies as bf
from driveflow.sim.vsc_system import VscSystem

TAU = 1e-4

#: Best available DPC checkpoint (Macro-fase B.2's second fine-tuning round -- see
#: docs/macro_fase_B2_dpc_deployment.md): closed-loop RMSE 1.18V, ~0.4deg phase lag, and back to
#: 100% holdout success rate. Not configurable per-Scenario (would need a new field + validation
#: for a path that, in practice, only ever has one sane value right now).
_DPC_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "dpc_trained_v3_closed_loop.weights.h5"
#: Matches the constant R found in Data4train.mat's holdout split -- see
#: docs/macro_fase_B1_dpc.md. Not a Scenario field yet: no data exists to validate any other R.
_VSC_R_OHM = 8.0064

_CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "configs" / "vibration_module_b.yaml"
#: Used only if configs/vibration_module_b.yaml has not been generated yet (see
#: experiments/calibrate_module_b.py) -- coarse, uncalibrated placeholder so the pipeline is still
#: runnable end-to-end without the Paderborn dataset.
_FALLBACK_MODES = [Mode(natural_freq_hz=800.0, damping_ratio=0.03), Mode(natural_freq_hz=2400.0, damping_ratio=0.03)]
_FALLBACK_BACKGROUND_GAIN = 0.05

#: KAt-DataCenter records a SINGLE accelerometer axis per run (vibration_1, see calibration.py's
#: module docstring), so only one axis of Module B was actually calibrated against real PSDs.
#: These per-axis scale factors are NOT measured -- they are a documented assumption about
#: typical radial/axial stiffness asymmetry in a ball-bearing housing (x = the calibrated/
#: measured radial direction; y = the orthogonal radial direction, similar order of stiffness but
#: not identical; z = axial, much lower vibration energy for a radial ball bearing under
#: primarily radial load). Without them all three axes would be literal copies of each other,
#: which is not how a real 3-axis accelerometer reads even along uncalibrated axes.
_AXIS_PROFILE = {
    "x": {"gain_scale": 1.00, "freq_scale": 1.00, "damping_scale": 1.00},
    "y": {"gain_scale": 0.80, "freq_scale": 1.05, "damping_scale": 1.15},
    "z": {"gain_scale": 0.35, "freq_scale": 0.92, "damping_scale": 0.85},
}


def _load_module_b_calibration():
    if _CALIBRATION_PATH.exists():
        with open(_CALIBRATION_PATH) as f:
            data = yaml.safe_load(f)
        modes = [Mode(**m) for m in data["modes"]]
        background_gain = data.get("background_noise_gain", _FALLBACK_BACKGROUND_GAIN)
        return modes, background_gain
    return _FALLBACK_MODES, _FALLBACK_BACKGROUND_GAIN


def build_plant(scenario: Scenario) -> DcMotorSystem:
    """Builds the SCMLSystem for scenario.plant_config_id. Macro-fase A only defines one plant
    (a DC permanently-excited motor); more plant configs (Macro-fase B's VSC+LC, other motor
    types) would extend this dispatch, not replace it.
    """
    if scenario.plant_config_id != "dc_perm_ex_v1":
        raise NotImplementedError(f"Unknown plant_config_id {scenario.plant_config_id!r}")

    base_load = PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025))
    extra_electrical = [(ft, scenario.electrical_severity, oo) for ft, oo in (scenario.extra_faults or [])]
    load = BearingFaultLoad(
        base_load,
        fault_type=scenario.fault_type,
        geometry=KAT_DATACENTER_6203_GEOMETRY,
        severity=scenario.electrical_severity,
        order_override=scenario.fault_order_override,
        extra_faults=extra_electrical,
    )
    return DcMotorSystem(
        converter=ContOneQuadrantConverter(),
        motor=DcPermanentlyExcitedMotor(motor_parameter=scenario.motor_parameter_overrides),
        load=load,
        supply=IdealVoltageSupply(u_nominal=65.0),  # motor is rated u=60V, see DcPermanentlyExcitedMotor
        ode_solver=EulerSolver(),
        tau=TAU,
    )


def build_vibration_synthesizer(scenario: Scenario) -> VibrationSynthesizer:
    modes, background_gain = _load_module_b_calibration()
    axes = {
        axis: ModalAxis(
            [
                Mode(
                    natural_freq_hz=m.natural_freq_hz * profile["freq_scale"],
                    damping_ratio=min(0.95, m.damping_ratio * profile["damping_scale"]),
                    gain=m.gain * profile["gain_scale"],
                )
                for m in modes
            ]
        )
        for axis, profile in _AXIS_PROFILE.items()
    }
    module_b = ModalVibrationModel(
        axes,
        KAT_DATACENTER_6203_GEOMETRY,
        n_substeps=10,
        background_gain=background_gain,
        seed=scenario.seed,
    )
    synth = VibrationSynthesizer(module_b, tau=TAU)
    extra_mechanical = [(ft, scenario.mechanical_severity, oo) for ft, oo in (scenario.extra_faults or [])]
    synth.set_fault(scenario.fault_type, scenario.mechanical_severity, order_override=scenario.fault_order_override, extra_faults=extra_mechanical)
    return synth


_DC_MOTOR_CONTROLLERS = {"PI": PICascadeController, "MPC": MpcController}


def _run_dc_motor_scenario(scenario: Scenario) -> list:
    system = build_plant(scenario)
    controller = _DC_MOTOR_CONTROLLERS[scenario.controller_type](system)
    vibration_synth = build_vibration_synthesizer(scenario)

    system.seed(np.random.SeedSequence(scenario.seed))
    state = system.reset()
    controller.reset()
    # explicit seed: BackgroundNoiseGenerator.reset(seed=None) would otherwise draw a fresh,
    # non-reproducible sequence, undoing the seeding ModalVibrationModel already got at
    # construction (see build_vibration_synthesizer)
    vibration_synth.reset(seed=scenario.seed)

    n_steps = int(round(scenario.duration_s / TAU))
    # True whenever ANY fault is active -- the primary (fault_type) or any of extra_faults (e.g.
    # dashboard's "healthy" primary with additional fault types combined in on top of it).
    any_fault_active = scenario.fault_type is not None or bool(scenario.extra_faults)
    records = []
    for k in range(n_steps):
        if scenario.torque_ref_nm is not None:
            action = controller.control_torque(state, scenario.torque_ref_nm)
        else:
            action = controller.control(state, scenario.omega_ref_rad_s)
        state = system.simulate(action)
        physical = state * system.limits

        omega = float(physical[system.OMEGA_IDX])
        torque = float(physical[system.TORQUE_IDX])
        current = float(physical[system.CURRENTS_IDX[0]])
        voltage = float(physical[system.VOLTAGES_IDX[0]])

        acc_x, acc_y, acc_z = vibration_synth.step(omega)

        if any_fault_active:
            fault_freqs = bf.fault_frequencies_hz(omega / (2 * np.pi), KAT_DATACENTER_6203_GEOMETRY)
        else:
            fault_freqs = {"outer_race": np.nan, "inner_race": np.nan, "ball": np.nan, "cage": np.nan}

        records.append(
            {
                "timestamp_s": (k + 1) * TAU,
                "current_r": current,
                "current_s": np.nan,  # DcMotorSystem has a single current channel -- see runner.py's docstring
                "current_t": np.nan,
                "acc_x": float(acc_x),
                "acc_y": float(acc_y),
                "acc_z": float(acc_z),
                "vibration_source": vibration_synth.vibration_source,
                "audio": np.nan,  # out of scope, per the addendum's schema (Sec. 6)
                "rpm": omega * 60.0 / (2 * np.pi),
                "torque_nm": torque,
                "voltage_v": voltage,
                "label": scenario.label,
                "electrical_severity_nm": scenario.electrical_severity if any_fault_active else 0.0,
                "mechanical_severity": scenario.mechanical_severity if any_fault_active else 0.0,
                "dataset_id": scenario.scenario_id,
                "source_file": scenario.scenario_id,
                "fs_hz": 1.0 / TAU,
                "controller_type": scenario.controller_type,
                "plant_config_id": scenario.plant_config_id,
                "bpfo_hz": fault_freqs["outer_race"],
                "bpfi_hz": fault_freqs["inner_race"],
                "bsf_hz": fault_freqs["ball"],
                "ftf_hz": fault_freqs["cage"],
                "seed": scenario.seed,
                # VSC-only columns (see _run_vsc_scenario) -- not applicable to a DC motor.
                "v_ref_real": np.nan,
                "v_ref_imag": np.nan,
                "vc_real": np.nan,
                "vc_imag": np.nan,
                "i_f_real": np.nan,
                "i_f_imag": np.nan,
            }
        )

    _apply_white_noise(records, scenario)
    return records


def _apply_white_noise(records: list, scenario: Scenario) -> None:
    """Adds white Gaussian noise to current_r (electrical_noise_pct) and/or acc_x/acc_y/acc_z
    (mechanical_noise_pct) in place, sized as a percentage of each signal's own std dev over the
    run -- a percentage of a fixed physical constant would either swamp a near-zero-amplitude
    healthy signal or be invisible on a saturated one, so each column's own spread is the only
    scale-free reference available. A no-op at the (default) 0% -- existing callers/tests that
    never set these fields see byte-identical output. Uses a dedicated RNG (not
    scenario.seed's SCMLSystem/ModalVibrationModel seeding above) so turning noise on/off doesn't
    change the underlying simulated trajectory, only what's added on top of it.
    """
    if scenario.electrical_noise_pct <= 0 and scenario.mechanical_noise_pct <= 0:
        return
    rng = np.random.default_rng(scenario.seed)
    if scenario.electrical_noise_pct > 0:
        values = np.array([r["current_r"] for r in records])
        std = values.std()
        noise = rng.normal(0.0, scenario.electrical_noise_pct / 100.0 * std, size=len(values))
        for r, n in zip(records, noise):
            r["current_r"] = float(r["current_r"] + n)
    if scenario.mechanical_noise_pct > 0:
        for axis in ("acc_x", "acc_y", "acc_z"):
            values = np.array([r[axis] for r in records])
            std = values.std()
            noise = rng.normal(0.0, scenario.mechanical_noise_pct / 100.0 * std, size=len(values))
            for r, n in zip(records, noise):
                r[axis] = float(r[axis] + n)


def _run_vsc_scenario(scenario: Scenario) -> list:
    r_ohm = scenario.load_resistance_ohm if scenario.load_resistance_ohm is not None else _VSC_R_OHM
    system = VscSystem(load_resistance_ohm=r_ohm, tau=TAU)
    controller = DpcController(_DPC_WEIGHTS_PATH, r_ohm=r_ohm)
    # No randomness in this plant/controller to seed (see scenario.py's docstring) -- seed instead
    # picks the reference's starting phase, so different seeds still give distinguishable runs.
    phase0 = (scenario.seed % 360) * (2 * np.pi / 360)
    # Only pass magnitude_v/omega_rad_s when explicitly set -- letting RotatingReference's own
    # dataclass defaults (50.0V, grid frequency) apply otherwise, rather than duplicating those
    # constants here.
    reference_kwargs = dict(tau=TAU, phase0_rad=phase0)
    if scenario.reference_magnitude_v is not None:
        reference_kwargs["magnitude_v"] = scenario.reference_magnitude_v
    if scenario.reference_omega_rad_s is not None:
        reference_kwargs["omega_rad_s"] = scenario.reference_omega_rad_s
    reference = RotatingReference(**reference_kwargs)

    state = system.reset()
    controller.reset()

    n_steps = int(round(scenario.duration_s / TAU))
    records = []
    for k in range(n_steps):
        v_o_real, v_o_imag = controller.control(state, reference, k)
        state = system.simulate(v_o_real, v_o_imag)
        vref_real, vref_imag = reference.at_step(k)

        records.append(
            {
                "timestamp_s": (k + 1) * TAU,
                "current_r": np.nan,  # DC-motor-only columns (see _run_dc_motor_scenario) -- a
                "current_s": np.nan,  # VSC has no abc/dq motor current, this plant's currents are
                "current_t": np.nan,  # i_f_real/i_f_imag below.
                "acc_x": np.nan,
                "acc_y": np.nan,
                "acc_z": np.nan,
                "vibration_source": None,
                "audio": np.nan,
                "rpm": np.nan,
                "torque_nm": np.nan,
                "voltage_v": np.nan,
                "label": "normal",  # no fault model exists for this plant yet
                "electrical_severity_nm": 0.0,
                "mechanical_severity": 0.0,
                "dataset_id": scenario.scenario_id,
                "source_file": scenario.scenario_id,
                "fs_hz": 1.0 / TAU,
                "controller_type": scenario.controller_type,
                "plant_config_id": scenario.plant_config_id,
                "bpfo_hz": np.nan,
                "bpfi_hz": np.nan,
                "bsf_hz": np.nan,
                "ftf_hz": np.nan,
                "seed": scenario.seed,
                "v_ref_real": vref_real,
                "v_ref_imag": vref_imag,
                "vc_real": state.vc_real,
                "vc_imag": state.vc_imag,
                "i_f_real": state.i_f_real,
                "i_f_imag": state.i_f_imag,
            }
        )
    return records


_SCENARIO_RUNNERS = {
    "dc_perm_ex_v1": _run_dc_motor_scenario,
    "vsc_dpc_v1": _run_vsc_scenario,
}


def run_scenario(scenario: Scenario) -> list:
    """Runs one Scenario to completion and returns its per-timestep records (list of dict).
    Dispatches on plant_config_id -- see this module's docstring for why the DC-motor and VSC
    paths are separate functions rather than branches inside one loop.
    """
    return _SCENARIO_RUNNERS[scenario.plant_config_id](scenario)


def run_scenarios(scenarios) -> list:
    """Runs several scenarios, returns a list of their record-lists (one per scenario)."""
    return [run_scenario(s) for s in scenarios]


def run_flow(scenarios: list) -> list:
    """Runs an ordered sequence of Scenarios back-to-back and concatenates them into ONE
    continuous timeline -- viz/dashboard.py's "Advanced Flow" mode, for exploring combinations of
    different operating states (control mode, setpoint, fault) run in a chosen order.

    Each segment is its OWN independent simulation: build_plant()/reset() runs fresh for every
    scenario in the list, not a continuation of the previous segment's exact instantaneous
    electromechanical state. "Flow" here means a sequence of different operating states shown
    together on one timeline, not one continuous ODE trajectory threaded across fault/reference
    changes -- true state-continuous segment transitions would need BearingFaultLoad to support a
    fixed, pre-allocated number of fault "slots" so its ODE state-vector size never changes
    mid-run (today it's sized from however many faults are active at construction, see
    datagen/fault_injection.py), which is a real change to that class, not made here.

    Args:
        scenarios: Ordered list of Scenario instances, one per segment. Each keeps its own
            duration_s/control mode/setpoint/fault/severity/noise; typically share the same
            motor_parameter_overrides and seed (the caller's choice, not enforced here) so at
            least the PLANT DEFINITION -- not its instantaneous state -- is consistent across
            segments.

    Returns:
        list(dict): All segments' records concatenated in order, each with timestamp_s offset by
        the cumulative duration of prior segments (so the whole flow reads as one timeline), plus
        two extra keys not part of export_parquet.py's schema (this is a dashboard-only view, not
        a dataset-generation path): "segment_index" (0-based position in `scenarios`) and
        "segment_label" (that segment's own Scenario.label).
    """
    all_records = []
    t_offset = 0.0
    for i, scenario in enumerate(scenarios):
        records = run_scenario(scenario)
        for r in records:
            r["timestamp_s"] = r["timestamp_s"] + t_offset
            r["segment_index"] = i
            r["segment_label"] = scenario.label
        all_records.extend(records)
        t_offset += scenario.duration_s
    return all_records
