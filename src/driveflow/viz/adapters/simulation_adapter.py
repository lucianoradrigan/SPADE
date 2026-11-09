"""Thin orchestration over driveflow.datagen / control for the Simulation pages. No physics here:
it builds Scenario objects from UI parameters, calls run_scenario/run_flow/the FOC cloud generator,
and summarises results (steady-state means over the settled tail of the run)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import TAU
from driveflow.sim import DcPermanentlyExcitedMotor

FS_HZ = 1.0 / TAU
#: Fraction of a run treated as start-up transient when summarising steady-state values.
SETTLE_FRACTION = 0.2
#: Shown wherever vibration channels appear (AUDIT AI-07): only one accelerometer axis of Module B
#: was calibrated against real PSDs; y/z are documented scale assumptions (datagen/runner.py).
VIBRATION_PROVENANCE = "synthetic_approximate — acc_x calibrated against KAt-DataCenter PSDs (one axis); acc_y/acc_z use assumed scale factors, not measured"


#: Decision E3 (Luciano, 2026-11-08): the DC plant the GUI simulates -- v2, the converter in discontinuous
#: conduction (S-21 corrected). dc_perm_ex_v1 (GEM's converter) is frozen and not offered by the GUI.
DC_PLANT = "dc_perm_ex_v2"
#: The plant modes the DC Motor page offers (V1, round B): v2 by default, v1 as the GEM-faithful mode.
DC_PLANT_MODES = {
    "dc_perm_ex_v2": "v2 (default): converter in discontinuous conduction, S-21 corrected",
    "dc_perm_ex_v1": "v1: GEM-faithful mode (GEM's one-quadrant converter, S-21 open)",
}
#: Shown when v1 is chosen (S-21, docs/SIMULATION_STABILITY_REPORT.md).
S21_V1_WARNING = ("S-21: in the v1 plant, GEM's one-quadrant converter applies the full supply voltage for a whole control period "
                  "whenever the armature current is negative. When the speed controller brakes with a light load, the simulated speed "
                  "rises instead of coasting. Use v1 only to reproduce GEM-based results; v2 corrects it.")
#: The stock DC-link voltage of the DC plants (datagen/runner.py: IdealVoltageSupply(u_nominal=65.0)).
DC_SUPPLY_V = 65.0


def dc_defaults() -> dict:
    m = DcPermanentlyExcitedMotor()
    return {"params": dict(m.motor_parameter), "limits": dict(m.limits), "nominal": dict(m.nominal_values)}


def resolve_fault(fault_label: str, custom_faults: dict) -> tuple:
    """fault label -> (fault_type, fault_order_override, extra_faults), same resolution as the
    legacy dashboard (a saved custom fault may combine several characteristic orders)."""
    fault_type = None if fault_label == "healthy" else fault_label
    if fault_label in custom_faults:
        orders = custom_faults[fault_label]["orders"]
        return fault_type, orders[0], [(fault_label, o) for o in orders[1:]]
    return fault_type, None, []


def dc_phase0_kwargs(params: dict) -> dict:
    """The Phase 0 options of the DC Motor page as Scenario keywords. A control left at its stock value
    is not sent, so the defaults reproduce the run of the page without these options exactly."""
    defaults = dc_defaults()
    out = {}
    if params.get("load_torque_nm"):
        out["load_torque_nm"] = float(params["load_torque_nm"])
    if params.get("dc_supply_v") is not None and float(params["dc_supply_v"]) != DC_SUPPLY_V:
        out["dc_supply_v"] = float(params["dc_supply_v"])
    for name, stock in (("dc_limit_overrides", defaults["limits"]), ("dc_nominal_overrides", defaults["nominal"])):
        changed = {k: float(v) for k, v in (params.get(name) or {}).items() if float(v) != float(stock[k])}
        if changed:
            out[name] = changed
    return out


def run_dc(params: dict, custom_faults: dict, scenario_id: str = "dashboard_run") -> pd.DataFrame:
    """Raises ValueError with Scenario's own message when an option is not valid."""
    fault_type, override, extras = resolve_fault(params["fault_label"], custom_faults)
    kwargs = dict(
        scenario_id=scenario_id, plant_config_id=params.get("plant", DC_PLANT), controller_type=params["controller"], fault_type=fault_type, duration_s=params["duration_s"], seed=int(params["seed"]),
        electrical_severity=params["electrical_severity"], mechanical_severity=params["mechanical_severity"],
        motor_parameter_overrides=params["motor"], electrical_noise_pct=params["electrical_noise_pct"], mechanical_noise_pct=params["mechanical_noise_pct"],
        fault_order_override=override, extra_faults=extras,
    )
    kwargs.update(dc_phase0_kwargs(params))
    if params.get("pi_setpoint_weight") is not None and params["controller"] == "PI":
        kwargs["pi_setpoint_weight"] = params["pi_setpoint_weight"]
    if params["control_mode"] == "speed":
        kwargs["omega_ref_rad_s"] = params["setpoint"]
    else:
        kwargs["torque_ref_nm"] = params["setpoint"]
    return pd.DataFrame.from_records(run_scenario(Scenario(**kwargs)))


def steady_state(df: pd.DataFrame, column: str, absolute: bool = False) -> float | None:
    if column not in df or df[column].isna().all():
        return None
    tail = df[column].iloc[int(len(df) * SETTLE_FRACTION):]
    tail = tail.abs() if absolute else tail
    return float(tail.mean())


def vibration_rms(df: pd.DataFrame, column: str = "acc_x") -> float | None:
    if column not in df or df[column].isna().all():
        return None
    tail = df[column].iloc[int(len(df) * SETTLE_FRACTION):]
    return float(np.sqrt(np.mean((tail - tail.mean()) ** 2)))


def fault_markers(df: pd.DataFrame) -> dict:
    """Characteristic frequencies computed by the simulator itself for this run (last sample's
    speed) plus the shaft rotation frequency -- labels for spectrum markers."""
    last = df.iloc[-1]
    markers = {}
    rpm = steady_state(df, "rpm")
    if rpm and rpm > 0:
        markers["f_shaft"] = rpm / 60.0
    for col, label in (("bpfo_hz", "BPFO"), ("bpfi_hz", "BPFI"), ("bsf_hz", "BSF"), ("ftf_hz", "FTF")):
        v = last.get(col)
        if v is not None and np.isfinite(v) and v > 0:
            markers[label] = float(v)
    return markers
