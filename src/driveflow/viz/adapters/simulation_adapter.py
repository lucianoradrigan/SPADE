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


def run_dc(params: dict, custom_faults: dict, scenario_id: str = "dashboard_run") -> pd.DataFrame:
    fault_type, override, extras = resolve_fault(params["fault_label"], custom_faults)
    kwargs = dict(
        scenario_id=scenario_id, controller_type=params["controller"], fault_type=fault_type, duration_s=params["duration_s"], seed=int(params["seed"]),
        electrical_severity=params["electrical_severity"], mechanical_severity=params["mechanical_severity"],
        motor_parameter_overrides=params["motor"], electrical_noise_pct=params["electrical_noise_pct"], mechanical_noise_pct=params["mechanical_noise_pct"],
        fault_order_override=override, extra_faults=extras,
    )
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
