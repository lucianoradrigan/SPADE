"""GUI boundary to the DC-DC plants (expansion task E6; spec Sec. 8.1). Buck only: the DAB tab
arrives with E9. The page passes plain parameters; the Scenario, the run and the metrics are built
here from datagen.runner and validation.step_metrics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from driveflow.datagen.export_parquet import records_to_dataframe
from driveflow.datagen.runner import BUCK_DEFAULT_VOUT_REF_V, run_scenario
from driveflow.datagen.scenario import Scenario
from driveflow.sim.buck_system import TAU as BUCK_TAU, BuckParams
from driveflow.validation.step_metrics import step_response_metrics

#: page key -> BuckParams field, for the plant-parameter overrides
PLANT_FIELDS = ("l_h", "c_f", "r_l_ohm", "i_max_a")


def buck_defaults() -> dict:
    p = BuckParams()
    return {"vout_ref_v": BUCK_DEFAULT_VOUT_REF_V, "vin_v": p.v_in_v, "load_resistance_ohm": p.load_resistance_ohm, "duration_s": 0.02,
            "load_step_on": True, "load_step_ohm": 3.0, "load_step_time_s": 0.01, **{f: getattr(p, f) for f in PLANT_FIELDS}, "tau_s": BUCK_TAU}


def buck_scenario(params: dict, scenario_id: str = "dashboard_buck_run") -> Scenario:
    """Raises ValueError with the backend's message for an invalid combination."""
    d = BuckParams()
    overrides = {f: float(params[f]) for f in PLANT_FIELDS if f in params and float(params[f]) != getattr(d, f)}
    step = params.get("load_step_on")
    return Scenario(scenario_id=scenario_id, plant_config_id="buck_v1", controller_type="PI", duration_s=float(params["duration_s"]),
                    buck_vout_ref_v=float(params["vout_ref_v"]), buck_vin_v=float(params["vin_v"]),
                    load_resistance_ohm=float(params["load_resistance_ohm"]),
                    buck_load_step_ohm=float(params["load_step_ohm"]) if step else None,
                    buck_load_step_time_s=float(params["load_step_time_s"]) if step else None,
                    buck_parameter_overrides=overrides or None)


def run_buck(params: dict) -> pd.DataFrame:
    return records_to_dataframe(run_scenario(buck_scenario(params)))


def buck_metrics(df: pd.DataFrame, params: dict) -> dict:
    """Start-up step metrics (0 V -> v_ref), the load-step response when there is one, and the
    steady-state operating point. Values are None when not defined (e.g. never settled)."""
    t, v, ref = df["timestamp_s"].to_numpy(), df["buck_v_out"].to_numpy(), float(params["vout_ref_v"])
    t_step = float(params["load_step_time_s"]) if params.get("load_step_on") else None
    startup_mask = t < t_step if t_step is not None else np.ones_like(t, dtype=bool)
    m = step_response_metrics(t[startup_mask], v[startup_mask], ref, 0.0)
    p = BuckParams(**{f: float(params[f]) for f in PLANT_FIELDS}, v_in_v=float(params["vin_v"]), load_resistance_ohm=float(params["load_resistance_ohm"]))
    out = {"settling_time_ms": None if m.settling_time_s is None else m.settling_time_s * 1e3, "overshoot_pct": m.overshoot_pct,
           "steady_state_error_mv": m.steady_state_error * 1e3, "finite": m.finite, "resonance_hz": p.resonance_hz,
           "final_v_out": float(v[-1]), "final_i_l": float(df["buck_i_l"].iloc[-1]), "final_duty": float(df["buck_duty"].iloc[-1]),
           "max_i_l": float(df["buck_i_l"].abs().max()), "load_dip_v": None, "load_recovery_ms": None}
    if t_step is not None and (t >= t_step).any():
        after = t >= t_step
        va, ta = v[after], t[after] - t_step
        out["load_dip_v"] = float(np.max(np.abs(va - ref)))
        outside = np.flatnonzero(np.abs(va - ref) > 0.01 * ref)
        out["load_recovery_ms"] = 0.0 if not outside.size else (None if outside[-1] == len(va) - 1 else float(ta[outside[-1] + 1]) * 1e3)
    return out


def buck_parameters_table(params: dict) -> pd.DataFrame:
    p = BuckParams(**{f: float(params[f]) for f in PLANT_FIELDS}, v_in_v=float(params["vin_v"]), load_resistance_ohm=float(params["load_resistance_ohm"]))
    shown = {"v_in_v": ("Input voltage V_in", 1.0, "V"), "l_h": ("Inductance L", 1e6, "µH"), "c_f": ("Capacitance C", 1e6, "µF"),
             "r_l_ohm": ("Inductor resistance r_L", 1.0, "Ω"), "load_resistance_ohm": ("Initial load R", 1.0, "Ω"), "i_max_a": ("Current rating i_max", 1.0, "A")}
    return pd.DataFrame([{"Parameter": shown[k][0], "Value": round(v * shown[k][1], 6), "Unit": shown[k][2]} for k, v in dataclasses.asdict(p).items()])
