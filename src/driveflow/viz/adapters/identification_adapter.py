"""GUI boundary to ai/pinn/identification.py (expansion task E10b; spec Sec. 8.3). The page passes
plain parameters and a DataFrame; windows, noise, the estimator and the validation numbers are
handled here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from driveflow.ai.pinn import identification as idn
from driveflow.datagen import Scenario, run_scenario

REPORT_PATH = Path(__file__).resolve().parents[4] / "docs" / "audit" / "ai" / "pinn_identification.json"
ESTIMATOR_LABELS = {"pinn": "PINN (Keras, physics residual)", "least_squares": "Least squares (exact ZOH model, reference)"}
PARAM_LABELS = {"L": ("Filter inductance L", 1e3, "mH"), "R_f": ("Filter resistance R_f", 1.0, "Ω"), "C": ("Filter capacitance C", 1e6, "µF")}


def simulate(c_scale: float, load_resistance_ohm: float, duration_s: float, seed: int = 0) -> pd.DataFrame:
    """A closed-loop VSC/DPC run (datagen/runner.py) with the capacitor at c_scale of nominal."""
    sc = Scenario(scenario_id="dashboard_pid_run", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=float(duration_s), seed=int(seed),
                  load_resistance_ohm=float(load_resistance_ohm), vsc_capacitor_scale=None if c_scale == 1.0 else float(c_scale))
    return pd.DataFrame.from_records(run_scenario(sc))


def true_parameters(c_scale: float) -> dict:
    return {"L": idn.NOMINAL["L"], "R_f": idn.NOMINAL["R_f"], "C": idn.NOMINAL["C"] * c_scale}


def run_identification(df: pd.DataFrame, load_resistance_ohm: float, start_s: float, length_s: float, estimator: str, n_bootstrap: int,
                       noise_pct: float = 0.0, seed: int = 0) -> dict:
    """Raises ValueError with a readable message for an unusable window or file."""
    data = idn.from_dataframe(df, load_resistance_ohm, start_s, length_s)
    data = idn.add_noise(data, noise_pct, np.random.default_rng(seed))
    return idn.identify(data, estimator, n_bootstrap=n_bootstrap, seed=seed)


def results_table(result: dict, truth: dict | None = None) -> pd.DataFrame:
    rows = []
    for p, (label, k, unit) in PARAM_LABELS.items():
        r = result["params"][p]
        row = {"Parameter": label, "Unit": unit, "Nominal": r["nominal"] * k, "Estimate": r["estimate"] * k,
               "Interval low (5 %)": r["low"] * k, "Interval high (95 %)": r["high"] * k, "Δ vs nominal": r["error_vs_nominal_pct"] / 100}
        if truth is not None:
            row["Simulated true value"] = truth[p] * k
            row["Error vs true"] = (r["estimate"] - truth[p]) / truth[p]
        rows.append(row)
    return pd.DataFrame(rows)


def validation_summary(estimator: str, noise_pct: float, path: Path = REPORT_PATH) -> pd.DataFrame | None:
    """The validation report's rows for this estimator at the closest evaluated noise level, or None
    if the report does not exist."""
    try:
        rep = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    rows = [r for r in rep.get("summary", []) if r["estimator"] == estimator]
    if not rows:
        return None
    levels = sorted({r["noise_pct"] for r in rows})
    level = min(levels, key=lambda x: abs(x - noise_pct))
    out = pd.DataFrame([r for r in rows if r["noise_pct"] == level])[["param", "noise_pct", "mean_abs_error_pct", "max_abs_error_pct", "coverage", "n"]]
    return out.rename(columns={"param": "Parameter", "noise_pct": "Noise (%)", "mean_abs_error_pct": "Mean abs. error (%)", "max_abs_error_pct": "Max abs. error (%)",
                               "coverage": "Interval coverage", "n": "n"})
