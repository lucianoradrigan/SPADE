"""GUI boundary to the buck surrogate (expansion task E5; spec Sec. 3.14.4 "Compare with SPADE")."""

from __future__ import annotations

import pandas as pd

from driveflow.envs import surrogate_env as se


def load() -> tuple[dict, str] | tuple[None, str]:
    """(surrogate, run name), or (None, reason) when no surrogate has been fitted."""
    try:
        s, path = se.latest_surrogate()
    except FileNotFoundError as exc:
        return None, str(exc)
    return s, path.name


def compare(surrogate: dict, family: str, vin: float, load_ohm: float, vref: float, step_ohm: float | None) -> dict:
    return se.compare_condition(surrogate, family, vin, load_ohm, vref, step_ohm)


def condition_grid(surrogate: dict) -> list[tuple[float, float, str]]:
    """The non-training conditions of the fit (validation, test, extrapolation)."""
    out = []
    for key, split in surrogate["conditions"].items():
        if split == "train":
            continue
        v, r = key.replace("ohm", "").split("V_")
        out.append((float(v), float(r), split))
    return out


def grid_table(surrogate: dict, family: str) -> pd.DataFrame:
    rows = []
    for vin, r, split in condition_grid(surrogate):
        c = se.compare_condition(surrogate, family, vin, r, 0.5 * vin, r / 2)
        rows.append({"V_in (V)": vin, "Load R (Ω)": r, "Split": split, "Load step to (Ω)": r / 2, "Status": "Failed" if c["status"] == "failed" else "OK",
                     "RMSE v_out (V)": c.get("rmse_v_out_v"), "Max |Δv_out| (V)": c.get("max_abs_v_out_v"), "RMSE i_L (A)": c.get("rmse_i_l_a"),
                     "Domain": "; ".join(c["out_of_domain"]) or "inside"})
    return pd.DataFrame(rows)


def overlay_frame(result: dict) -> pd.DataFrame:
    n, s = result["native"], result["surrogate"]
    return pd.DataFrame({"t_ms": n["t"] * 1e3, "native_v_out": n["v_out"], "surrogate_v_out": s["v_out"], "native_i_l": n["i_l"], "surrogate_i_l": s["i_l"],
                         "error_v_out": s["v_out"] - n["v_out"]})
