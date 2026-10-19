"""Step-response and tracking metrics.

Definitions (stated once here so every test and report uses the same ones):
- rise time: time for the response to go from 10 % to 90 % of the step (y0 -> y_final_ref),
  measured from the step instant. None if 90 % is never reached.
- settling time: last time the response is OUTSIDE a +-band (fraction of the step size) around the
  reference, measured from the step instant; 0 if it never leaves the band, None if it is still
  outside at the end of the record (did not settle).
- overshoot: max excursion past the reference in the direction of the step, as % of step size
  (0 if none).
- steady-state error: mean |ref - y| over the last `ss_fraction` of the record.
- RMSE / max absolute error: over the whole record after the step (transient included).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StepMetrics:
    rise_time_s: float | None
    settling_time_s: float | None
    overshoot_pct: float
    steady_state_error: float
    rmse: float
    max_abs_error: float
    finite: bool

    def as_dict(self) -> dict:
        return asdict(self)


def step_response_metrics(t, y, ref: float, y0: float, t_step: float = 0.0, band: float = 0.02, ss_fraction: float = 0.1) -> StepMetrics:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = bool(np.all(np.isfinite(y)))
    mask = t >= t_step
    t, y = t[mask] - t_step, y[mask]
    step = ref - y0
    if step == 0:
        raise ValueError("step_response_metrics needs ref != y0")
    direction = np.sign(step)
    progress = (y - y0) / step  # 0 at start, 1 at the reference

    def first_crossing(level):
        idx = np.flatnonzero(progress >= level)
        return float(t[idx[0]]) if idx.size else None

    t10, t90 = first_crossing(0.1), first_crossing(0.9)
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None

    outside = np.abs(y - ref) > band * abs(step)
    if not outside.any():
        settling = 0.0
    elif outside[-1]:
        settling = None
    else:
        settling = float(t[np.flatnonzero(outside)[-1] + 1])

    overshoot = max(0.0, float(np.max(direction * (y - ref))) / abs(step) * 100.0)
    n_ss = max(1, int(len(y) * ss_fraction))
    err = ref - y
    return StepMetrics(
        rise_time_s=rise,
        settling_time_s=settling,
        overshoot_pct=overshoot,
        steady_state_error=float(np.mean(np.abs(err[-n_ss:]))),
        rmse=float(np.sqrt(np.mean(err**2))),
        max_abs_error=float(np.max(np.abs(err))),
        finite=finite,
    )


def tracking_metrics(y, ref) -> dict:
    """RMSE / max abs error / mean abs error between a response and a time-varying reference."""
    err = np.asarray(ref, dtype=float) - np.asarray(y, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mae": float(np.mean(np.abs(err))),
        "finite": bool(np.all(np.isfinite(err))),
    }
