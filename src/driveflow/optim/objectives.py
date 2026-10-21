"""Controller-tuning objectives -- expansion task E7 (spec Sec. 4; first case S-04).

S-04: the DC-motor PI cascade gives ≈15 % overshoot on small, unsaturated speed steps. Its source
is the PI zero, not the poles. The search space is the speed loop's design parameters, all existing
arguments of ``PICascadeController``:
- ``speed_bandwidth_rad_s``;
- ``speed_damping``;
- ``setpoint_weight`` b (task C2), which weights the reference in the proportional term only.
The current loop is not touched.

Every candidate is simulated with ``validation.benches.run_dc_closed_loop``, the same closed-loop
bench as the controller tests. Each condition is either a speed-reference step or a load-torque
step, and gives:
- step: overshoot %, 2 % settling time (ms), RMSE over the 0.3 s after the step;
- load step: the speed dip (rad/s);
- every condition: CPU wall time.

The two grids are fixed in this module, before any optimisation:
- ``TUNING_GRID`` is what the optimiser sees: the S-04 step and a load step.
- ``EVALUATION_GRID`` is used only to compare the baseline with the candidate. It is distinct: other
  step sizes, a saturating step, plant drift, another load and speed.

Scalar objective (PSO, BO), fixed before running:
    J = mean overshoot % + 0.1 × mean settling time (ms) + 2 × mean load-step dip (rad/s)
Vector objective (NSGA-II): (mean overshoot %, mean settling time ms, mean dip rad/s).
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from driveflow.validation.benches import run_dc_closed_loop
from driveflow.validation.step_metrics import step_response_metrics

PARAMS = ("speed_bandwidth_rad_s", "speed_damping", "setpoint_weight")
BOUNDS = ((40.0, 160.0), (0.6, 1.2), (0.3, 1.0))
BASELINE = {"speed_bandwidth_rad_s": 80.0, "speed_damping": 0.9, "setpoint_weight": 1.0}
T_EVENT_S = 0.4
DURATION_S = 0.7
DRIFT = {"psi_e": 0.165 * 0.9, "j_rotor": 0.025 * 1.5}
REUSED_WARNING = "The evaluation reused tuning conditions; results are not an independent generalization estimate."


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # "step" or "load"
    omega_rad_s: float
    step_rad_s: float = 0.0
    load_nm: float = 0.0
    motor_parameter: dict | None = None


TUNING_GRID = (
    Condition("s04_step_2rad_s_at_150", "step", 150.0, step_rad_s=2.0),
    Condition("load_10Nm_at_150", "load", 150.0, load_nm=10.0),
)
EVALUATION_GRID = (
    Condition("step_5rad_s_at_120", "step", 120.0, step_rad_s=5.0),
    Condition("saturating_step_30rad_s_at_100", "step", 100.0, step_rad_s=30.0),
    Condition("drift_step_2rad_s_at_150", "step", 150.0, step_rad_s=2.0, motor_parameter=DRIFT),
    Condition("drift_load_15Nm_at_200", "load", 200.0, load_nm=15.0, motor_parameter=DRIFT),
)


def as_kwargs(x) -> dict:
    return {p: float(v) for p, v in zip(PARAMS, x)}


def simulate_condition(c: Condition, controller_kwargs: dict) -> dict:
    t0 = time.process_time()
    if c.kind == "step":
        r = run_dc_closed_loop("PI", lambda tt, a=c.omega_rad_s, b=c.omega_rad_s + c.step_rad_s: a if tt < T_EVENT_S else b, duration_s=DURATION_S,
                               motor_parameter=c.motor_parameter, controller_kwargs=controller_kwargs)
    else:
        r = run_dc_closed_loop("PI", c.omega_rad_s, duration_s=DURATION_S, load_step_nm=c.load_nm, load_step_t_s=T_EVENT_S,
                               motor_parameter=c.motor_parameter, controller_kwargs=controller_kwargs)
    cpu = time.process_time() - t0
    after = r["t"] >= T_EVENT_S
    omega = r["omega"][after]
    out = {"condition": c.name, "kind": c.kind, "cpu_s": cpu, "finite": bool(np.all(np.isfinite(r["omega"])))}
    if c.kind == "step":
        m = step_response_metrics(r["t"][after] - T_EVENT_S, omega, c.omega_rad_s + c.step_rad_s, c.omega_rad_s)
        win = (r["t"] >= T_EVENT_S) & (r["t"] < T_EVENT_S + 0.3)
        out.update({"overshoot_pct": float(m.overshoot_pct), "settling_ms": None if m.settling_time_s is None else float(m.settling_time_s * 1e3),
                    "rise_ms": None if m.rise_time_s is None else float(m.rise_time_s * 1e3),
                    "rmse_rad_s": float(np.sqrt(np.mean((r["omega"][win] - r["omega_ref"][win]) ** 2)))})
    else:
        out.update({"dip_rad_s": float(c.omega_rad_s - omega.min()), "final_error_rad_s": float(abs(omega[-1] - c.omega_rad_s))})
    return out


#: settling time used for a step that never settles inside the run (penalty, not a measurement)
UNSETTLED_MS = (DURATION_S - T_EVENT_S) * 1e3


def summarize(results: list[dict]) -> dict:
    steps = [r for r in results if r["kind"] == "step"]
    loads = [r for r in results if r["kind"] == "load"]
    ov = float(np.mean([r["overshoot_pct"] for r in steps])) if steps else 0.0
    st = float(np.mean([r["settling_ms"] if r["settling_ms"] is not None else UNSETTLED_MS for r in steps])) if steps else 0.0
    dip = float(np.mean([r["dip_rad_s"] for r in loads])) if loads else 0.0
    finite = all(r["finite"] for r in results)
    J = ov + 0.1 * st + 2.0 * dip if finite else math.inf
    return {"J": J, "mean_overshoot_pct": ov, "mean_settling_ms": st, "mean_dip_rad_s": dip, "cpu_s": float(sum(r["cpu_s"] for r in results)), "finite": finite}


def evaluate(x, grid=TUNING_GRID) -> tuple[dict, list[dict]]:
    kw = as_kwargs(x) if not isinstance(x, dict) else x
    results = [simulate_condition(c, kw) for c in grid]
    return summarize(results), results


def scalar_objective(x) -> float:
    return evaluate(x)[0]["J"]


def vector_objective(x) -> np.ndarray:
    s, _ = evaluate(x)
    if not s["finite"]:
        return np.array([math.inf, math.inf, math.inf])
    return np.array([s["mean_overshoot_pct"], s["mean_settling_ms"], s["mean_dip_rad_s"]])


def grids_overlap(tuning=TUNING_GRID, evaluation=EVALUATION_GRID) -> bool:
    """True when an evaluation condition equals a tuning one (then REUSED_WARNING applies)."""
    key = lambda c: (c.kind, c.omega_rad_s, c.step_rad_s, c.load_nm, tuple(sorted((c.motor_parameter or {}).items())))  # noqa: E731
    return bool({key(c) for c in tuning} & {key(c) for c in evaluation})
