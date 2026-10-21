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

Objective "J_effort" (2026-10-21, decided by the user after E7 pushed the bandwidth to its bound;
fixed here before running it):
- control effort: RMS of the change of the armature-voltage command per control step over the
  0.3 s after the event (V), i.e. the variation of the action;
- constraint: peak |armature current| after a reference step <= I_NOMINAL_A (the motor's rated
  current); each 1 % above it adds 0.1 to J_effort;
- J_effort = overshoot/ov0 + settling/st0 + dip/dip0 + effort/eff0 + 10*max(0, peak/I_NOMINAL_A - 1),
  every term divided by the BASELINE's value on the tuning grid, so the baseline scores 4.
- evaluation on EVALUATION_GRID_NOISY: the evaluation grid with a speed-sensor noise of
  NOISE_STD_RAD_S on every condition (plant drift stays on two of them).
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
#: rated armature current of DcPermanentlyExcitedMotor (nominal_values["i"]); J_effort's constraint
I_NOMINAL_A = 97.0
#: speed-sensor noise of the noisy evaluation grid: 0.2 % of the rated 300 rad/s
NOISE_STD_RAD_S = 0.6
#: window after the event over which effort and peak current are measured
EFFORT_WINDOW_S = 0.3


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # "step" or "load"
    omega_rad_s: float
    step_rad_s: float = 0.0
    load_nm: float = 0.0
    motor_parameter: dict | None = None
    noise_std_rad_s: float = 0.0


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
EVALUATION_GRID_NOISY = tuple(dataclasses.replace(c, name=c.name + "_noisy", noise_std_rad_s=NOISE_STD_RAD_S) for c in EVALUATION_GRID)


def as_kwargs(x) -> dict:
    return {p: float(v) for p, v in zip(PARAMS, x)}


def simulate_condition(c: Condition, controller_kwargs: dict) -> dict:
    t0 = time.process_time()
    if c.kind == "step":
        r = run_dc_closed_loop("PI", lambda tt, a=c.omega_rad_s, b=c.omega_rad_s + c.step_rad_s: a if tt < T_EVENT_S else b, duration_s=DURATION_S,
                               motor_parameter=c.motor_parameter, controller_kwargs=controller_kwargs, speed_noise_std_rad_s=c.noise_std_rad_s)
    else:
        r = run_dc_closed_loop("PI", c.omega_rad_s, duration_s=DURATION_S, load_step_nm=c.load_nm, load_step_t_s=T_EVENT_S,
                               motor_parameter=c.motor_parameter, controller_kwargs=controller_kwargs, speed_noise_std_rad_s=c.noise_std_rad_s)
    cpu = time.process_time() - t0
    after = r["t"] >= T_EVENT_S
    omega = r["omega"][after]
    out = {"condition": c.name, "kind": c.kind, "cpu_s": cpu, "finite": bool(np.all(np.isfinite(r["omega"])))}
    ew = (r["t"] >= T_EVENT_S) & (r["t"] < T_EVENT_S + EFFORT_WINDOW_S)
    du = np.diff(r["u"])[ew[1:]]
    out.update({"effort_v": float(np.sqrt(np.mean(du ** 2))) if du.size else 0.0, "peak_i_a": float(np.max(np.abs(r["i"][ew]))) if ew.any() else 0.0})
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
    effort = float(np.mean([r.get("effort_v", 0.0) for r in results])) if results else 0.0
    peak = float(max([r.get("peak_i_a", 0.0) for r in steps], default=0.0))
    return {"J": J, "mean_overshoot_pct": ov, "mean_settling_ms": st, "mean_dip_rad_s": dip, "cpu_s": float(sum(r["cpu_s"] for r in results)), "finite": finite,
            "mean_effort_v": effort, "max_step_peak_i_a": peak, "current_constraint_ok": peak <= I_NOMINAL_A}


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


_BASELINE_TUNING: dict | None = None


def baseline_tuning_summary() -> dict:
    """The BASELINE's summary on the tuning grid: the normalisers of J_effort (computed once)."""
    global _BASELINE_TUNING
    if _BASELINE_TUNING is None:
        _BASELINE_TUNING = evaluate(BASELINE, TUNING_GRID)[0]
    return _BASELINE_TUNING


def j_effort(summary: dict) -> float:
    if not summary["finite"]:
        return math.inf
    b = baseline_tuning_summary()
    terms = [summary["mean_overshoot_pct"] / b["mean_overshoot_pct"], summary["mean_settling_ms"] / b["mean_settling_ms"],
             summary["mean_dip_rad_s"] / b["mean_dip_rad_s"], summary["mean_effort_v"] / b["mean_effort_v"]]
    return float(sum(terms) + 10.0 * max(0.0, summary["max_step_peak_i_a"] / I_NOMINAL_A - 1.0))


def scalar_objective_effort(x) -> float:
    return j_effort(evaluate(x)[0])


def vector_objective_effort(x) -> np.ndarray:
    """(overshoot %, dip rad/s, effort V); a candidate breaking the current constraint is infeasible."""
    s, _ = evaluate(x)
    if not s["finite"] or not s["current_constraint_ok"]:
        return np.array([math.inf, math.inf, math.inf])
    return np.array([s["mean_overshoot_pct"], s["mean_dip_rad_s"], s["mean_effort_v"]])


#: objective name -> (scalar, vector, evaluation grid, description)
OBJECTIVES = {
    "J": (scalar_objective, vector_objective, EVALUATION_GRID, "J = mean overshoot % + 0.1 x mean settling ms + 2 x mean load dip rad/s"),
    "J_effort": (scalar_objective_effort, vector_objective_effort, EVALUATION_GRID_NOISY,
                 "J_effort = overshoot/ov0 + settling/st0 + dip/dip0 + effort/eff0 + 10*max(0, peak step current/97 A - 1); baseline = 4; "
                 "NSGA-II: (overshoot, dip, effort) with peak step current <= 97 A; evaluation grid with 0.6 rad/s speed-sensor noise"),
}


def grids_overlap(tuning=TUNING_GRID, evaluation=EVALUATION_GRID) -> bool:
    """True when an evaluation condition equals a tuning one (then REUSED_WARNING applies)."""
    key = lambda c: (c.kind, c.omega_rad_s, c.step_rad_s, c.load_nm, tuple(sorted((c.motor_parameter or {}).items())), c.noise_std_rad_s)  # noqa: E731
    return bool({key(c) for c in tuning} & {key(c) for c in evaluation})
