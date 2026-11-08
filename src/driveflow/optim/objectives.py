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

Another base motor (round Assistant, task F0.6, decision A4). Every function takes an optional
``motor`` (BaseMotor: motor_parameter_overrides, supply_v, limit_overrides, nominal_overrides, as the
Scenario fields). The controller is designed on the base motor; a drift condition multiplies the
base motor's psi_e by 0.9 and j_rotor by 1.5 (DRIFT_FACTORS). With motor=None or STOCK every
simulation receives exactly the arguments it received before, so results are identical. J_effort's
current constraint uses the base motor's rated current: 97 A for the stock motor, otherwise
nominal_overrides["i"]; with a changed motor and no rated current J_effort is refused. The grids'
speeds, steps, loads and the noise level are scaled to the base motor (round Assistant 0b, D2, user
decision 2026-11-02): grids_for(motor) multiplies speeds, speed steps and the sensor noise by
omega_rated / 300 rad/s and load steps by torque_rated / 16 N*m (the stock ratings). The stock motor
uses the module's grids unchanged; a changed motor without rated speed and torque
(nominal_overrides 'omega', 'torque') cannot be tuned, since nothing is invented.
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
#: plant drift of the drift conditions, relative to the base motor (F0.6)
DRIFT_FACTORS = {"psi_e": 0.9, "j_rotor": 1.5}
#: the same drift in absolute values for the stock motor (kept for reference; == 0.165 * 0.9, 0.025 * 1.5)
DRIFT = {"psi_e": 0.165 * 0.9, "j_rotor": 0.025 * 1.5}
REUSED_WARNING = "The evaluation reused tuning conditions; results are not an independent generalization estimate."
#: rated armature current of DcPermanentlyExcitedMotor (nominal_values["i"]); J_effort's constraint
I_NOMINAL_A = 97.0
#: speed-sensor noise of the noisy evaluation grid: 0.2 % of the rated 300 rad/s
NOISE_STD_RAD_S = 0.6
#: window after the event over which effort and peak current are measured
EFFORT_WINDOW_S = 0.3


class MotorSpecError(ValueError):
    """The base motor cannot be tuned with the requested objective (e.g. no rated current)."""


@dataclasses.dataclass(frozen=True)
class BaseMotor:
    """The motor the PI is designed for and tuned on (F0.6). Fields as the Scenario fields
    motor_parameter_overrides, dc_supply_v, dc_limit_overrides, dc_nominal_overrides; all None is the
    stock motor (GEM PMG 132) on the 65 V supply."""
    motor_parameter_overrides: dict | None = None
    supply_v: float | None = None
    limit_overrides: dict | None = None
    nominal_overrides: dict | None = None
    #: decision E3: None = GEM's one-quadrant converter (unchanged); "dcm_exact" = the v2 plants' converter
    converter_model: str | None = None

    @property
    def is_stock(self) -> bool:
        return self == STOCK

    @property
    def changes_motor(self) -> bool:
        return bool(self.motor_parameter_overrides)

    def key(self) -> tuple:
        return tuple((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in dataclasses.asdict(self).items())

    def rated_current_a(self) -> float:
        """J_effort's current bound: 97 A for the stock motor, else nominal_overrides["i"]."""
        i = (self.nominal_overrides or {}).get("i")
        if i is not None:
            return float(i)
        if self.changes_motor:
            raise MotorSpecError("J_effort needs the motor's rated current: set nominal_overrides['i'] (dc_nominal_overrides['i']); "
                                 "SPADE does not invent one for a motor other than the stock one")
        return I_NOMINAL_A

    def plant_parameters(self, drift_factors: dict | None) -> dict | None:
        """motor_parameter for the simulated plant: the base overrides, times the drift factors."""
        if not drift_factors:
            return self.motor_parameter_overrides
        from driveflow.sim import DcPermanentlyExcitedMotor

        base = {**DcPermanentlyExcitedMotor._default_motor_parameter, **(self.motor_parameter_overrides or {})}
        if self.motor_parameter_overrides is None:  # stock: the exact former DRIFT dict
            return {k: base[k] * f for k, f in drift_factors.items()}
        return {**self.motor_parameter_overrides, **{k: base[k] * f for k, f in drift_factors.items()}}

    def bench_kwargs(self) -> dict:
        """Extra run_dc_closed_loop arguments; empty for the stock motor so its calls are unchanged.
        converter_model (E3) only adds the bench's converter argument."""
        conv = {"converter": self.converter_model} if self.converter_model else {}
        if dataclasses.replace(self, converter_model=None).is_stock:
            return conv
        kw = {**conv, "design_motor_parameter": self.motor_parameter_overrides}
        if self.supply_v is not None:
            kw["supply_v"] = self.supply_v
        if self.limit_overrides is not None:
            kw["limit_values"] = self.limit_overrides
        if self.nominal_overrides is not None:
            kw["nominal_values"] = self.nominal_overrides
        from driveflow.datagen.runner import dc_default_load_parameter

        load = dc_default_load_parameter(self.motor_parameter_overrides, self.nominal_overrides)  # 0b D1, same rule as Scenario
        if load is not None:
            kw["load_parameter"] = load
        return kw


STOCK = BaseMotor()


def motor_from_spec(block: dict | None) -> BaseMotor:
    """A BaseMotor from a job's params["motor"] block (keys as Scenario: motor_parameter_overrides,
    dc_supply_v, dc_limit_overrides, dc_nominal_overrides), validated by Scenario itself."""
    if not block:
        return STOCK
    allowed = ("motor_parameter_overrides", "dc_supply_v", "dc_limit_overrides", "dc_nominal_overrides", "dc_converter_model")
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise MotorSpecError(f"motor: unknown key(s) {unknown}; expected some of {allowed}")
    from driveflow.datagen import Scenario

    Scenario("tuning_motor_check", **block)  # same validation as a simulation (ValueError if invalid)
    mp = block.get("motor_parameter_overrides")
    if mp:
        from driveflow.sim import DcPermanentlyExcitedMotor

        unknown = sorted(set(mp) - set(DcPermanentlyExcitedMotor._default_motor_parameter))
        if unknown:
            raise MotorSpecError(f"motor.motor_parameter_overrides: unknown parameter(s) {unknown}")
    conv = block.get("dc_converter_model")
    if conv not in (None, "dcm_exact"):
        raise MotorSpecError(f"motor.dc_converter_model: tuning supports GEM's converter (unset) or 'dcm_exact' (v2 plants), got {conv!r}")
    return BaseMotor(mp or None, block.get("dc_supply_v"), block.get("dc_limit_overrides"), block.get("dc_nominal_overrides"), conv)


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    kind: str  # "step" or "load"
    omega_rad_s: float
    step_rad_s: float = 0.0
    load_nm: float = 0.0
    motor_parameter: dict | None = None  # absolute plant parameters (not used by the shipped grids)
    noise_std_rad_s: float = 0.0
    drift: dict | None = None  # plant drift relative to the base motor (F0.6), e.g. DRIFT_FACTORS

    @property
    def is_drift(self) -> bool:
        return bool(self.drift or self.motor_parameter)


TUNING_GRID = (
    Condition("s04_step_2rad_s_at_150", "step", 150.0, step_rad_s=2.0),
    Condition("load_10Nm_at_150", "load", 150.0, load_nm=10.0),
)
EVALUATION_GRID = (
    Condition("step_5rad_s_at_120", "step", 120.0, step_rad_s=5.0),
    Condition("saturating_step_30rad_s_at_100", "step", 100.0, step_rad_s=30.0),
    Condition("drift_step_2rad_s_at_150", "step", 150.0, step_rad_s=2.0, drift=DRIFT_FACTORS),
    Condition("drift_load_15Nm_at_200", "load", 200.0, load_nm=15.0, drift=DRIFT_FACTORS),
)
EVALUATION_GRID_NOISY = tuple(dataclasses.replace(c, name=c.name + "_noisy", noise_std_rad_s=NOISE_STD_RAD_S) for c in EVALUATION_GRID)


#: the stock motor's rated speed and torque, the reference of the grid scaling (DcPermanentlyExcitedMotor nominal values)
STOCK_RATED_OMEGA_RAD_S = 300.0
STOCK_RATED_TORQUE_NM = 16.0


def grids_for(motor: "BaseMotor | None" = None) -> tuple:
    """(tuning grid, evaluation grid, noisy evaluation grid) for a base motor (0b D2)."""
    motor = motor or STOCK
    if not motor.changes_motor:
        return TUNING_GRID, EVALUATION_GRID, EVALUATION_GRID_NOISY
    nominal = motor.nominal_overrides or {}
    if "omega" not in nominal or "torque" not in nominal:
        raise MotorSpecError("tuning on another motor needs its rated speed and torque (nominal_overrides / dc_nominal_overrides 'omega' and 'torque'): "
                             "the grids are scaled to them, and SPADE does not invent ratings")
    ks, kt = float(nominal["omega"]) / STOCK_RATED_OMEGA_RAD_S, float(nominal["torque"]) / STOCK_RATED_TORQUE_NM

    def scale(c: Condition) -> Condition:
        return dataclasses.replace(c, name=c.name + "_scaled", omega_rad_s=c.omega_rad_s * ks, step_rad_s=c.step_rad_s * ks, load_nm=c.load_nm * kt,
                                   noise_std_rad_s=c.noise_std_rad_s * ks)

    return tuple(tuple(scale(c) for c in g) for g in (TUNING_GRID, EVALUATION_GRID, EVALUATION_GRID_NOISY))


def as_kwargs(x) -> dict:
    return {p: float(v) for p, v in zip(PARAMS, x)}


def simulate_condition(c: Condition, controller_kwargs: dict, motor: BaseMotor | None = None) -> dict:
    motor = motor or STOCK
    plant_mp = c.motor_parameter if c.motor_parameter is not None else motor.plant_parameters(c.drift)
    extra = motor.bench_kwargs()
    t0 = time.process_time()
    if c.kind == "step":
        r = run_dc_closed_loop("PI", lambda tt, a=c.omega_rad_s, b=c.omega_rad_s + c.step_rad_s: a if tt < T_EVENT_S else b, duration_s=DURATION_S,
                               motor_parameter=plant_mp, controller_kwargs=controller_kwargs, speed_noise_std_rad_s=c.noise_std_rad_s, **extra)
    else:
        r = run_dc_closed_loop("PI", c.omega_rad_s, duration_s=DURATION_S, load_step_nm=c.load_nm, load_step_t_s=T_EVENT_S,
                               motor_parameter=plant_mp, controller_kwargs=controller_kwargs, speed_noise_std_rad_s=c.noise_std_rad_s, **extra)
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


def summarize(results: list[dict], i_nominal_a: float | None = I_NOMINAL_A) -> dict:
    """i_nominal_a: the rated current of J_effort's constraint (None: not known, so not judged)."""
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
            "mean_effort_v": effort, "max_step_peak_i_a": peak, "current_constraint_ok": None if i_nominal_a is None else peak <= i_nominal_a}


def _rated_or_none(motor: BaseMotor) -> float | None:
    try:
        return motor.rated_current_a()
    except MotorSpecError:
        return None


def evaluate(x, grid=TUNING_GRID, motor: BaseMotor | None = None) -> tuple[dict, list[dict]]:
    motor = motor or STOCK
    kw = as_kwargs(x) if not isinstance(x, dict) else x
    results = [simulate_condition(c, kw, motor) for c in grid]
    return summarize(results, _rated_or_none(motor)), results


def scalar_objective(x) -> float:
    return evaluate(x)[0]["J"]


def vector_objective(x) -> np.ndarray:
    s, _ = evaluate(x)
    if not s["finite"]:
        return np.array([math.inf, math.inf, math.inf])
    return np.array([s["mean_overshoot_pct"], s["mean_settling_ms"], s["mean_dip_rad_s"]])


_BASELINE_TUNING: dict = {}


def baseline_tuning_summary(motor: BaseMotor | None = None) -> dict:
    """The BASELINE's summary on the tuning grid: the normalisers of J_effort (computed once per motor)."""
    motor = motor or STOCK
    if motor.key() not in _BASELINE_TUNING:
        _BASELINE_TUNING[motor.key()] = evaluate(BASELINE, grids_for(motor)[0], motor)[0]
    return _BASELINE_TUNING[motor.key()]


def j_effort(summary: dict, motor: BaseMotor | None = None) -> float:
    motor = motor or STOCK
    i_rated = motor.rated_current_a()  # raises MotorSpecError for a changed motor without one
    if not summary["finite"]:
        return math.inf
    b = baseline_tuning_summary(motor)
    terms = [summary["mean_overshoot_pct"] / b["mean_overshoot_pct"], summary["mean_settling_ms"] / b["mean_settling_ms"],
             summary["mean_dip_rad_s"] / b["mean_dip_rad_s"], summary["mean_effort_v"] / b["mean_effort_v"]]
    return float(sum(terms) + 10.0 * max(0.0, summary["max_step_peak_i_a"] / i_rated - 1.0))


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


def objective_functions(objective: str, motor: BaseMotor | None = None) -> tuple:
    """(scalar, vector, evaluation grid, description, score) of an objective on a base motor (F0.6).
    For the stock motor these are OBJECTIVES' own functions. J_effort on a changed motor without a
    rated current raises MotorSpecError here, before any simulation."""
    motor = motor or STOCK
    scalar, vector, grid, text = OBJECTIVES[objective]
    if objective == "J_effort":
        i_rated = motor.rated_current_a()
    if motor.is_stock:
        return scalar, vector, grid, text, ((lambda s: s["J"]) if objective == "J" else j_effort)
    tuning, evaluation, evaluation_noisy = grids_for(motor)  # 0b D2: MotorSpecError without ratings
    grid = evaluation if objective == "J" else evaluation_noisy
    if objective == "J":
        def scalar(x):
            return evaluate(x, tuning, motor)[0]["J"]

        def vector(x):
            s, _ = evaluate(x, tuning, motor)
            return np.array([math.inf] * 3) if not s["finite"] else np.array([s["mean_overshoot_pct"], s["mean_settling_ms"], s["mean_dip_rad_s"]])

        return scalar, vector, grid, text, (lambda s: s["J"])

    def scalar(x):
        return j_effort(evaluate(x, tuning, motor)[0], motor)

    def vector(x):
        s, _ = evaluate(x, tuning, motor)
        if not s["finite"] or not s["current_constraint_ok"]:
            return np.array([math.inf] * 3)
        return np.array([s["mean_overshoot_pct"], s["mean_dip_rad_s"], s["mean_effort_v"]])

    text = text.replace("0.6 rad/s", f"{NOISE_STD_RAD_S * float(motor.nominal_overrides['omega']) / STOCK_RATED_OMEGA_RAD_S:.3g} rad/s")
    return scalar, vector, grid, text.replace("97 A", f"{i_rated:g} A") + "; grids scaled to the base motor's ratings", (lambda s: j_effort(s, motor))


def grids_overlap(tuning=TUNING_GRID, evaluation=EVALUATION_GRID) -> bool:
    """True when an evaluation condition equals a tuning one (then REUSED_WARNING applies)."""
    key = lambda c: (c.kind, c.omega_rad_s, c.step_rad_s, c.load_nm, tuple(sorted((c.motor_parameter or {}).items())), tuple(sorted((c.drift or {}).items())), c.noise_std_rad_s)  # noqa: E731
    return bool({key(c) for c in tuning} & {key(c) for c in evaluation})
