"""Does the DC-motor controllers' design assumption hold for a given motor? (round Assistant, F0.2)

Diagnostic only: nothing here changes a controller. ``describe_plant()`` and the future assistant
use it to warn before a run whose controller is designed outside its assumptions.

PI cascade (control/classical/pi_controller.py), auto-tuned from r_a, l_a, psi_e and j_total:
- the current loop is a magnitude-optimum PI with the control period tau as its small delay, so
  its closed-loop bandwidth is about 1 / (2 tau) rad/s whatever the motor;
- the speed loop is designed on the plant i_ref -> omega = psi_e / (j_total s), i.e. assuming the
  current loop is ideal. The module states the condition: the electrical time constant
  tau_a = l_a / r_a much smaller than the mechanical one tau_m = j_total r_a / psi_e**2 (about
  14x for the stock motor), and the current loop much faster than the speed loop.

MPC (control/mpc/controller.py) uses the same two-state model, built from the same parameters, with
a fixed horizon of 20 control periods (2 ms); its docstring sizes it to ~1.7 tau_a of the stock
motor. A motor with a longer tau_a sees less of its current dynamics inside the horizon.

Thresholds (declared here, not tuned on any result):
- tau_m / tau_a >= 10 (the instruction's "much smaller");
- current-loop bandwidth / speed-loop bandwidth >= 10;
- tau_a >= 2 tau: below that, forward Euler at the control period is unstable on the electrical
  state (the plant integrates with one Euler step per tau by default);
- MPC horizon >= tau_a.
"""

from __future__ import annotations

#: S-21 mitigation (decision S21-3): the fixed warning text, and the light-load level below which an
#: experiment is flagged (datagen/s21_warning.py). Non-blocking: it never changes ``status``.
S21_MESSAGE = "one-quadrant converter: light-load deceleration not physical (S-21)"
S21_LIGHT_LOAD_NM = 1.0
S21_EVIDENCE = "docs/audit/assistant/s21_one_quadrant.md"

TAU_RATIO_MIN = 10.0
BANDWIDTH_RATIO_MIN = 10.0
EULER_TAU_A_MIN_STEPS = 2.0
MPC_HORIZON_STEPS = 20  # MpcController's default horizon


def _time_constants(motor_params: dict, j_total: float) -> tuple[float, float]:
    r_a, l_a, psi_e = float(motor_params["r_a"]), float(motor_params["l_a"]), float(motor_params["psi_e"])
    return l_a / r_a, j_total * r_a / psi_e**2


def s21_warning(applies_to: str) -> dict:
    """The S-21 warning (S21-3), as listed in ``warnings``: never blocking, never a status change."""
    return {"id": "S-21", "severity": "warning", "blocking": False, "message": S21_MESSAGE, "applies_to": applies_to, "evidence": S21_EVIDENCE,
            "not_affected_with": "dc_converter_model='dcm' (S21-2)"}


def controller_design_check(motor_params: dict, j_total: float, tau: float, speed_bandwidth_rad_s: float = 80.0, *, s21_reason: str | None = None) -> dict:
    """The PI cascade's design assumptions for this motor. motor_params needs r_a, l_a, psi_e;
    j_total is rotor + load inertia (kg*m^2); tau the control period (s); speed_bandwidth_rad_s the
    PI's speed-loop natural frequency (PICascadeController default 80). ``s21_reason`` (S21-3, from
    datagen.s21_warning.motor_reason) adds the S-21 warning to ``warnings``; ``status`` is about the
    design assumptions only and does not change with it."""
    tau_a, tau_m = _time_constants(motor_params, j_total)
    current_bw = 1.0 / (2.0 * tau)
    reasons = []
    if tau_m / tau_a < TAU_RATIO_MIN:
        reasons.append(f"tau_m/tau_a = {tau_m / tau_a:.1f} < {TAU_RATIO_MIN:g}: the speed loop is designed assuming an ideal current loop, "
                       "which needs the electrical time constant to be much smaller than the mechanical one")
    if current_bw / speed_bandwidth_rad_s < BANDWIDTH_RATIO_MIN:
        reasons.append(f"current-loop bandwidth / speed-loop bandwidth = {current_bw / speed_bandwidth_rad_s:.1f} < {BANDWIDTH_RATIO_MIN:g}")
    if tau_a < EULER_TAU_A_MIN_STEPS * tau:
        reasons.append(f"tau_a = {tau_a:.3g} s is below {EULER_TAU_A_MIN_STEPS:g} control periods: one forward-Euler step per period is unstable on the current")
    return {
        "controller": "PI", "tau_a_s": tau_a, "tau_m_s": tau_m, "tau_m_over_tau_a": tau_m / tau_a,
        "current_loop_bandwidth_rad_s": current_bw, "speed_loop_bandwidth_rad_s": float(speed_bandwidth_rad_s),
        "bandwidth_ratio": current_bw / speed_bandwidth_rad_s, "status": "warning" if reasons else "ok", "reasons": reasons,
        "warnings": [s21_warning(s21_reason)] if s21_reason else [],
    }


def mpc_design_check(motor_params: dict, j_total: float, tau: float, horizon: int = MPC_HORIZON_STEPS) -> dict:
    """The MPC's model and horizon for this motor (same two-state model as the PI design)."""
    tau_a, tau_m = _time_constants(motor_params, j_total)
    horizon_s = horizon * tau
    reasons = []
    if horizon_s < tau_a:
        reasons.append(f"the horizon ({horizon} steps = {horizon_s * 1e3:.2g} ms) is shorter than tau_a = {tau_a * 1e3:.3g} ms: "
                       "the QP sees only part of the current response it decides over")
    if tau_a < EULER_TAU_A_MIN_STEPS * tau:
        reasons.append(f"tau_a = {tau_a:.3g} s is below {EULER_TAU_A_MIN_STEPS:g} control periods: the forward-Euler prediction model is unstable")
    return {"controller": "MPC", "tau_a_s": tau_a, "tau_m_s": tau_m, "horizon_steps": horizon, "horizon_s": horizon_s,
            "horizon_over_tau_a": horizon_s / tau_a, "status": "warning" if reasons else "ok", "reasons": reasons}
