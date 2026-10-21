"""Closed-/open-loop benches over the platform's existing plants and controllers.

The DC-motor plant is built with exactly the parameters of datagen/runner.py's build_plant()
(dc_perm_ex_v1: GEM DcPermanentlyExcitedMotor, PolynomialStaticLoad a=0.01 b=0.05 c=0 j_load=0.0025,
65 V ideal supply, one-quadrant converter, tau=1e-4) -- tests/simulation asserts that equality --
but with a selectable ODE solver and an optional load-torque step, which build_plant() does not
expose. The PMSM bench reuses control/classical/pmsm_foc.py; the VSC bench goes through the real
datagen path (run_scenario), since that plant is a fixed identified discrete-time model.
"""

from __future__ import annotations

import time

import numpy as np

from driveflow.control.classical import PICascadeController
from driveflow.control.classical import pmsm_foc
from driveflow.control.mpc import MpcController
from driveflow.datagen import Scenario, run_scenario
from driveflow.sim import (
    ConstantSpeedLoad,
    ContB6BridgeConverter,
    ContFourQuadrantConverter,
    ContOneQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PermanentMagnetSynchronousMotor,
    PolynomialStaticLoad,
    SynchronousMotorSystem,
)

TAU = 1e-4
#: datagen/runner.py build_plant()'s load and supply (dc_perm_ex_v1).
DC_LOAD_PARAMETER = dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)
DC_SUPPLY_V = 65.0
_CONTROLLERS = {"PI": PICascadeController, "MPC": MpcController}


class LoadTorqueStep(PolynomialStaticLoad):
    """PolynomialStaticLoad plus an external load torque that steps from 0 to `step_nm` at
    `t_step_s` (ODE time). A disturbance input only -- the polynomial friction is unchanged."""

    def __init__(self, load_parameter=None, step_nm: float = 0.0, t_step_s: float = np.inf, **kwargs):
        super().__init__(load_parameter=load_parameter, **kwargs)
        self.step_nm = step_nm
        self.t_step_s = t_step_s

    def mechanical_ode(self, t, mechanical_state, torque):
        extra = self.step_nm if t >= self.t_step_s - 1e-12 else 0.0
        return super().mechanical_ode(t, mechanical_state, torque - extra)


def build_dc_system(ode_solver=None, tau: float = TAU, motor_parameter: dict | None = None, load_parameter: dict | None = None,
                    load_step_nm: float = 0.0, load_step_t_s: float = np.inf, converter: str = "one_quadrant",
                    supply_v: float = DC_SUPPLY_V) -> DcMotorSystem:
    load = LoadTorqueStep(load_parameter={**DC_LOAD_PARAMETER, **(load_parameter or {})}, step_nm=load_step_nm, t_step_s=load_step_t_s)
    conv = {"one_quadrant": ContOneQuadrantConverter, "four_quadrant": ContFourQuadrantConverter}[converter]()
    return DcMotorSystem(
        converter=conv,
        motor=DcPermanentlyExcitedMotor(motor_parameter=motor_parameter),
        load=load,
        supply=IdealVoltageSupply(u_nominal=supply_v),
        ode_solver=ode_solver if ode_solver is not None else EulerSolver(),
        tau=tau,
    )


def _dc_record(system, state):
    p = state * system.limits
    return p[system.OMEGA_IDX], p[system.CURRENTS_IDX[0]], p[system.VOLTAGES_IDX[0]], p[system.TORQUE_IDX]


def run_dc_closed_loop(controller: str = "PI", omega_ref=150.0, duration_s: float = 0.5, ode_solver=None, tau: float = TAU,
                       motor_parameter: dict | None = None, load_parameter: dict | None = None, load_step_nm: float = 0.0,
                       load_step_t_s: float = np.inf, torque_ref_nm: float | None = None, seed: int = 0,
                       controller_kwargs: dict | None = None, speed_noise_std_rad_s: float = 0.0, noise_seed: int = 0) -> dict:
    """omega_ref: constant (rad/s) or callable t -> rad/s (reference profile).
    speed_noise_std_rad_s: white Gaussian noise (rad/s) added to the speed the CONTROLLER sees (a
    sensor); the recorded omega is the true speed. 0 (default) leaves the loop untouched.
    The controller is tuned from the NOMINAL plant (motor_parameter=None) when `motor_parameter`
    perturbs the plant -- i.e. a robustness test (the controller does not know the drift)."""
    system = build_dc_system(ode_solver, tau, motor_parameter, load_parameter, load_step_nm, load_step_t_s)
    design_plant = system if motor_parameter is None else build_dc_system(None, tau, None, load_parameter)
    ctrl = _CONTROLLERS[controller](design_plant, **(controller_kwargs or {}))
    system.seed(np.random.SeedSequence(seed))
    state = system.reset()
    ctrl.reset()
    n = int(round(duration_s / tau))
    t = (np.arange(n) + 1) * tau
    ref_fn = omega_ref if callable(omega_ref) else (lambda _t, v=float(omega_ref): v)
    noise = None
    if speed_noise_std_rad_s > 0:
        w_idx = list(system.state_names).index("omega")
        noise = np.random.default_rng(noise_seed).normal(0.0, speed_noise_std_rad_s / float(system.limits[w_idx]), n)
    out = np.empty((n, 5))
    t0 = time.perf_counter()
    for k in range(n):
        ref_k = ref_fn(k * tau)
        seen = state
        if noise is not None:
            seen = np.array(state, dtype=float, copy=True)
            seen[w_idx] += noise[k]
        action = ctrl.control_torque(seen, torque_ref_nm) if torque_ref_nm is not None else ctrl.control(seen, ref_k)
        state = system.simulate(action)
        out[k, :4] = _dc_record(system, state)
        out[k, 4] = ref_k
    wall = time.perf_counter() - t0
    return {"t": t, "omega": out[:, 0], "i": out[:, 1], "u": out[:, 2], "torque": out[:, 3], "omega_ref": out[:, 4],
            "wall_time_s": wall, "system": system}


def run_dc_open_loop(duty: float = 0.5, duration_s: float = 0.2, ode_solver=None, tau: float = TAU, converter: str = "four_quadrant") -> dict:
    """Constant converter duty cycle, no controller -- isolates the plant + integrator."""
    system = build_dc_system(ode_solver, tau, converter=converter)
    system.seed(np.random.SeedSequence(0))
    state = system.reset()
    n = int(round(duration_s / tau))
    out = np.empty((n, 4))
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n):
            state = system.simulate(np.array([duty]))
            out[k] = _dc_record(system, state)
    return {"t": (np.arange(n) + 1) * tau, "omega": out[:, 0], "i": out[:, 1], "system": system}


def run_pmsm_open_loop(omega_mech: float, duration_s: float = 0.4, ode_solver=None, tau: float = pmsm_foc.TAU, mp: dict = pmsm_foc.MOTOR_PARAMS) -> dict:
    """PMSM with its speed HELD by the load (ConstantSpeedLoad) and zero dq voltage: the currents
    settle to the analytic short-circuit point (i_d -> -psi_p*omega_el^2*l_q/(r_s^2+omega_el^2*l_d*l_q),
    |i| -> ~psi_p/l_d at high speed) if the integration is stable."""
    system = SynchronousMotorSystem(
        converter=ContB6BridgeConverter(),
        motor=PermanentMagnetSynchronousMotor(motor_parameter=mp),
        load=ConstantSpeedLoad(omega_fixed=omega_mech),
        supply=IdealVoltageSupply(u_nominal=pmsm_foc.U_SUPPLY),
        ode_solver=ode_solver if ode_solver is not None else pmsm_foc.default_ode_solver(),
        tau=tau,
        control_space="dq",
    )
    system.seed(np.random.SeedSequence(0))
    state = system.reset()
    n = int(round(duration_s / tau))
    out = np.empty((n, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n):
            state = system.simulate(np.array([0.0, 0.0]))
            p = state * system.limits
            out[k] = p[system.CURRENTS_IDX[3]], p[system.CURRENTS_IDX[4]]
    return {"t": (np.arange(n) + 1) * tau, "i_d": out[:, 0], "i_q": out[:, 1]}


def pmsm_short_circuit_current(omega_mech: float, mp: dict = pmsm_foc.MOTOR_PARAMS) -> tuple:
    """Analytic steady state of run_pmsm_open_loop (u_d = u_q = 0, constant omega_el)."""
    w = omega_mech * mp["p"]
    r, ld, lq, psi = mp["r_s"], mp["l_d"], mp["l_q"], mp["psi_p"]
    den = r**2 + w**2 * ld * lq
    return -psi * w**2 * lq / den, -psi * w * r / den


def run_pmsm_current_step(i_d_ref: float, i_q_ref: float, duration_s: float = 0.03, ode_solver=None, mp: dict = pmsm_foc.MOTOR_PARAMS) -> dict:
    traj = pmsm_foc.run_current_step(i_d_ref, i_q_ref, duration_s=duration_s, mp=mp, ode_solver=ode_solver)
    arr = np.asarray(traj, dtype=float)
    return {"t": (np.arange(len(arr)) + 1) * pmsm_foc.TAU, "i_d": arr[:, 0], "i_q": arr[:, 1]}


def run_vsc_closed_loop(r_ohm: float | None = None, magnitude_v: float | None = None, omega_rad_s: float | None = None,
                        duration_s: float = 0.1, seed: int = 0) -> dict:
    """Real datagen path (Scenario/run_scenario) for the VSC + trained DPC network."""
    t0 = time.perf_counter()
    with np.errstate(over="ignore", invalid="ignore"):
        recs = run_scenario(Scenario(scenario_id="bench_vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=duration_s, seed=seed,
                                     load_resistance_ohm=r_ohm, reference_magnitude_v=magnitude_v, reference_omega_rad_s=omega_rad_s))
    wall = time.perf_counter() - t0
    col = lambda k: np.array([r[k] for r in recs], dtype=float)  # noqa: E731
    return {"t": col("timestamp_s"), "vc_real": col("vc_real"), "vc_imag": col("vc_imag"), "v_ref_real": col("v_ref_real"),
            "v_ref_imag": col("v_ref_imag"), "i_f_real": col("i_f_real"), "i_f_imag": col("i_f_imag"), "wall_time_s": wall}
