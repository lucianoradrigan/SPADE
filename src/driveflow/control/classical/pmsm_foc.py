"""Native FOC (field-oriented control) dq current controller for a PMSM, plus the analytic MTPA
(max torque per ampere) locus -- built earlier this session as a one-off demo (only ever lived in
scratch scripts, never in the driveflow package) to reproduce the "profile performance" id/iq
plane figure style (a circular current limit + torque-reference isoline + two-policy point cloud)
using GEM's own vendored PMSM physics, for a *salient* motor (l_d != l_q) where MTPA is a real,
non-trivial curve -- not the "no torque-current" motor comparison GEM's own MPC examples use.

Not wired into Scenario/datagen/runner.py (unlike PICascadeController and DpcController): this is
a standalone illustrative comparison of two control LAWS (MTPA vs. naive i_d=0) on short current
steps, not a scenario generator with fault injection -- there is no fault model for a PMSM in this
project. Used directly by viz/dashboard.py's "Plano i_d-i_q PMSM" tab.
"""

import numpy as np

from driveflow.sim import (
    ContB6BridgeConverter,
    EulerSolver,
    IdealVoltageSupply,
    PermanentMagnetSynchronousMotor,
    PolynomialStaticLoad,
    SynchronousMotorSystem,
)

TAU = 1e-4
#: PermanentMagnetSynchronousMotor's own default parameters (GEM's real spec, not re-derived) --
#: l_d != l_q, a genuinely salient motor, so MTPA is a real curve (not just i_d=0).
MOTOR_PARAMS = dict(p=3, l_d=0.37e-3, l_q=1.2e-3, r_s=18e-3, psi_p=66e-3)
I_LIM = 400.0
U_SUPPLY = 330.0  # 10% headroom over the motor's own u limit (300V), same ratio as the DC case
#: Load inertia large enough that even the highest-torque MTPA step (~370Nm at i_s=390A) doesn't
#: spin the rotor away from ~0 within a short current-step window -- see docs/macro_fase_C1... no,
#: this predates Fase C; found by direct simulation earlier this session (j_load=0.01 let the
#: rotor run away mid-step; j_load=5.0 keeps omega_el's decoupling term negligible for ~30ms).
LOAD_INERTIA = 5.0


def mtpa_id_iq(i_s: float, mp: dict = MOTOR_PARAMS) -> tuple:
    """Closed-form MTPA locus (id<=0 branch) for a salient PMSM, generalizing GEM's own
    PermanentMagnetSynchronousMotor._torque_limit() formula (only evaluated there at the nominal
    current) to any current magnitude i_s. Verified against that method's own result at i_nom."""
    ld, lq, psi_p = mp["l_d"], mp["l_q"], mp["psi_p"]
    p_ = psi_p / (2 * (ld - lq))
    q_ = -(i_s**2) / 2
    i_d = -p_ / 2 - np.sqrt((p_ / 2) ** 2 - q_)
    i_q = np.sqrt(max(i_s**2 - i_d**2, 0.0))
    return i_d, i_q


def torque_of(i_d: float, i_q: float, mp: dict = MOTOR_PARAMS) -> float:
    return 1.5 * mp["p"] * (mp["psi_p"] + (mp["l_d"] - mp["l_q"]) * i_d) * i_q


class DqCurrentController:
    """Magnitude-optimum PI per axis + cross-coupling decoupling feedforward + anti-windup --
    same design philosophy as control/classical/pi_controller.py's current loop, generalized to
    the PMSM's two current axes (with the omega*L*i cross terms a DC motor doesn't have)."""

    def __init__(self, system, mp: dict = MOTOR_PARAMS):
        self.mp = mp
        tau_eq = system.tau
        self.kp_d = mp["l_d"] / (2 * tau_eq)
        self.ki_d = mp["r_s"] / (2 * tau_eq)
        self.kp_q = mp["l_q"] / (2 * tau_eq)
        self.ki_q = mp["r_s"] / (2 * tau_eq)
        self.reset()

    def reset(self):
        self.int_d = 0.0
        self.int_q = 0.0

    def control(self, state, system, i_d_ref: float, i_q_ref: float):
        physical = state * system.limits
        i_d = physical[system.CURRENTS_IDX[3]]
        i_q = physical[system.CURRENTS_IDX[4]]
        omega_el = physical[system.OMEGA_IDX] * self.mp["p"]

        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        u_sup = system.supply.u_nominal
        half = u_sup / 2.0
        # Anti-windup: clamp each integrator so ki*int alone never exceeds what the converter can
        # deliver -- without this, a step near the current limit saturates the output for many
        # steps while the integral keeps growing, causing overshoot/sign-flip once it unwinds
        # (found directly while building this: an unclamped version diverged at i_s=380A).
        self.int_d = np.clip(self.int_d + e_d * system.tau, -half / self.ki_d, half / self.ki_d)
        self.int_q = np.clip(self.int_q + e_q * system.tau, -half / self.ki_q, half / self.ki_q)

        u_d = self.kp_d * e_d + self.ki_d * self.int_d - omega_el * self.mp["l_q"] * i_q
        u_q = self.kp_q * e_q + self.ki_q * self.int_q + omega_el * (self.mp["l_d"] * i_d + self.mp["psi_p"])

        action = np.clip(np.array([u_d, u_q]) / half, -1.0, 1.0)
        return action, (i_d, i_q)


def build_pmsm_system(mp: dict = MOTOR_PARAMS, u_supply: float = U_SUPPLY) -> SynchronousMotorSystem:
    load = PolynomialStaticLoad(load_parameter=dict(a=0.0, b=0.0, c=0.0, j_load=LOAD_INERTIA))
    return SynchronousMotorSystem(
        converter=ContB6BridgeConverter(),
        motor=PermanentMagnetSynchronousMotor(motor_parameter=mp),
        load=load,
        supply=IdealVoltageSupply(u_nominal=u_supply),
        ode_solver=EulerSolver(),
        tau=TAU,
        control_space="dq",
    )


def run_current_step(i_d_ref: float, i_q_ref: float, duration_s: float = 0.03, seed: int = 0, mp: dict = MOTOR_PARAMS, u_supply: float = U_SUPPLY) -> list:
    """Runs one current-step transient (system.reset() -> repeatedly control()+simulate()) and
    returns the [(i_d, i_q), ...] trajectory -- the raw material for the id/iq point-cloud plot.
    """
    system = build_pmsm_system(mp=mp, u_supply=u_supply)
    controller = DqCurrentController(system, mp=mp)
    system.seed(np.random.SeedSequence(seed))
    state = system.reset()
    n_steps = int(round(duration_s / TAU))
    traj = []
    for _ in range(n_steps):
        action, (i_d, i_q) = controller.control(state, system, i_d_ref, i_q_ref)
        state = system.simulate(action)
        traj.append((float(i_d), float(i_q)))
    return traj


def generate_mtpa_vs_naive_cloud(i_s_values=None, duration_s: float = 0.03, subsample: int = 15, mp: dict = MOTOR_PARAMS, i_lim: float = I_LIM, u_supply: float = U_SUPPLY) -> dict:
    """Runs both policies (MTPA and naive i_d=0) across a sweep of current magnitudes and both
    i_q signs, returning the point clouds + the analytic MTPA curve + the current-limit circle --
    everything viz/dashboard.py's PMSM tab needs, generated live rather than baked into a script.
    """
    if i_s_values is None:
        i_s_values = [40, 80, 120, 160, 200, 240, 280, 320, 360, min(390, i_lim - 10)]

    out = {"mtpa": [], "naive": []}
    for i, i_s in enumerate(i_s_values):
        id_mtpa, iq_mtpa_pos = mtpa_id_iq(i_s, mp=mp)
        for sign in (1, -1):
            iq_mtpa = sign * iq_mtpa_pos
            traj = run_current_step(id_mtpa, iq_mtpa, duration_s=duration_s, seed=i * 2 + (0 if sign > 0 else 1), mp=mp, u_supply=u_supply)
            out["mtpa"].extend(traj[::subsample])

            iq_naive = sign * i_s
            traj2 = run_current_step(0.0, iq_naive, duration_s=duration_s, seed=i * 2 + (0 if sign > 0 else 1) + 100, mp=mp, u_supply=u_supply)
            out["naive"].extend(traj2[::subsample])

    # Traced as ONE continuous path (up the positive branch, back down the negative branch) --
    # not interleaved (id0,+iq0),(id0,-iq0),(id1,+iq1)... which zigzags between the two branches
    # at every step and, rendered as a line, looks like a solid filled wedge instead of a curve
    # (a real bug found by screenshotting the actual rendered chart, not by inspection).
    i_s_range = np.linspace(0, i_lim, 200)
    upper = [mtpa_id_iq(i_s, mp=mp) for i_s in i_s_range]
    lower = [(id_c, -iq_c) for id_c, iq_c in reversed(upper)]
    out["mtpa_curve"] = upper + lower

    theta = np.linspace(0, 2 * np.pi, 200)
    out["limit_circle"] = list(zip(i_lim * np.cos(theta), i_lim * np.sin(theta)))
    return out
