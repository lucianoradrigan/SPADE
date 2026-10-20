"""Native linear MPC for a DcMotorSystem -- Patch 10 (see docs/patch10_implementacion_mpc.md):
resolves the Fase A / INSTRUCTIONS.md discrepancy where "PI/MPC" was listed as delivered but only
PI existed (`control/mpc/` was an empty `.gitkeep`).

Same physical model PICascadeController's speed loop already commits to (see that module's
docstring): the current loop is fast enough (l_a/r_a ~ 100x smaller than the mechanical time
constant j_total*r_a/psi_e**2) that friction (PolynomialStaticLoad's a/b/c coefficients) is
treated as a disturbance rejected by closed-loop feedback, not as part of the design model -- so
this controller does not need to read load_parameter at all (and works unmodified whether
system.mechanical_load is a bare PolynomialStaticLoad or a BearingFaultLoad wrapping one, since
only system.mechanical_load.j_total is used, a property both expose). Unlike PI, which decomposes
the problem into two hand-tuned cascaded loops, this predicts the full 2-state [i, omega] response
over a receding horizon using the plant's own linear model and solves a single QP per step for the
control sequence -- textbook linear MPC, not a port of anything.

State-space model (physical units, forward-Euler discretization at system.tau -- matching the
plant's own EulerSolver, not a re-derived approximation of it):
    d(i)/dt     = (-r_a*i - psi_e*omega + u) / l_a
    d(omega)/dt = (psi_e*i) / j_total
"""

import numpy as np
import threadpoolctl
from scipy.optimize import lsq_linear, minimize

_TRACK_OMEGA = np.array([0.0, 1.0])
_TRACK_CURRENT = np.array([1.0, 0.0])


def _box_qp(h: np.ndarray, g: np.ndarray, lb: float, ub: float, u0: np.ndarray | None = None) -> np.ndarray | None:
    """min 0.5 u'Hu + g'u subject to lb <= u <= ub, H positive definite. Rewritten as the bounded
    least-squares problem min |L'u + L^-1 g|^2 (H = L L') and solved exactly by scipy's BVLS
    (Stark & Parker). Returns None if the factorization or the solver fails."""
    try:
        chol = np.linalg.cholesky(h)
    except np.linalg.LinAlgError:
        return None
    rhs = -np.linalg.solve(chol, g)
    res = lsq_linear(chol.T, rhs, bounds=(lb, ub), method="bvls", tol=1e-12)
    return np.clip(res.x, lb, ub) if res.success else None  # BVLS can return -5e-17 for a bound at 0


class MpcController:
    """Speed- or torque-tracking linear MPC for a DcMotorSystem, mirroring PICascadeController's
    interface (reset()/control()/control_torque()) so it is a drop-in alternative controller_type
    for the same plant (see datagen/scenario.py's _VALID_PAIRS)."""

    def __init__(
        self,
        system,
        horizon: int = 20,
        q_track: float = 1.0,
        r_effort: float = 1e-6,
        q_current_penalty: float = 50.0,
        current_safety_margin: float = 0.85,
        qp_solver: str = "exact",
    ):
        """
        Args:
            system: A DcMotorSystem instance (already constructed -- this controller reads its
                parameters, it does not own the plant, same convention as PICascadeController).
            horizon: Prediction/control horizon, in control steps (tau each). 20 steps at
                tau=1e-4s covers ~1.7 electrical time constants (l_a/r_a ~= 1.19ms) -- enough for
                the QP to see the current dynamics it's actually deciding over; the (much slower)
                mechanical response is handled by receding-horizon replanning every step, not by
                the horizon spanning the full settling time.
            q_track: Weight on (tracked_state - reference)**2 at every horizon step.
            r_effort: Weight on u**2 (control effort) -- kept small; this plant's real limits are
                the converter voltage bound and the current safety cap below, not effort as such.
            q_current_penalty: Weight on a soft (smooth, quadratic-beyond-the-limit) penalty for
                predicted |i_k| exceeding the safety-margined current limit -- there is no
                automatic current clamp in the simulated plant (unlike PICascadeController, which
                clips its explicit i_ref; this controller never forms an i_ref, so the penalty is
                applied directly to the predicted state instead).
            current_safety_margin: Same role as PICascadeController's own parameter -- fraction of
                the current limit the soft penalty is centered on.
            qp_solver: "exact" (default since 2026-10-20; C3, docs/SIMULATION_STABILITY_REPORT.md §5.1):
                the closed-form optimum when no bound or current limit is active, otherwise an
                exact piecewise-QP solution (bounded least squares on the active current-limit
                set), with L-BFGS-B only as a last resort; 8-190x faster on the benchmark runs and
                closer to the true optimum. "lbfgsb": the original solver (scipy L-BFGS-B, at most
                50 iterations, warm-started), kept for comparison.
        """
        if qp_solver not in ("lbfgsb", "exact"):
            raise ValueError(f"qp_solver must be 'lbfgsb' or 'exact', got {qp_solver!r}")
        self.qp_solver = qp_solver
        motor_params = system.electrical_motor.motor_parameter
        self.r_a = motor_params["r_a"]
        self.l_a = motor_params["l_a"]
        self.psi_e = motor_params["psi_e"]
        self.j_total = system.mechanical_load.j_total
        self.tau = system.tau
        self.n = horizon
        self.q_track = q_track
        self.r_effort = r_effort
        self.q_current_penalty = q_current_penalty

        self.limits = system.limits
        self.omega_idx = system.OMEGA_IDX
        self.current_idx = system.CURRENTS_IDX[0]
        self.u_sup_idx = system.U_SUP_IDX[0]
        self.i_max = self.limits[self.current_idx] * current_safety_margin

        ac = np.array([[-self.r_a / self.l_a, -self.psi_e / self.l_a], [self.psi_e / self.j_total, 0.0]])
        bc = np.array([[1.0 / self.l_a], [0.0]])
        self.ad = np.eye(2) + self.tau * ac
        self.bd = self.tau * bc

        n = self.n
        # x_1..x_n = Sx @ x0 + Su @ U (U = [u_0, ..., u_{n-1}]), block rows of 2 (i, omega) per step.
        self.sx = np.zeros((2 * n, 2))
        self.su = np.zeros((2 * n, n))
        a_power = np.eye(2)
        for k in range(n):
            a_power = self.ad @ a_power
            self.sx[2 * k : 2 * k + 2] = a_power
            b_power = self.bd
            for j in range(k, -1, -1):
                self.su[2 * k : 2 * k + 2, j] = b_power[:, 0]
                b_power = self.ad @ b_power

        self._sel_omega = self._block_select(_TRACK_OMEGA)
        self._sel_current = self._block_select(_TRACK_CURRENT)
        self._m_omega = self._sel_omega @ self.su  # (n, n): predicted omega trajectory, linear part in U
        self._m_current = self._sel_current @ self.su  # (n, n): predicted current trajectory, linear part in U
        self._i_pred_slope = self._sel_current @ self.su  # same matrix, named for the soft-penalty use below
        self._sx_omega = self._sel_omega @ self.sx  # (n, 2): free response of the tracked state
        self._sx_current = self._sel_current @ self.sx

        # Closed-form solution of the QP without bounds or current penalty (C3): the minimizer of
        # q*|m u + y0|^2 + r*|u|^2 is u* = -K y0 with K = (q m'm + r I)^-1 q m'. When u* already
        # satisfies the voltage bounds and the current limit, the penalty and its gradient are zero
        # there, so u* is the exact optimum of the full convex problem and L-BFGS-B is skipped.
        eye = np.eye(n)
        self._h_omega = q_track * self._m_omega.T @ self._m_omega + r_effort * eye
        self._h_current = q_track * self._m_current.T @ self._m_current + r_effort * eye
        self._k_omega = np.linalg.solve(self._h_omega, q_track * self._m_omega.T)
        self._k_current = np.linalg.solve(self._h_current, q_track * self._m_current.T)
        self.closed_form_steps = 0
        self.box_qp_steps = 0
        self.iterative_steps = 0

        self._warm_start = None

    def _block_select(self, c: np.ndarray) -> np.ndarray:
        """(n, 2n) matrix picking out c @ x_k for each of the n horizon steps, block-diagonal."""
        sel = np.zeros((self.n, 2 * self.n))
        for k in range(self.n):
            sel[k, 2 * k : 2 * k + 2] = c
        return sel

    def reset(self):
        self._warm_start = None

    def _solve(self, x0: np.ndarray, sel: np.ndarray, m: np.ndarray, ref: float, u_sup: float) -> float:
        """Solves the horizon's QP for a reference on the state selected by `sel`/`m`
        (omega or current -- see control()/control_torque()) and returns only the first step's
        voltage command (standard receding-horizon deployment, same idea as DpcController)."""
        tracking_omega = m is self._m_omega
        y0_track = (self._sx_omega if tracking_omega else self._sx_current) @ x0 - ref
        i_pred_x0 = self._sx_current @ x0  # predicted-current offset, for the soft safety penalty

        if self.qp_solver == "lbfgsb":
            return self._solve_lbfgsb(y0_track, i_pred_x0, m, u_sup)
        u_star = -((self._k_omega if tracking_omega else self._k_current) @ y0_track)
        current_ok = lambda u: np.all(np.abs(i_pred_x0 + self._i_pred_slope @ u) <= self.i_max)  # noqa: E731
        if np.all(u_star >= 0.0) and np.all(u_star <= u_sup) and current_ok(u_star):
            self.closed_form_steps += 1
            self._warm_start = np.concatenate([u_star[1:], u_star[-1:]])
            return float(u_star[0])
        # Otherwise the soft current penalty q_pen * sum(max(|i_k| - i_max, 0)^2) is piecewise
        # quadratic: for a fixed set A of horizon steps that exceed the limit (with their signs) the
        # problem is a box-constrained QP, solved exactly by _box_qp. Iterate on A until it stops
        # changing; the result then satisfies the KKT conditions of the full convex problem.
        h0 = self._h_omega if tracking_omega else self._h_current
        m_track = self._m_omega if tracking_omega else self._m_current
        b0 = self.q_track * (m_track.T @ y0_track)
        u = np.clip(u_star, 0.0, u_sup)
        active = None
        for _ in range(20):
            i_pred = i_pred_x0 + self._i_pred_slope @ u
            sign = np.where(i_pred > self.i_max, 1.0, np.where(i_pred < -self.i_max, -1.0, 0.0))
            if active is not None and np.array_equal(sign, active):
                self.box_qp_steps += 1
                self._warm_start = np.concatenate([u[1:], u[-1:]])
                return float(u[0])
            active = sign
            rows = self._i_pred_slope[sign != 0]
            target = i_pred_x0[sign != 0] - sign[sign != 0] * self.i_max
            h = h0 + self.q_current_penalty * rows.T @ rows
            b = b0 + self.q_current_penalty * rows.T @ target
            u_new = _box_qp(h, b, 0.0, u_sup, u)
            if u_new is None:
                break
            u = u_new
        self.iterative_steps += 1
        return self._solve_lbfgsb(y0_track, i_pred_x0, m, u_sup)

    def _solve_lbfgsb(self, y0_track: np.ndarray, i_pred_x0: np.ndarray, m: np.ndarray, u_sup: float) -> float:
        """The original solver: L-BFGS-B on the full cost, warm-started from the shifted previous
        solution."""

        def cost_and_grad(u):
            r_track = m @ u + y0_track
            i_pred = i_pred_x0 + self._i_pred_slope @ u
            excess = np.maximum(np.abs(i_pred) - self.i_max, 0.0) * np.sign(i_pred)
            cost = (
                self.q_track * float(r_track @ r_track)
                + self.r_effort * float(u @ u)
                + self.q_current_penalty * float(np.sum(excess**2))
            )
            grad = 2 * self.q_track * (m.T @ r_track) + 2 * self.r_effort * u
            grad += 2 * self.q_current_penalty * (self._i_pred_slope.T @ excess)
            return cost, grad

        x_init = self._warm_start if self._warm_start is not None else np.full(self.n, u_sup / 2.0)
        # BLAS auto-threading is actively harmful here: every L-BFGS-B iteration does several
        # matmuls on n x n (~20x20) arrays, and thread-pool spin-up/sync overhead for an array
        # that small dwarfs the actual FLOPs -- measured ~100x slower than single-threaded on
        # this machine (multi-second single control step vs. ~6ms) without this.
        with threadpoolctl.threadpool_limits(limits=1):
            result = minimize(
                cost_and_grad,
                x_init,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, u_sup)] * self.n,
                options={"maxiter": 50},
            )
        u_opt = result.x
        self._warm_start = np.concatenate([u_opt[1:], u_opt[-1:]])  # shift for next step's warm start
        return float(u_opt[0])

    def control(self, state: np.ndarray, omega_ref: float) -> np.ndarray:
        """
        Args:
            state: Normalized SCMLSystem state (as returned by reset()/simulate()).
            omega_ref: Target mechanical speed (rad/s).

        Returns:
            ndarray(float), shape (1,): duty-cycle action for a Cont*QuadrantConverter, in [0, 1].
        """
        physical = state * self.limits
        x0 = np.array([physical[self.current_idx], physical[self.omega_idx]])
        u_sup = physical[self.u_sup_idx]
        u0 = self._solve(x0, self._sel_omega, self._m_omega, omega_ref, u_sup)
        return np.array([float(np.clip(u0 / u_sup, 0.0, 1.0))])

    def control_torque(self, state: np.ndarray, torque_ref: float) -> np.ndarray:
        """Torque-control variant: tracks i_ref = torque_ref/psi_e directly (same linear
        torque/current relation PICascadeController.control_torque() uses), instead of omega.

        Args:
            state: Normalized SCMLSystem state (as returned by reset()/simulate()).
            torque_ref: Target torque (Nm).

        Returns:
            ndarray(float), shape (1,): duty-cycle action for a Cont*QuadrantConverter, in [0, 1].
        """
        physical = state * self.limits
        x0 = np.array([physical[self.current_idx], physical[self.omega_idx]])
        u_sup = physical[self.u_sup_idx]
        i_ref = torque_ref / self.psi_e
        u0 = self._solve(x0, self._sel_current, self._m_current, i_ref, u_sup)
        return np.array([float(np.clip(u0 / u_sup, 0.0, 1.0))])
