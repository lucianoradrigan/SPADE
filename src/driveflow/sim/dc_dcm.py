"""Optional DC plant with discontinuous conduction of the one-quadrant converter (S-21, decision S21-2).

GEM's ``ContOneQuadrantConverter`` decides the converter output ONCE per control period from the
current at the start of the period: duty * U_sup if i >= 0, and the full supply if i < 0. A real
one-quadrant chopper (transistor + free-wheeling diode) cannot carry a negative current: when the
current falls to zero the diode blocks, the current stays at zero (discontinuous conduction) and the
machine coasts on friction and load until the applied voltage exceeds the back-EMF again. Holding
the full supply for a whole period on a negative current injects energy (finding S-21).

``DcmDcMotorSystem`` keeps every component of ``DcMotorSystem`` (motor, load, supply, the same
averaged duty * U_sup voltage while conducting) and changes only the integration of one period:

- the period is split into ``substeps`` explicit Euler substeps, each evaluated at its own START
  time ``t0 + k*h`` (with one substep this is exactly the default ``EulerSolver()`` update; it does
  NOT copy ``EulerSolver(nsteps>1)``'s clock, which runs one period late -- finding S-17), and the
  converter is re-evaluated in each substep;
- at zero current the converter blocks while the applied voltage does not exceed the back-EMF
  (``u <= e``): the current stays at 0 and only the mechanics advance (zero torque);
- if a substep would take the current below zero, the zero crossing is located by linear
  interpolation inside the substep; the state advances to the crossing, the current is set to 0
  and the rest of the substep advances the mechanics with zero current.

The recorded voltage is the mean terminal voltage over the period (u while conducting, e while
blocked). It is used only with ``Scenario.dc_converter_model = "dcm"``; the default plant is not
touched (decision S21-4).
"""

from __future__ import annotations

import numpy as np

from driveflow.sim.converters import ContOneQuadrantConverter
from driveflow.sim.scml_system import DcMotorSystem


class DcmDcMotorSystem(DcMotorSystem):
    def __init__(self, *args, substeps: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._converter, ContOneQuadrantConverter):
            raise ValueError("the discontinuous-conduction model applies to the one-quadrant converter only")
        if not (isinstance(substeps, int) and substeps >= 1):
            raise ValueError("substeps must be a positive integer")
        self.substeps = substeps
        self.blocked_fraction = 0.0  # fraction of the last period spent with the converter blocked

    def _back_emf(self, state) -> float:
        mp = self._electrical_motor.motor_parameter
        return float(mp["psi_e"] * state[self._omega_ode_idx])

    def simulate(self, action, *_, **__):
        state = np.array(self._ode_solver.y, dtype=float, copy=True)
        self._converter.set_action(action, self._t)
        duty = float(self._converter._current_action[0])
        i_idx = self._ode_currents_idx[0]
        i_in = self._electrical_motor.i_in(state[self._ode_currents_idx])
        u_sup = self._supply.get_voltage(self._t, self._converter.i_sup(i_in))
        u = duty * float(u_sup[0])
        t_end = self._t + self.tau
        h = self.tau / self.substeps
        t = self._t
        u_terminal_sum, blocked = 0.0, 0.0
        for _ in range(self.substeps):
            if state[i_idx] <= 0.0 and u <= self._back_emf(state):
                # diode blocked: no current, only the mechanics advance
                state[i_idx] = 0.0
                d = self._system_equation(t, state, [u])
                state = state + d * h
                state[i_idx] = 0.0
                u_terminal_sum += self._back_emf(state) * h
                blocked += h
            else:
                d = self._system_equation(t, state, [u])
                new = state + d * h
                if new[i_idx] < 0.0:
                    theta = state[i_idx] / (state[i_idx] - new[i_idx])  # in [0, 1): crossing inside the substep
                    cross = state + d * (theta * h)
                    cross[i_idx] = 0.0
                    d0 = self._system_equation(t + theta * h, cross, [u])
                    new = cross + d0 * ((1.0 - theta) * h)
                    new[i_idx] = 0.0
                    u_terminal_sum += u * theta * h + self._back_emf(new) * (1.0 - theta) * h
                    blocked += (1.0 - theta) * h
                else:
                    u_terminal_sum += u * h
                state = new
            t += h
        self._ode_solver.set_initial_value(state, t_end)
        self._t = t_end
        self._k += 1
        self.blocked_fraction = blocked / self.tau
        torque = self._electrical_motor.torque(state[self._motor_ode_idx])
        n_mech = len(self.mechanical_load.state_names)
        motor_state = state[n_mech:]
        self.system_state[:n_mech] = state[:n_mech]
        self.system_state[self.TORQUE_IDX] = torque
        self.system_state[self.CURRENTS_IDX] = motor_state[self._electrical_motor.CURRENTS_IDX]
        self.system_state[self.VOLTAGES_IDX] = [u_terminal_sum / self.tau]
        self.system_state[self.U_SUP_IDX] = u_sup
        return self.system_state / self._limits


class DcmExactDcMotorSystem(DcMotorSystem):
    """Variant B of the efficient discontinuous-conduction converter (``dc_converter_model="dcm_exact"``;
    pre-registration ``docs/audit/assistant/dc_perm_ex_v2_convergence_preregistration.md`` with its
    amendment 1, decision E2 of 2026-11-08). One evaluation per control period, no substeps:

    - electrical: the armature equation L_a di/dt = u - R_a i - e(t) is solved in closed form, with the
      back-EMF LINEAR within the period, e(t) = e0 + e_dot t (e0 = psi_e omega0, e_dot = psi_e
      omega_dot0, omega_dot0 the mechanical derivative at the start state);
    - conduction ends at the first zero of the current (bisection on the closed form to 1e-12 tau), after
      which the diode blocks and i = 0 while u <= e(t); if e(t) falls below u inside the period,
      conduction resumes from i = 0 at t_on = (u - e0) / e_dot;
    - mechanics: Heun (explicit trapezoid) over the period on the mechanical states (omega and the fault
      phases) with the period-average electrical torque psi_e * i_mean and the plant's own load model.
      [CC] implementation choice fixed before any measurement; the pre-registration says "the period-average
      electrical torque and the same load model" and leaves the mechanical integrator open.

    Only for the one-quadrant converter and the permanently excited DC motor (r_a, l_a, psi_e). The
    default plant and ``"dcm"`` are not touched.
    """

    MAX_SEGMENTS = 8
    BISECTION_REL_TOL = 1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._converter, ContOneQuadrantConverter):
            raise ValueError("the discontinuous-conduction model applies to the one-quadrant converter only")
        mp = self._electrical_motor.motor_parameter
        self._r, self._l, self._psi = float(mp["r_a"]), float(mp["l_a"]), float(mp["psi_e"])
        self.blocked_fraction = 0.0

    # --- closed-form armature current while conducting, from (s=0, i=i_start, e=e_start) ----------
    def _current(self, s, i_start, u, e_start, e_dot):
        r, tau_e = self._r, self._l / self._r
        i_ss = (u - e_start) / r
        c = i_start - i_ss - e_dot * tau_e / r
        return i_ss - (e_dot / r) * (s - tau_e) + c * np.exp(-s / tau_e)

    def _charge(self, s, i_start, u, e_start, e_dot):
        """Integral of the current over [0, s] (closed form)."""
        r, tau_e = self._r, self._l / self._r
        i_ss = (u - e_start) / r
        c = i_start - i_ss - e_dot * tau_e / r
        return i_ss * s - (e_dot / r) * (0.5 * s * s - tau_e * s) + c * tau_e * (1.0 - np.exp(-s / tau_e))

    def _first_zero(self, r_len, i_start, u, e_start, e_dot):
        """First s in (0, r_len] where the current reaches 0, or None. The closed form has at most one
        extremum, so the current goes negative inside the interval iff it is negative at the end or at
        that extremum."""
        tau_e = self._l / self._r
        i_ss = (u - e_start) / self._r
        c = i_start - i_ss - e_dot * tau_e / self._r
        probe = r_len
        if c != 0.0:
            x = -(e_dot / self._r) * tau_e / c  # e^{-s/tau_e} at the extremum
            if 0.0 < x < 1.0:
                s_ext = -tau_e * np.log(x)
                if 0.0 < s_ext < r_len and self._current(s_ext, i_start, u, e_start, e_dot) < 0.0:
                    probe = s_ext
        if self._current(probe, i_start, u, e_start, e_dot) >= 0.0:
            return None
        lo, hi = 0.0, probe
        while hi - lo > self.BISECTION_REL_TOL * self.tau:
            mid = 0.5 * (lo + hi)
            if self._current(mid, i_start, u, e_start, e_dot) > 0.0:
                lo = mid
            else:
                hi = mid
        return hi

    def _mech(self, t, mech_state, torque):
        return np.asarray(self._mechanical_load.mechanical_ode(t, mech_state, torque), dtype=float)

    def simulate(self, action, *_, **__):
        state = np.array(self._ode_solver.y, dtype=float, copy=True)
        self._converter.set_action(action, self._t)
        duty = float(self._converter._current_action[0])
        i_idx = self._ode_currents_idx[0]
        n_mech = len(self.mechanical_load.state_names)
        i_in = self._electrical_motor.i_in(state[self._ode_currents_idx])
        u_sup = self._supply.get_voltage(self._t, self._converter.i_sup(i_in))
        u = duty * float(u_sup[0])
        tau, psi = self.tau, self._psi
        i0 = max(float(state[i_idx]), 0.0)
        omega_dot0 = float(self._mech(self._t, state[:n_mech], psi * i0)[self._omega_ode_idx])
        e0, e_dot = psi * float(state[self._omega_ode_idx]), psi * omega_dot0
        t, i, charge, v_int, blocked = 0.0, i0, 0.0, 0.0, 0.0
        conduct_next = False  # a blocked segment that ended inside the period: conduction resumes (no float re-test)
        for _ in range(self.MAX_SEGMENTS):
            if t >= tau:
                break
            e_t = e0 + e_dot * t
            if not conduct_next and i <= 0.0 and u <= e_t:  # diode blocked: i = 0 until e(t) falls below u
                t_on = t + (e_t - u) / (-e_dot) if e_dot < 0.0 else np.inf
                end = min(tau, t_on)
                v_int += e_t * (end - t) + 0.5 * e_dot * (end - t) ** 2
                blocked += end - t
                t, i = end, 0.0
                conduct_next = end < tau
                continue
            conduct_next = False
            r_len = tau - t
            s_zero = self._first_zero(r_len, i, u, e_t, e_dot)
            s = r_len if s_zero is None else s_zero
            charge += self._charge(s, i, u, e_t, e_dot)
            v_int += u * s
            i = 0.0 if s_zero is not None else float(self._current(s, i, u, e_t, e_dot))
            t += s
        else:
            raise RuntimeError("dcm_exact: too many conduction segments in one period")
        i_mean = charge / tau
        torque_mean = psi * i_mean
        x0 = state[:n_mech]
        k1 = self._mech(self._t, x0, torque_mean)
        k2 = self._mech(self._t + tau, x0 + tau * k1, torque_mean)
        state[:n_mech] = x0 + 0.5 * tau * (k1 + k2)
        state[i_idx] = max(i, 0.0)
        t_end = self._t + tau
        self._ode_solver.set_initial_value(state, t_end)
        self._t = t_end
        self._k += 1
        self.blocked_fraction = blocked / tau
        torque = self._electrical_motor.torque(state[self._motor_ode_idx])
        motor_state = state[n_mech:]
        self.system_state[:n_mech] = state[:n_mech]
        self.system_state[self.TORQUE_IDX] = torque
        self.system_state[self.CURRENTS_IDX] = motor_state[self._electrical_motor.CURRENTS_IDX]
        self.system_state[self.VOLTAGES_IDX] = [v_int / tau]
        self.system_state[self.U_SUP_IDX] = u_sup
        return self.system_state / self._limits
