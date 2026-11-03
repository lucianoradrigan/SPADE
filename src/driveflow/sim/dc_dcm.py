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
