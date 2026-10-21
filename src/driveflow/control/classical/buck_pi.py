"""Output-voltage controller for sim/buck_system.BuckSystem -- expansion task E6.

Average-current-mode cascade, the same structure as PICascadeController (pi_controller.py) for the
DC motor: an outer PI on v_out produces the inductor current reference, an inner PI on i_L
produces the switch-node voltage, and the duty is that voltage over V_in. Both loops use
conditional anti-windup (the integrator freezes while its output is saturated), the current
reference is capped at the inductor rating, and the inner loop feeds the measured v_out forward
(it is the "back-EMF" of the inductor in L di/dt = d·V_in − r_L·i − v).

Tuning from the plant parameters, not hand-picked numbers:
- inner loop: plant 1/(sL + r_L) after the v_out feedforward; kp_i = L·ω_ci, ki_i = r_L·ω_ci
  (zero cancels the electrical pole), ω_ci = 2π·f_sw/10 by default;
- outer loop: with the inner loop ideal, the plant is 1/(sC + 1/R); kp_v = C·ω_cv,
  ki_v = kp_v·ω_cv/4, ω_cv = ω_ci/5 by default (1 kHz at f_sw = 50 kHz, above the 723 Hz L-C
  resonance: the inner loop has already made the plant first order), plus a load-current
  feedforward v_ref/R_nominal.

Measured on the default BuckParams (tests/control/test_buck_controller.py): 0 -> 24 V start-up
settles to within 2 % in 1.32 ms with a 24.94 V peak (3.9 % overshoot); a 6 -> 3 ohm load step
dips to 22.0 V and is back within 1 % in 1.7 ms; steady-state error is below 1 mV. With the
voltage loop at 500 Hz instead the dip was 20.7 V and the recovery 5.0 ms.
"""

from __future__ import annotations

import numpy as np


class BuckVoltageController:
    def __init__(self, plant, current_bandwidth_hz: float | None = None, voltage_bandwidth_hz: float | None = None):
        p = plant.params
        self.tau = plant.tau
        self.v_in = p.v_in_v
        self.i_max = p.i_max_a
        self.r_nominal = p.load_resistance_ohm
        f_ci = current_bandwidth_hz if current_bandwidth_hz is not None else 1.0 / (10 * plant.tau)
        f_cv = voltage_bandwidth_hz if voltage_bandwidth_hz is not None else f_ci / 5
        w_ci, w_cv = 2 * np.pi * f_ci, 2 * np.pi * f_cv
        self.kp_i, self.ki_i = p.l_h * w_ci, max(p.r_l_ohm, 1e-3) * w_ci
        self.kp_v = p.c_f * w_cv
        self.ki_v = self.kp_v * w_cv / 4
        self.reset()

    def reset(self) -> None:
        self._v_integral = 0.0
        self._i_integral = 0.0
        self.last_i_ref = 0.0

    def control(self, state, v_ref: float) -> float:
        """Duty cycle in [0, 1] for the next period."""
        e_v = v_ref - state.v_out
        i_raw = self.kp_v * e_v + self._v_integral + v_ref / self.r_nominal
        i_ref = float(np.clip(i_raw, -self.i_max, self.i_max))
        if i_ref == i_raw:
            self._v_integral += self.ki_v * e_v * self.tau
        e_i = i_ref - state.i_l
        u_raw = self.kp_i * e_i + self._i_integral + state.v_out
        u = float(np.clip(u_raw, 0.0, self.v_in))
        if u == u_raw:
            self._i_integral += self.ki_i * e_i * self.tau
        self.last_i_ref = i_ref
        return u / self.v_in
