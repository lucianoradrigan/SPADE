"""Capacitor degradation for the VSC plant -- TODO_CLAUDE_CODE.md A4 (INSTRUCTIONS.md D.2, Step 0).

Two synthetic degradation parameters of the filter capacitor, applied to the same L-C model whose
exact ZOH discretization is ADF/BDF (sim/vsc_params.py):
- ``c_scale``: effective capacitance as a fraction of the nominal 15 uF (aging electrolytic and
  film capacitors lose capacitance);
- ``esr_ohm``: an equivalent series resistance in series with the capacitor (it grows with
  aging).

With an ESR R_c, the voltage the controller measures and the load sees is the terminal voltage
v_t = v_cap + R_c * (i_f - i_load), not the internal capacitor voltage. The continuous model,
with state [i_f, v_cap] and input [v_o, i_load], is

    di_f/dt  = (v_o - v_cap - (R_f + R_c) * i_f + R_c * i_load) / L
    dv_cap/dt = (i_f - i_load) / C

and it is discretized with the same exact ZOH as ADF/BDF. With c_scale = 1 and esr_ohm = 0 it
reduces to the nominal plant (checked in tests/simulation/test_vsc_faults.py).

EPISTEMIC WARNING (INSTRUCTIONS.md D.2): no real reference dataset exists for capacitor
degradation in this converter, unlike Paderborn for bearings. Anything built on this module is
validated only against this synthetic definition of the fault, not against independent evidence.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from driveflow.sim.vsc_params import CAPACITANCE_F, FILTER_RESISTANCE_OHM, INDUCTANCE_H, SAMPLE_PERIOD_S
from driveflow.sim.vsc_system import VscState


def degraded_matrices(c_scale: float = 1.0, esr_ohm: float = 0.0, tau: float = SAMPLE_PERIOD_S, l_scale: float = 1.0, rf_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact ZOH (A_d, B_d) of the model above, state [i_f, v_cap], input [v_o, i_load].
    l_scale / rf_scale: the real filter's L and R_f as a fraction of nominal -- parameter
    uncertainty that is NOT a fault (used by the A4 robustness check)."""
    if c_scale <= 0 or esr_ohm < 0 or tau <= 0 or l_scale <= 0 or rf_scale < 0:
        raise ValueError(f"need c_scale, l_scale > 0, esr_ohm, rf_scale >= 0, tau > 0; got {c_scale}, {esr_ohm}, {tau}, {l_scale}, {rf_scale}")
    L, r_f, c = INDUCTANCE_H * l_scale, FILTER_RESISTANCE_OHM * rf_scale, CAPACITANCE_F * c_scale
    a = np.array([[-(r_f + esr_ohm) / L, -1.0 / L], [1.0 / c, 0.0]])
    b = np.array([[1.0 / L, esr_ohm / L], [0.0, -1.0 / c]])
    m = np.zeros((4, 4))
    m[:2, :2], m[:2, 2:] = a, b
    phi = expm(m * tau)
    return phi[:2, :2], phi[:2, 2:]


class DegradedVscSystem:
    """Same reset()/simulate() interface and VscState output as sim/vsc_system.VscSystem, with a
    degraded capacitor. ``state.vc_*`` is the terminal voltage (what the controller measures)."""

    def __init__(self, load_resistance_ohm: float = 8.0064, c_scale: float = 1.0, esr_ohm: float = 0.0, tau: float = SAMPLE_PERIOD_S,
                 l_scale: float = 1.0, rf_scale: float = 1.0):
        self.r = load_resistance_ohm
        self.tau = tau
        self.c_scale, self.esr_ohm = c_scale, esr_ohm
        self._ad, self._bd = degraded_matrices(c_scale, esr_ohm, tau, l_scale, rf_scale)
        self._x = None  # complex [i_f, v_cap]
        self.state: VscState = None

    def _terminal(self, i_f: complex, v_cap: complex) -> complex:
        # v_t = v_cap + R_c (i_f - v_t / R)  ->  v_t = (v_cap + R_c i_f) / (1 + R_c / R)
        return (v_cap + self.esr_ohm * i_f) / (1.0 + self.esr_ohm / self.r)

    def _publish(self) -> VscState:
        i_f, v_cap = self._x
        v_t = self._terminal(i_f, v_cap)
        self.state = VscState(i_f.real, i_f.imag, v_t.real, v_t.imag)
        return self.state

    def reset(self, i_f_real=0.0, i_f_imag=0.0, vc_real=0.0, vc_imag=0.0) -> VscState:
        self._x = np.array([complex(i_f_real, i_f_imag), complex(vc_real, vc_imag)])
        return self._publish()

    def simulate(self, v_o_real: float, v_o_imag: float) -> VscState:
        """One step: x <- A_d x + B_d [v_o, i_load], with i_load = v_t / R held over the step
        (the same convention as VscSystem, which uses the state's v_c / R)."""
        i_load = complex(self.state.vc_real, self.state.vc_imag) / self.r
        u = np.array([complex(v_o_real, v_o_imag), i_load])
        self._x = self._ad @ self._x + self._bd @ u
        return self._publish()
