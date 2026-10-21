"""Buck (step-down) DC-DC converter plant -- expansion task E6.

MODEL: state-space AVERAGED model of a SYNCHRONOUS buck (two active switches, no diode), not a
switched model. Over each control period the duty cycle d is held and the switch node is replaced
by its average d·V_in, so the PWM ripple at the switching frequency is not represented: results are
meaningful for dynamics well below f_sw = 1/tau (the default tau = 20 us, i.e. one control update
per switching period at 50 kHz). Because the converter is synchronous, the inductor current may
go negative and the converter never enters discontinuous conduction mode; a diode buck at light
load would.

    L di_L/dt = d·V_in − r_L·i_L − v_C
    C dv_C/dt = i_L − v_C / R

State x = [i_L, v_C] (v_out = v_C; no capacitor ESR). The step is the EXACT zero-order-hold
discretization of this linear model (matrix exponential), so for a constant d the discrete
trajectory equals the continuous one at the sample instants up to floating-point error.

Measured (tests/simulation/test_buck_system.py):
- steady state v_C = d·V_in·R/(R + r_L) (= d·V_in when r_L = 0) to < 1e-9 relative error;
- the transient of a duty step 0 -> 0.5 from rest against scipy.integrate.solve_ivp (Radau,
  rtol 1e-10, atol 1e-12) at every sample over 20 ms (1000 steps, ~14 L-C periods): max
  |Δv_C| = 3.0e-10 V and max |Δi_L| = 3.2e-10 A, towards the 23.8 V / 4.0 A operating point.

The interface mirrors sim/vsc_system.VscSystem: ``reset(...) -> state`` and
``simulate(duty) -> state``. The design follows Fund. BuckAvgEnv (L, C, R, V_in, duty) and
9_Case_Studies_PE/Buck_Design/; it is a reimplementation, no code was copied.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.linalg import expm

TAU = 2e-5


@dataclasses.dataclass(frozen=True)
class BuckParams:
    v_in_v: float = 48.0
    l_h: float = 220e-6
    c_f: float = 220e-6
    r_l_ohm: float = 0.05
    load_resistance_ohm: float = 6.0
    #: inductor current rating -- used by the controller's current limit and the Gym env
    i_max_a: float = 15.0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not (isinstance(v, (int, float)) and np.isfinite(v)):
                raise ValueError(f"BuckParams.{f.name} must be a finite number, got {v!r}")
            if v < 0 or (v == 0 and f.name != "r_l_ohm"):
                raise ValueError(f"BuckParams.{f.name} must be {'>= 0' if f.name == 'r_l_ohm' else '> 0'}, got {v!r}")

    @property
    def resonance_hz(self) -> float:
        return 1.0 / (2 * np.pi * np.sqrt(self.l_h * self.c_f))

    def steady_state(self, duty: float, load_resistance_ohm: float | None = None) -> tuple[float, float]:
        """(i_L, v_C) in steady state for a constant duty."""
        r = self.load_resistance_ohm if load_resistance_ohm is None else load_resistance_ohm
        v = duty * self.v_in_v * r / (r + self.r_l_ohm)
        return v / r, v


@dataclasses.dataclass
class BuckState:
    i_l: float
    v_out: float


def continuous_matrices(p: BuckParams, load_resistance_ohm: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[-p.r_l_ohm / p.l_h, -1.0 / p.l_h], [1.0 / p.c_f, -1.0 / (load_resistance_ohm * p.c_f)]])
    b = np.array([p.v_in_v / p.l_h, 0.0])
    return a, b


def zoh(a: np.ndarray, b: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact ZOH: expm([[A, b], [0, 0]]·tau) = [[Ad, bd], [0, 1]]."""
    m = np.zeros((3, 3))
    m[:2, :2], m[:2, 2] = a, b
    e = expm(m * tau)
    return e[:2, :2], e[:2, 2]


class BuckSystem:
    def __init__(self, params: BuckParams | None = None, tau: float = TAU):
        if not (np.isfinite(tau) and tau > 0):
            raise ValueError(f"tau must be > 0, got {tau!r}")
        self.params = params or BuckParams()
        self.tau = tau
        self.load_resistance_ohm = None
        self.set_load(self.params.load_resistance_ohm)
        self.state: BuckState = None

    def set_load(self, load_resistance_ohm: float) -> None:
        """Changes R from the next step on (a load step); the matrices are re-discretized."""
        if not (np.isfinite(load_resistance_ohm) and load_resistance_ohm > 0):
            raise ValueError(f"load resistance must be > 0, got {load_resistance_ohm!r}")
        if load_resistance_ohm != self.load_resistance_ohm:
            self.load_resistance_ohm = float(load_resistance_ohm)
            self._ad, self._bd = zoh(*continuous_matrices(self.params, self.load_resistance_ohm), self.tau)

    def reset(self, i_l: float = 0.0, v_out: float = 0.0) -> BuckState:
        self.state = BuckState(float(i_l), float(v_out))
        return self.state

    def simulate(self, duty: float) -> BuckState:
        """One period with duty held (clipped to [0, 1], the physical range)."""
        d = min(1.0, max(0.0, float(duty)))
        s, ad, bd = self.state, self._ad, self._bd
        i_l = ad[0, 0] * s.i_l + ad[0, 1] * s.v_out + bd[0] * d
        v = ad[1, 0] * s.i_l + ad[1, 1] * s.v_out + bd[1] * d
        self.state = BuckState(float(i_l), float(v))
        return self.state
