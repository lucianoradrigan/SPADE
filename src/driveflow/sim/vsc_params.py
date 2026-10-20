"""Physical parameters behind the identified VSC model ADF/BDF (control/dpc/loss.py).

ADF/BDF were pasted from DPC4PowerElectronics' Custom_Loss.txt as bare numbers. They are the exact
zero-order-hold (ZOH) discretization, at Ts = 100 us, of a two-state L-C filter with a series
resistance, state x = [i_f, v_c] and input u = [v_o, i_load]:

    di_f/dt = (v_o - v_c - R_f * i_f) / L
    dv_c/dt = (i_f - i_load) / C

with L = 5.02 mH, R_f = 2.0 ohm and C = 15.0 uF. These values were recovered with
``scipy.linalg.logm(ADF) / Ts`` (the continuous state matrix) plus the ZOH relation for BDF, and
pinned by tests/simulation/test_vsc_params.py: ``discretize_zoh`` reproduces both matrices to
float32 precision. Because the identified C scales with the assumed Ts, landing on exactly the
loss's CAPACITANCE_F = 15 uF confirms Ts = 100 us. That was previously only inferred from
Data4train.mat's phase advance.

The model has two states (one inductor, one capacitor), so it is an L-C filter with a resistive
load, not an LCL filter as earlier docstrings said. The upstream Simulink models agree: the plant
subsystem is ``Inverter_LC_filter1``, with one RL branch (R = Rl = 2, L = 5.02e-3) and one C
branch (C = 15e-6), and Ts = 1e-4 is set in CCS_MPC.slx's InitFcn
(docs/audit/upstream/dpc4pe_parameters.md).

``discretize_zoh`` makes the plant parametric, so a degraded capacitor (smaller C) can be
re-discretized instead of being stuck with the fixed identified matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

#: Filter inductance [H].
INDUCTANCE_H = 5.02e-3
#: Filter series resistance [ohm].
FILTER_RESISTANCE_OHM = 2.0
#: Filter capacitance [F]; equals control/dpc/loss.py CAPACITANCE_F.
CAPACITANCE_F = 15e-6
#: Sample period [s] of ADF/BDF.
SAMPLE_PERIOD_S = 1e-4
#: DC-link voltage of the original converter (CCS_MPC.slx InitFcn ``Vdc = 200``) and the per-axis
#: clamp its active controller applies (Saturation6/9, +-134 V ~ 2*Vdc/3) -- S-09,
#: docs/audit/upstream/dpc4pe_parameters.md. SPADE's plant has no limit unless a Scenario asks.
UPSTREAM_DC_LINK_V = 200.0
UPSTREAM_CONVERTER_LIMIT_V = 134.0


def continuous_lc_model(L: float, R_f: float, C: float) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) of the continuous model above, state [i_f, v_c], input [v_o, i_load]."""
    if min(L, C) <= 0 or R_f < 0:
        raise ValueError(f"need L > 0, C > 0, R_f >= 0; got L={L}, C={C}, R_f={R_f}")
    a = np.array([[-R_f / L, -1.0 / L], [1.0 / C, 0.0]])
    b = np.array([[1.0 / L, 0.0], [0.0, -1.0 / C]])
    return a, b


def discretize_zoh(
    L: float = INDUCTANCE_H,
    R_f: float = FILTER_RESISTANCE_OHM,
    C: float = CAPACITANCE_F,
    Ts: float = SAMPLE_PERIOD_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ZOH discretization (ADF, BDF) of the L-C filter, in float64.

    Uses the augmented-matrix form expm([[A, B], [0, 0]] * Ts) = [[ADF, BDF], [0, I]].
    """
    if Ts <= 0:
        raise ValueError(f"need Ts > 0; got {Ts}")
    a, b = continuous_lc_model(L, R_f, C)
    m = np.zeros((4, 4))
    m[:2, :2], m[:2, 2:] = a, b
    phi = expm(m * Ts)
    return phi[:2, :2], phi[:2, 2:]
