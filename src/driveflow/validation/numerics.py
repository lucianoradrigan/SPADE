"""Analytic stability bounds for the explicit (forward) Euler integrator the platform uses at
tau = 1e-4 s, derived from the linearised plant matrices. Used to tell a numerical instability
(spectral radius of I + tau*A above 1 while A itself is Hurwitz) from a physical one (A itself has
an eigenvalue with positive real part).
"""

from __future__ import annotations

import numpy as np


def dc_motor_linear_matrix(r_a: float, l_a: float, psi_e: float, j_total: float, b: float = 0.0) -> np.ndarray:
    """State x = [i, omega]; di/dt = (-r_a i - psi_e omega + u)/l_a, domega/dt = (psi_e i - b omega)/j_total.
    The constant (Coulomb) friction term is an input-like offset and does not enter A."""
    return np.array([[-r_a / l_a, -psi_e / l_a], [psi_e / j_total, -b / j_total]])


def pmsm_dq_matrix(omega_el: float, r_s: float, l_d: float, l_q: float) -> np.ndarray:
    """Electrical dq dynamics at a FIXED electrical speed omega_el (speed held by the load):
    di_d/dt = (-r_s i_d + omega_el l_q i_q + u_d)/l_d, di_q/dt = (-r_s i_q - omega_el l_d i_d - omega_el psi_p + u_q)/l_q."""
    return np.array([[-r_s / l_d, omega_el * l_q / l_d], [-omega_el * l_d / l_q, -r_s / l_q]])


def euler_spectral_radius(a: np.ndarray, tau: float) -> float:
    """Spectral radius of the forward-Euler update matrix I + tau*A (> 1 => the discretisation diverges)."""
    return float(np.max(np.abs(np.linalg.eigvals(np.eye(a.shape[0]) + tau * a))))


def euler_max_stable_speed_pmsm(mp: dict, tau: float, omega_mech_max: float = 2000.0) -> float:
    """Largest mechanical speed (rad/s) at which forward Euler at step tau stays stable on the PMSM's
    open-loop dq electrical dynamics (bisection on euler_spectral_radius == 1)."""
    def rho(w):
        return euler_spectral_radius(pmsm_dq_matrix(w * mp["p"], mp["r_s"], mp["l_d"], mp["l_q"]), tau)

    lo, hi = 0.0, omega_mech_max
    if rho(hi) < 1.0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho(mid) < 1.0 else (lo, mid)
    return lo
