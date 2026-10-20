"""Continuous-control-set MPC for the VSC, ported from DPC4PowerElectronics' data-collection
controller -- TODO_CLAUDE_CODE.md V2.

Source: CCS_MPC.slx, block ``PRN_controller1/MATLAB Function4`` (simulink/stateflow/chart_361.xml),
the only active controller in that model (docs/audit/upstream/dpc4pe_parameters.md, S-10). It is
the controller whose trajectories became Data4train.mat, so it is the natural yardstick for what
the DPC network was meant to learn.

What is ported as is:
- the plant model ADF/BDF (identical to control/dpc/loss.py);
- each alpha/beta axis is solved separately, with B1 = BDF[:, 0] (v_o) and B2 = BDF[:, 1]
  (i_load) and the measured load current held constant over the horizon;
- horizon N = 5 and per-step weight Q = diag(2, 1) on [i_f, v_c];
- the per-step target [i_load - C*w*v_ref_imag, v_ref_real] on the real axis and
  [i_load + C*w*v_ref_real, v_ref_imag] on the imaginary axis, with C = 15 uF, w = 2*pi*50;
- the unconstrained condensed QP solved in closed form, U = -H^-1 Psi1' Q (Phi x + Psi2 a - c),
  applying only U[0];
- optionally, the downstream per-axis clamp (Saturation6/9 = +-134 V in CCS_MPC.slx).

What is adapted to SPADE's plant, and why:
- **No one-step delay compensation.** Upstream predicts the state one step ahead with the
  previous action because the Simulink model applies each action one sample late (a Delay block
  and SVPWM). SPADE's averaged plant (sim/vsc_system.py) applies v_o in the same step, so
  compensating for a delay that does not exist would mis-predict.
- **Exact future references instead of extrapolation.** Upstream extrapolates future references
  with 3*x_k - 3*x_(k-1) + x_(k-2) and targets the references 3 to 7 steps ahead. SPADE knows the
  rotating reference exactly, and uses the same alignment as the DPC network and SPADE's tracking
  metric: the i-th predicted state targets reference k+i-1 (reference.horizon(k)).
- Upstream's imaginary-axis target list repeats v_ref4 where v_ref5 is expected (``c_target6b``);
  that typo is not reproduced.

So this is "ported from MATLAB Function4, adapted to the delay-free averaged plant", not a
bit-exact reproduction of the Simulink run.
"""

from __future__ import annotations

import numpy as np

from driveflow.control.dpc.loss import ADF, BDF, CAPACITANCE_F, OMEGA_REF_RAD_S

#: Per-axis clamp of the active upstream controller (CCS_MPC.slx Saturation6/9), ~ 2*Vdc/3 with
#: Vdc = 200 V.
UPSTREAM_VOLTAGE_LIMIT_V = 134.0
UPSTREAM_HORIZON = 5
UPSTREAM_CURRENT_WEIGHT = 2.0


def condensed_matrices(adf: np.ndarray, bdf: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phi (2N x 2), Psi1 (2N x N) for the input, Psi2 (2N x N) for the load current, so that the
    stacked predictions are X = Phi x0 + Psi1 U + Psi2 A."""
    n = horizon
    b1, b2 = bdf[:, :1], bdf[:, 1:]
    phi = np.zeros((2 * n, 2))
    psi1 = np.zeros((2 * n, n))
    psi2 = np.zeros((2 * n, n))
    powers = [np.eye(2)]
    for _ in range(n):
        powers.append(powers[-1] @ adf)
    for i in range(1, n + 1):
        phi[2 * (i - 1) : 2 * i] = powers[i]
        for j in range(1, i + 1):
            psi1[2 * (i - 1) : 2 * i, j - 1 : j] = powers[i - j] @ b1
            psi2[2 * (i - 1) : 2 * i, j - 1 : j] = powers[i - j] @ b2
    return phi, psi1, psi2


class VscCcsMpc:
    """Same reset()/control(state, reference, k) interface as control/dpc/controller.py's
    DpcController, so both run in the same closed-loop harness."""

    def __init__(
        self,
        r_ohm: float,
        horizon: int = UPSTREAM_HORIZON,
        current_weight: float = UPSTREAM_CURRENT_WEIGHT,
        voltage_weight: float = 1.0,
        voltage_limit_v: float | None = None,
        capacitance_f: float = CAPACITANCE_F,
        omega_ref_rad_s: float = OMEGA_REF_RAD_S,
        adf: np.ndarray = ADF,
        bdf: np.ndarray = BDF,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if voltage_limit_v is not None and voltage_limit_v <= 0:
            raise ValueError(f"voltage_limit_v must be positive or None, got {voltage_limit_v}")
        self.r = r_ohm
        self.horizon = horizon
        self.voltage_limit_v = voltage_limit_v
        self.cw = capacitance_f * omega_ref_rad_s
        adf, bdf = np.asarray(adf, dtype=np.float64), np.asarray(bdf, dtype=np.float64)
        self.phi, psi1, self.psi2 = condensed_matrices(adf, bdf, horizon)
        q = np.kron(np.eye(horizon), np.diag([current_weight, voltage_weight]))
        h = psi1.T @ q @ psi1
        # U* = -H^-1 Psi1' Q e, with e = Phi x + Psi2 a - c. Only the first row is applied.
        self._gain = np.linalg.solve(h, psi1.T @ q)[0]

    def reset(self):
        pass  # no memory: the delay compensation that needed the previous action is not ported

    def _axis(self, x0: np.ndarray, i_load: float, targets: np.ndarray) -> float:
        e = self.phi @ x0 + self.psi2 @ np.full(self.horizon, i_load) - targets
        return float(-self._gain @ e)

    def control(self, state, reference, k: int) -> tuple[float, float]:
        refs = np.asarray(reference.horizon(k, horizon=self.horizon), dtype=np.float64)  # (N, 2)
        i_load_r, i_load_i = state.vc_real / self.r, state.vc_imag / self.r
        tgt_r = np.column_stack([i_load_r - self.cw * refs[:, 1], refs[:, 0]]).ravel()
        tgt_i = np.column_stack([i_load_i + self.cw * refs[:, 0], refs[:, 1]]).ravel()
        u_r = self._axis(np.array([state.i_f_real, state.vc_real]), i_load_r, tgt_r)
        u_i = self._axis(np.array([state.i_f_imag, state.vc_imag]), i_load_i, tgt_i)
        if self.voltage_limit_v is not None:
            u_r = float(np.clip(u_r, -self.voltage_limit_v, self.voltage_limit_v))
            u_i = float(np.clip(u_i, -self.voltage_limit_v, self.voltage_limit_v))
        return u_r, u_i
