"""The ported upstream CCS-MPC (control/mpc/vsc_ccs_mpc.py) -- TODO_CLAUDE_CODE.md V2,
docs/audit/stability/dpc_vs_mpc.md."""

import numpy as np
import pytest

from driveflow.control.dpc.loss import ADF, BDF
from driveflow.control.dpc.reference import RotatingReference
from driveflow.control.mpc.vsc_ccs_mpc import UPSTREAM_VOLTAGE_LIMIT_V, VscCcsMpc, condensed_matrices
from driveflow.sim.vsc_system import VscSystem

pytestmark = pytest.mark.control

R_NOMINAL = 8.0064
WARMUP = 50


def _settled_rmse(controller, r_ohm=R_NOMINAL, magnitude_v=50.0, n_steps=2000):
    system, reference = VscSystem(load_resistance_ohm=r_ohm), RotatingReference(magnitude_v=magnitude_v)
    state, err, u_max = system.reset(), [], 0.0
    for k in range(n_steps):
        u = controller.control(state, reference, k)
        if k >= WARMUP:
            u_max = max(u_max, abs(u[0]), abs(u[1]))
        state = system.simulate(*u)
        vr = reference.at_step(k)
        err.append(np.hypot(state.vc_real - vr[0], state.vc_imag - vr[1]))
    err = np.asarray(err)
    return float(np.sqrt(np.mean(err[WARMUP:] ** 2))), u_max


def test_condensed_prediction_equals_stepping_the_model():
    adf, bdf = ADF.astype(float), BDF.astype(float)
    phi, psi1, psi2 = condensed_matrices(adf, bdf, 5)
    rng = np.random.default_rng(0)
    x0, u, a = rng.normal(size=2), rng.normal(size=5), rng.normal(size=5)
    x, stacked = x0.copy(), []
    for i in range(5):
        x = adf @ x + bdf @ np.array([u[i], a[i]])
        stacked.extend(x)
    np.testing.assert_allclose(phi @ x0 + psi1 @ u + psi2 @ a, stacked, atol=1e-12)


@pytest.mark.slow
def test_mpc_tracks_the_reference_at_the_nominal_point():
    # Measured 2026-10-20: 0.136 V (N=5), against 1.18 V for DPC v3 and 2.52 V for DPC v4.
    rmse, u_max = _settled_rmse(VscCcsMpc(R_NOMINAL))
    assert rmse < 0.2
    # Once settled the nominal point stays inside the upstream clamp. The start-up step from rest
    # does not: the unclamped MPC asks for ~547 V on its first step.
    assert u_max < UPSTREAM_VOLTAGE_LIMIT_V


def test_clamp_limits_every_axis_and_costs_tracking_only_when_active():
    clamped = VscCcsMpc(R_NOMINAL, voltage_limit_v=UPSTREAM_VOLTAGE_LIMIT_V)
    free = VscCcsMpc(R_NOMINAL)
    rmse_50, _ = _settled_rmse(clamped, n_steps=600)
    assert rmse_50 == pytest.approx(_settled_rmse(free, n_steps=600)[0], rel=1e-9)
    rmse_150, u_max = _settled_rmse(clamped, magnitude_v=150.0, n_steps=600)
    assert u_max <= UPSTREAM_VOLTAGE_LIMIT_V + 1e-12
    # 150 V at 8 ohm needs ~189 V of converter voltage (S-09), so the clamped MPC cannot track it.
    assert rmse_150 > 10 * _settled_rmse(free, magnitude_v=150.0, n_steps=600)[0]


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"voltage_limit_v": 0.0}])
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        VscCcsMpc(R_NOMINAL, **kwargs)
