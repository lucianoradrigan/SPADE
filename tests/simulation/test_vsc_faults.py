"""Degraded-capacitor VSC plant (sim/vsc_faults.py) -- TODO_CLAUDE_CODE.md A4."""

import numpy as np
import pytest

from driveflow.control.dpc.loss import ADF, BDF
from driveflow.sim.vsc_faults import DegradedVscSystem, degraded_matrices
from driveflow.sim.vsc_params import CAPACITANCE_F, SAMPLE_PERIOD_S
from driveflow.sim.vsc_system import VscSystem

pytestmark = pytest.mark.simulation


def test_nominal_parameters_reproduce_the_identified_matrices():
    ad, bd = degraded_matrices(1.0, 0.0)
    np.testing.assert_allclose(ad, ADF, rtol=1e-6)
    np.testing.assert_allclose(bd, BDF, rtol=1e-6)


def test_nominal_degraded_plant_tracks_vscsystem():
    nominal, degraded = VscSystem(8.0064), DegradedVscSystem(8.0064)
    a, b = nominal.reset(), degraded.reset()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        u = rng.normal(size=2) * 60
        a, b = nominal.simulate(*u), degraded.simulate(*u)
    # ADF/BDF are float32 constants; the ZOH here is float64 -- 1.6e-6 relative was measured.
    np.testing.assert_allclose([b.i_f_real, b.i_f_imag, b.vc_real, b.vc_imag], [a.i_f_real, a.i_f_imag, a.vc_real, a.vc_imag], rtol=1e-4, atol=1e-4)


def test_smaller_capacitance_raises_the_lc_resonance():
    def resonance(c_scale):
        ad, _ = degraded_matrices(c_scale, 0.0)
        return np.max(np.abs(np.angle(np.linalg.eigvals(ad)))) / SAMPLE_PERIOD_S

    w_nom, w_80 = resonance(1.0), resonance(0.8)
    # Undamped LC resonance 1/sqrt(L C): x 1/sqrt(0.8) = 1.118 for 80 % C.
    assert w_80 / w_nom == pytest.approx(1 / np.sqrt(0.8), rel=0.01)
    assert w_nom == pytest.approx(1 / np.sqrt(5.02e-3 * CAPACITANCE_F), rel=0.02)


def test_esr_shows_in_the_terminal_voltage():
    plant = DegradedVscSystem(10.0, esr_ohm=0.5)
    plant.reset(i_f_real=4.0, vc_real=30.0)
    # v_t = (v_cap + R_c i_f) / (1 + R_c / R) with v_cap = 30, i_f = 4, R_c = 0.5, R = 10.
    assert plant.state.vc_real == pytest.approx((30.0 + 0.5 * 4.0) / 1.05)


@pytest.mark.parametrize("kwargs", [{"c_scale": 0.0}, {"esr_ohm": -0.1}, {"tau": 0.0}])
def test_rejects_non_physical_degradation(kwargs):
    with pytest.raises(ValueError):
        degraded_matrices(**kwargs)
