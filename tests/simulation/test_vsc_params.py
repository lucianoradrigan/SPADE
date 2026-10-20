"""The identified VSC matrices ADF/BDF are the exact ZOH discretization of a physical L-C filter
(sim/vsc_params.py) -- TODO_CLAUDE_CODE.md V0."""

import numpy as np
import pytest
from scipy.linalg import logm

from driveflow.control.dpc.loss import ADF, BDF, CAPACITANCE_F
from driveflow.sim import vsc_params
from driveflow.sim.vsc_params import discretize_zoh
from driveflow.sim.vsc_system import TAU

pytestmark = pytest.mark.simulation


def test_zoh_of_the_physical_parameters_reproduces_the_identified_matrices():
    adf, bdf = discretize_zoh(5.02e-3, 2.0, 15e-6, 1e-4)
    # ADF/BDF are stored as float32, so float32 rounding (~6e-8 relative) is the floor.
    np.testing.assert_allclose(adf, ADF, rtol=1e-6)
    np.testing.assert_allclose(bdf, BDF, rtol=1e-6)


def test_defaults_are_the_identified_parameters():
    adf, bdf = discretize_zoh()
    np.testing.assert_allclose(adf, ADF, rtol=1e-6)
    np.testing.assert_allclose(bdf, BDF, rtol=1e-6)
    assert vsc_params.CAPACITANCE_F == CAPACITANCE_F
    assert vsc_params.SAMPLE_PERIOD_S == TAU


def test_matrix_logarithm_recovers_the_parameters():
    a = np.real(logm(ADF.astype(np.float64))) / vsc_params.SAMPLE_PERIOD_S
    L = -1.0 / a[0, 1]
    assert L == pytest.approx(5.02e-3, rel=1e-5)
    assert 1.0 / a[1, 0] == pytest.approx(15e-6, rel=1e-5)
    assert -a[0, 0] * L == pytest.approx(2.0, rel=1e-5)
    assert abs(a[1, 1]) < 1e-3  # no shunt conductance across C


def test_smaller_capacitance_changes_the_plant():
    adf_nom, _ = discretize_zoh()
    adf_80, _ = discretize_zoh(C=0.8 * vsc_params.CAPACITANCE_F)
    assert not np.allclose(adf_nom, adf_80)
    # Energy stored in C enters ADF[1, 0] ~ Ts / C to first order.
    assert adf_80[1, 0] == pytest.approx(adf_nom[1, 0] / 0.8, rel=0.05)


@pytest.mark.parametrize("kwargs", [{"L": 0.0}, {"C": -1e-6}, {"R_f": -1.0}, {"Ts": 0.0}])
def test_rejects_non_physical_parameters(kwargs):
    with pytest.raises(ValueError):
        discretize_zoh(**kwargs)
