"""VSC + DPC numerical behaviour beyond tests/test_dpc_robustness_grid.py --
docs/SIMULATION_STABILITY_REPORT.md Sec. 4.

The VSC plant is a fixed identified discrete-time model (ADF/BDF), so there is no integration step
to vary: its numerical questions are boundedness, reproducibility and where instability is
physical (open-loop spectral radius > 1) vs. caused by the controller.
"""

import numpy as np
import pytest

from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM, load_feedback_spectral_radius
from driveflow.validation import run_vsc_closed_loop

pytestmark = pytest.mark.simulation

V_REF = 50.0
WARMUP = 50


def _settled_rmse(r):
    e = np.hypot(r["vc_real"] - r["v_ref_real"], r["vc_imag"] - r["v_ref_imag"])[WARMUP:]
    return float(np.sqrt(np.mean(e**2))) if np.all(np.isfinite(e)) else np.nan


def test_reproducible_run_to_run():
    a, b = run_vsc_closed_loop(duration_s=0.02, seed=3), run_vsc_closed_loop(duration_s=0.02, seed=3)
    assert np.array_equal(a["vc_real"], b["vc_real"]) and np.array_equal(a["i_f_imag"], b["i_f_imag"])


@pytest.mark.parametrize("r_ohm", [MIN_STABLE_LOAD_RESISTANCE_OHM, 8.0064, 20.0])
def test_closed_loop_stays_bounded_over_the_gui_range(r_ohm):
    """GUI slider range is [4, 20] ohm. Bounded = finite and |v_c| never above 3x |v_ref|."""
    r = run_vsc_closed_loop(r_ohm, duration_s=0.2)
    mag = np.hypot(r["vc_real"], r["vc_imag"])
    assert np.all(np.isfinite(mag)) and mag.max() < 3 * V_REF


def test_open_circuit_is_open_loop_stable():
    """R -> infinity (no load) leaves the plant's own spectral radius < 1: any divergence there
    would be the controller's, not physics'."""
    assert load_feedback_spectral_radius(1e9) < 1.0


@pytest.mark.xfail(strict=True, reason="S-05 (open): the DPC network was trained at ONE load R=8.0064 ohm; it is "
                   "stable but tracks poorly away from it (measured settled RMSE 18 V at 4 ohm, 15 V at 20 ohm, "
                   "111 V at 1 kohm vs 1.18 V at 8 ohm). Needs retraining over a range of R.")
@pytest.mark.parametrize("r_ohm", [MIN_STABLE_LOAD_RESISTANCE_OHM, 20.0, 1000.0])
def test_tracking_within_10pct_of_reference_wherever_the_plant_is_stable(r_ohm):
    """Criterion: settled RMSE <= 10 % of |v_ref| (5 V) -- ~4x the on-distribution RMSE (1.18 V,
    docs/macro_fase_B2_dpc_deployment.md) -- at load values where the plant is open-loop stable."""
    assert load_feedback_spectral_radius(r_ohm) < 1.0
    assert _settled_rmse(run_vsc_closed_loop(r_ohm, duration_s=0.2)) <= 0.10 * V_REF


def test_on_distribution_tracking_regression():
    assert _settled_rmse(run_vsc_closed_loop(8.0064, duration_s=0.2)) == pytest.approx(1.18, abs=0.1)
