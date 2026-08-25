"""Formalizes the dashboard's "off-distribution robustness probes" (viz/dashboard.py Fase B
sidebar: load resistance, reference magnitude, reference frequency sliders) into a fixed,
automated regression grid -- Patch 8 Sec. 7. The dashboard's interactive exploration stays as-is;
this is the same closed-loop simulation (VscSystem + DpcController + RotatingReference, the same
physics/controller the dashboard's "Generate DPC scenario" button runs) evaluated once per grid
point, with thresholds derived from behavior actually measured against the shipped v3 checkpoint,
so a change to the checkpoint or the VSC plant that meaningfully degrades tracking anywhere in the
dashboard's explorable range fails a test instead of only being noticed by clicking around.

Grid design: a "star" around the on-distribution point (R, magnitude, omega all at their exact
training values) -- vary ONE axis at a time to its dashboard slider's min/max, holding the other
two at the on-distribution default. This covers the full explorable range of each of the 3
sliders (viz/dashboard.py: R in [1.0, 20.0]ohm, magnitude in [10.0, 150.0]V, omega in
[50.0, 700.0]rad/s) without a full cross-product grid (5x4x3=60 runs, slow and mostly redundant --
the dashboard itself exposes these as 3 independent sliders, so a one-axis-at-a-time probe matches
how a user actually explores it, not a hypothetical worst-case combination of all three at once).

Thresholds are NOT arbitrary: each carries >=1.4x margin over the settled RMSE actually measured
against dpc_trained_v3_closed_loop.weights.h5 (n_steps=2000, tau=1e-4s, first 50 steps excluded as
transient) when this file was written, in the same spirit as the existing on-distribution
regression check in tests/test_datagen.py (RMSE < 5V over a documented v3 baseline of ~1.18V, see
docs/macro_fase_B2_dpc_deployment.md). If the checkpoint or plant model changes and a threshold
here needs updating, re-measure rather than loosening blindly -- these numbers are a snapshot of
v3's actual behavior, not a design target.

KNOWN, DOCUMENTED LIMITATION found while building this grid (not previously tracked anywhere):
the closed-loop simulation DIVERGES (state grows unbounded to NaN within ~2000 steps) for R
roughly in [1.0, 3.0]ohm -- a real, currently-reachable slice of the dashboard's own R slider
range (which allows down to 1.0ohm). This is not a test-infrastructure bug: VscSystem's ADF/BDF
discrete-time model goes unstable in closed loop as i_load = vc/R grows large at small R, and
nothing in the dashboard currently warns the user or clips the slider before this point -- see
TestKnownDivergenceAtLowR below and the Patch 8 summary this file was added under.
"""

import numpy as np
import pytest

from driveflow.control.dpc.controller import DpcController
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V, RotatingReference
from driveflow.datagen.runner import TAU, _DPC_WEIGHTS_PATH, _VSC_R_OHM
from driveflow.sim.vsc_system import VscSystem

N_STEPS = 2000
WARMUP_STEPS = 50  # first ~5ms: startup transient from state=0, excluded from "settled" RMSE


def _settled_rmse(r_ohm: float, magnitude_v: float, omega_rad_s: float, n_steps: int = N_STEPS) -> float:
    """Runs the closed loop and returns the settled-window RMSE (V), or np.nan if the simulation
    diverges (a non-finite tracking error at any step) -- diverging is a real, distinct outcome
    from "large but finite RMSE," not something to let a NaN silently propagate into an assert."""
    system = VscSystem(load_resistance_ohm=r_ohm, tau=TAU)
    controller = DpcController(_DPC_WEIGHTS_PATH, r_ohm=r_ohm)
    reference = RotatingReference(tau=TAU, magnitude_v=magnitude_v, omega_rad_s=omega_rad_s)
    state = system.reset()
    controller.reset()

    errors = np.empty(n_steps)
    for k in range(n_steps):
        v_o_real, v_o_imag = controller.control(state, reference, k)
        state = system.simulate(v_o_real, v_o_imag)
        vref_real, vref_imag = reference.at_step(k)
        errors[k] = np.hypot(state.vc_real - vref_real, state.vc_imag - vref_imag)
        if not np.isfinite(errors[k]):
            return float("nan")
    return float(np.sqrt(np.mean(errors[WARMUP_STEPS:] ** 2)))


R_DEFAULT, MAG_DEFAULT, OMEGA_DEFAULT = float(_VSC_R_OHM), float(REFERENCE_MAGNITUDE_V), float(GRID_OMEGA_RAD_S)

#: (label, r_ohm, magnitude_v, omega_rad_s, max_acceptable_rmse_v). max_acceptable_rmse_v carries
#: >=1.4x margin over what was actually measured against v3 -- see module docstring.
GRID = [
    ("on-distribution (baseline)", R_DEFAULT, MAG_DEFAULT, OMEGA_DEFAULT, 3.0),  # measured ~1.18V
    ("R at slider max (20.0 ohm)", 20.0, MAG_DEFAULT, OMEGA_DEFAULT, 21.0),  # measured ~14.7V
    ("magnitude at slider min (10V)", R_DEFAULT, 10.0, OMEGA_DEFAULT, 3.0),  # measured ~1.75V
    ("magnitude at slider max (150V)", R_DEFAULT, 150.0, OMEGA_DEFAULT, 12.0),  # measured ~8.0V
    ("omega at slider min (50 rad/s)", R_DEFAULT, MAG_DEFAULT, 50.0, 6.0),  # measured ~3.4V
    ("omega at slider max (700 rad/s)", R_DEFAULT, MAG_DEFAULT, 700.0, 6.0),  # measured ~3.9V
]


class TestRobustnessGrid:
    @pytest.mark.parametrize("label,r_ohm,magnitude_v,omega_rad_s,max_rmse_v", GRID, ids=[g[0] for g in GRID])
    def test_settled_rmse_within_threshold(self, label, r_ohm, magnitude_v, omega_rad_s, max_rmse_v):
        rmse = _settled_rmse(r_ohm, magnitude_v, omega_rad_s)
        assert np.isfinite(rmse), (
            f"{label} (R={r_ohm}ohm, |v_ref|={magnitude_v}V, omega={omega_rad_s}rad/s): closed loop "
            f"diverged (non-finite tracking error) -- this grid point was expected to stay finite, "
            f"unlike the documented low-R divergence region (see TestKnownDivergenceAtLowR)."
        )
        assert rmse < max_rmse_v, (
            f"{label} (R={r_ohm}ohm, |v_ref|={magnitude_v}V, omega={omega_rad_s}rad/s): settled RMSE "
            f"{rmse:.3f}V exceeds the {max_rmse_v}V threshold -- a real tracking-quality regression, "
            f"not test noise (the threshold already carries >=1.4x margin over what v3 measured at "
            f"this condition when this test was written)."
        )


class TestKnownDivergenceAtLowR:
    """R in [1.0, 3.0]ohm is within the dashboard's own slider range (1.0-20.0ohm) but makes the
    closed loop diverge -- see module docstring. These assertions exist so this stays a tracked,
    intentional finding: if a future plant/controller change makes low-R stable, THIS test should
    start failing (isfinite becomes true) and should be revisited/loosened deliberately, not have
    its assumption silently bit-rot unnoticed either way."""

    @pytest.mark.parametrize("r_ohm", [1.0, 1.5, 2.0, 2.5, 3.0])
    def test_low_r_currently_diverges(self, r_ohm):
        rmse = _settled_rmse(r_ohm, MAG_DEFAULT, OMEGA_DEFAULT)
        assert not np.isfinite(rmse), (
            f"R={r_ohm}ohm no longer diverges (settled RMSE={rmse:.3f}V) -- this was a documented "
            "instability at the time this test was written (Patch 8 Sec. 7). If a plant/controller "
            "change genuinely fixed it, update this test (and the module docstring, and consider "
            "whether the dashboard's R slider minimum should change) rather than just deleting the "
            "assertion."
        )
