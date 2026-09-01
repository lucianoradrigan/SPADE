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

ROOT CAUSE, identified and fixed under Patch 9 (Corrección de la divergencia DPC en R bajo): the
closed-loop simulation DIVERGES (state grows unbounded to NaN within ~2000 steps) for R roughly in
[1.0, 3.0]ohm. This is NOT a training-data coverage gap (R is constant at 8.0064ohm across every
row of v1/v2/v3's training data, so no amount of "more data at low R" was ever going to fix it via
this same fine-tuning approach) and NOT a test-infrastructure bug: it is a structural, analytically
verified open-loop instability of VscSystem's own ADF/BDF discrete-time model. Folding the
resistive load's feedback (i_load = vc/R) into the state matrix and computing its spectral radius
as a function of R (see sim.vsc_system.load_feedback_spectral_radius) shows the plant itself --
with NO controller at all -- is unstable below R* ~= 3.3707ohm (spectral radius grows from ~1.25 at
R=3.0ohm to ~12.16 at R=0.5ohm). No receding-horizon network trained the way v1/v2/v3 were (a
single fixed R, no R variation to learn a compensating gain from) can be expected to tame a real
pole of that magnitude -- and even a network that *could* would face an unrelated, unresolved
bootstrapping problem for collecting DAgger-style rollout data there: the current controller
diverges to NaN immediately at those R, so there is no well-formed "current policy's visited
states" to record in the first place.

MITIGATION (implemented, not deferred): sim.vsc_system.MIN_STABLE_LOAD_RESISTANCE_OHM (4.0, ~19%
margin over R*) is now the floor of the dashboard's "Load resistance" slider (viz/dashboard.py) --
the divergent region is only reachable via that slider's "Custom value" override, and doing so
shows a visible warning naming the analytic cause, rather than silently returning NaN with no
explanation. TestKnownDivergenceAtLowR below is kept (the plant genuinely still diverges there --
that is physics, not a bug to "fix" away) but is now backed by TestOpenLoopStabilityThreshold,
which asserts the analytic mechanism itself and WILL fail if MIN_STABLE_LOAD_RESISTANCE_OHM (or a
future Adf/Bdf change) ever stops leaving a safe margin over R* -- i.e. this stays a live
regression check on the mitigation, not a permanent acceptance of the divergence.
"""

import numpy as np
import pytest

from driveflow.control.dpc.controller import DpcController
from driveflow.control.dpc.reference import GRID_OMEGA_RAD_S, REFERENCE_MAGNITUDE_V, RotatingReference
from driveflow.datagen.runner import TAU, _DPC_WEIGHTS_PATH, _VSC_R_OHM
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM, VscSystem, load_feedback_spectral_radius

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
    """R in [1.0, 3.0]ohm is below the plant's own open-loop stability floor (see module docstring
    and TestOpenLoopStabilityThreshold) -- the closed loop is expected to diverge there for
    structural reasons, not a bug. It is no longer reachable via the dashboard's default slider
    (floored at MIN_STABLE_LOAD_RESISTANCE_OHM), only via its 'Custom value' override, which now
    shows a warning when used there. These assertions exist so this stays a tracked, intentional
    finding: if a future plant/controller change makes low-R stable, THIS test should start
    failing (isfinite becomes true) and should be revisited/loosened deliberately -- and the
    dashboard's slider floor and warning threshold reconsidered too -- not have its assumption
    silently bit-rot unnoticed either way."""

    @pytest.mark.parametrize("r_ohm", [1.0, 1.5, 2.0, 2.5, 3.0])
    def test_low_r_currently_diverges(self, r_ohm):
        rmse = _settled_rmse(r_ohm, MAG_DEFAULT, OMEGA_DEFAULT)
        assert not np.isfinite(rmse), (
            f"R={r_ohm}ohm no longer diverges (settled RMSE={rmse:.3f}V) -- this was a documented, "
            "analytically-explained open-loop instability (Patch 9). If a plant/controller change "
            "genuinely fixed it, update this test, the module docstring, TestOpenLoopStabilityThreshold, "
            "and reconsider whether MIN_STABLE_LOAD_RESISTANCE_OHM (sim/vsc_system.py) and the dashboard's "
            "slider floor/warning should relax -- rather than just deleting the assertion."
        )


class TestOpenLoopStabilityThreshold:
    """Verifies the ROOT CAUSE mechanism itself (not just its symptom): the plant's own
    load-feedback dynamics, with no controller at all, are unstable below R* ~= 3.3707ohm. This is
    what makes TestKnownDivergenceAtLowR's divergence structural rather than a training artifact,
    and it's what MIN_STABLE_LOAD_RESISTANCE_OHM / the dashboard's slider floor are supposed to
    stay a safe margin above. If Adf/Bdf ever change (e.g. a different identified plant), THIS
    class is what should fail first and force MIN_STABLE_LOAD_RESISTANCE_OHM to be re-derived,
    rather than the failure only surfacing as a mysterious NaN somewhere downstream."""

    @pytest.mark.parametrize(
        "r_ohm,expect_unstable",
        [(0.5, True), (1.0, True), (2.0, True), (3.0, True), (3.3707, None), (3.5, False), (8.0064, False), (20.0, False)],
    )
    def test_spectral_radius_matches_empirical_divergence(self, r_ohm, expect_unstable):
        """Spot-checks that the analytic spectral radius crosses 1.0 in the same place the
        empirical divergence grid (TestKnownDivergenceAtLowR / the on-distribution baseline) does
        -- the two are independent checks of the same underlying fact and should agree."""
        radius = load_feedback_spectral_radius(r_ohm)
        if expect_unstable is None:  # right at the analytic crossing -- direction, not magnitude
            return
        assert (radius > 1.0) == expect_unstable, (
            f"R={r_ohm}ohm: load_feedback_spectral_radius={radius:.4f}, expected "
            f"{'unstable (>1)' if expect_unstable else 'stable (<=1)'}"
        )

    def test_min_stable_load_resistance_has_safe_margin_over_analytic_threshold(self):
        """The dashboard's slider floor must sit in genuinely stable territory, with margin -- not
        exactly at the crossing, where floating-point/model-mismatch noise could still diverge."""
        radius_at_floor = load_feedback_spectral_radius(MIN_STABLE_LOAD_RESISTANCE_OHM)
        assert radius_at_floor < 0.95, (
            f"MIN_STABLE_LOAD_RESISTANCE_OHM={MIN_STABLE_LOAD_RESISTANCE_OHM}ohm has spectral radius "
            f"{radius_at_floor:.4f} -- expected a real margin (<0.95) below the R*~=3.3707ohm instability "
            "crossing, not a value that barely clears it."
        )

    def test_min_stable_load_resistance_closed_loop_actually_converges(self):
        """End-to-end confirmation that the chosen floor is not just analytically stable in the
        open-loop sense but also gives a finite, sane settled RMSE in the real closed loop."""
        rmse = _settled_rmse(MIN_STABLE_LOAD_RESISTANCE_OHM, MAG_DEFAULT, OMEGA_DEFAULT)
        assert np.isfinite(rmse), f"R=MIN_STABLE_LOAD_RESISTANCE_OHM={MIN_STABLE_LOAD_RESISTANCE_OHM}ohm still diverges in closed loop"
