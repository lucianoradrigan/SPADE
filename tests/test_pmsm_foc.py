"""Tests for control/classical/pmsm_foc.py -- the native PMSM FOC + MTPA controller used by
viz/dashboard.py's "Plano i_d-i_q PMSM" tab (only ever a scratch script before this session's
GUI work; now a real library module)."""

import numpy as np
import pytest

from driveflow.control.classical import DqCurrentController, generate_mtpa_vs_naive_cloud, mtpa_id_iq, torque_of
from driveflow.control.classical.pmsm_foc import I_LIM, build_pmsm_system, run_current_step
from driveflow.sim import PermanentMagnetSynchronousMotor


class TestMtpaIdIq:
    def test_matches_gems_own_torque_limit_at_nominal_current(self):
        """mtpa_id_iq is a generalization of PermanentMagnetSynchronousMotor._torque_limit()
        (only evaluated there at the nominal current) to any current magnitude -- must reproduce
        GEM's own number exactly at that one point."""
        motor = PermanentMagnetSynchronousMotor()
        i_nom = motor.nominal_values["i"]
        gem_torque_limit = motor._torque_limit()
        i_d, i_q = mtpa_id_iq(i_nom)
        assert torque_of(i_d, i_q) == pytest.approx(gem_torque_limit, rel=1e-3)

    def test_id_is_negative_for_a_salient_motor(self):
        """l_d < l_q for this motor (real GEM spec) -- MTPA always uses some negative i_d
        (reluctance-assisting), never i_d=0 (that would be the naive/non-salient assumption)."""
        i_d, _ = mtpa_id_iq(200.0)
        assert i_d < 0

    def test_current_magnitude_is_preserved(self):
        i_s = 250.0
        i_d, i_q = mtpa_id_iq(i_s)
        assert np.hypot(i_d, i_q) == pytest.approx(i_s, rel=1e-6)


class TestDqCurrentController:
    def test_tracks_mtpa_reference_within_a_few_percent(self):
        """Regression for the anti-windup fix found while building this: an unclamped integrator
        diverged (even sign-flipped) at i_s=380A -- this must stay close to the target."""
        i_d_ref, i_q_ref = mtpa_id_iq(250)
        traj = run_current_step(i_d_ref, i_q_ref, duration_s=0.03)
        final_i_d, final_i_q = traj[-1]
        assert abs(final_i_d - i_d_ref) < 0.05 * abs(i_d_ref)
        assert abs(final_i_q - i_q_ref) < 0.05 * abs(i_q_ref)

    def test_high_current_step_does_not_diverge(self):
        """The specific case that failed before the anti-windup fix."""
        i_d_ref, i_q_ref = mtpa_id_iq(380)
        traj = run_current_step(i_d_ref, i_q_ref, duration_s=0.03)
        final_i_d, final_i_q = traj[-1]
        assert np.isfinite(final_i_d) and np.isfinite(final_i_q)
        assert abs(final_i_d - i_d_ref) < 0.1 * abs(i_d_ref)


class TestGenerateMtpaVsNaiveCloud:
    def test_returns_both_policy_clouds_and_reference_geometry(self):
        data = generate_mtpa_vs_naive_cloud(i_s_values=[100, 300], duration_s=0.02, subsample=5)
        assert len(data["mtpa"]) > 0
        assert len(data["naive"]) > 0
        assert len(data["mtpa_curve"]) > 0
        assert len(data["limit_circle"]) > 0

    def test_mtpa_delivers_more_torque_than_naive_at_equal_current_magnitude(self):
        """The whole point of MTPA: same |i_s|, more torque, because it exploits the motor's
        reluctance torque (l_d != l_q) instead of ignoring it (naive i_d=0)."""
        i_s = 300.0
        id_mtpa, iq_mtpa = mtpa_id_iq(i_s)
        t_mtpa = torque_of(id_mtpa, iq_mtpa)
        t_naive = torque_of(0.0, i_s)
        assert t_mtpa > t_naive

    def test_limit_circle_radius_matches_i_lim(self):
        data = generate_mtpa_vs_naive_cloud(i_s_values=[100], duration_s=0.01, subsample=5)
        radii = [np.hypot(x, y) for x, y in data["limit_circle"]]
        assert np.allclose(radii, I_LIM, atol=0.5)


class TestBuildPmsmSystem:
    def test_reset_and_simulate_run_without_error(self):
        system = build_pmsm_system()
        state = system.reset()
        controller = DqCurrentController(system)
        action, (i_d, i_q) = controller.control(state, system, 0.0, 0.0)
        new_state = system.simulate(action)
        assert new_state.shape == state.shape
