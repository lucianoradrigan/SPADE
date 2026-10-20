"""Closed-loop performance of the DC-motor controllers -- docs/SIMULATION_STABILITY_REPORT.md Sec. 5.

Acceptance criteria are taken from reference cases, not from a generic table:
- small (unsaturated) speed steps and load-torque steps are compared with the LINEAR closed loop
  the PI speed loop was designed on (ideal inner current loop, plant psi_e/(J s + b)). The only
  effects that model leaves out are the inner current loop (crossover 1/(2 tau) = 5000 rad/s,
  ~60x faster than the 80 rad/s speed loop) and the 0.1 ms sampling delay (1 % of the ~10 ms
  rise time), so timing must agree within 10 %, overshoot within 1.5 percentage points and the
  disturbance dip within 5 %;
- large steps are current-limited: the rise time cannot beat the physical bound set by the
  current cap (0.8 x 210 A), and a controller that uses the available current fully reaches it;
- PI has integral action, so its steady-state error must vanish (integrator theory), while the MPC
  has no disturbance model or integrator and is only required to stay an order of magnitude
  inside the 2 % settling band.
"""

import numpy as np
import pytest
from scipy import signal

from driveflow.control.classical import PICascadeController
from driveflow.validation import build_dc_system, run_dc_closed_loop, step_response_metrics
from driveflow.validation.benches import DC_LOAD_PARAMETER

pytestmark = pytest.mark.control

T_STEP = 0.4


def _linear_loop():
    s = build_dc_system()
    c = PICascadeController(s)
    psi, j, b = c.psi_e, s.mechanical_load.j_total, DC_LOAD_PARAMETER["b"]
    den = [j, b + psi * c.kp_omega, psi * c.ki_omega]
    reference = signal.lti([psi * c.kp_omega, psi * c.ki_omega], den)
    disturbance = signal.lti([-1.0, 0.0], den)  # omega / T_load
    return reference, disturbance, c


def _linear_step(sys, amplitude, t_end=0.4):
    t = np.linspace(0, t_end, 40001)
    _, y = signal.step(sys, T=t)
    return t, amplitude * y


@pytest.fixture(scope="module")
def metrics():
    reference, _, _ = _linear_loop()
    t, y = _linear_step(reference, 2.0)
    lin = step_response_metrics(t, 150.0 + y, 152.0, 150.0)
    r = run_dc_closed_loop("PI", lambda tt: 150.0 if tt < T_STEP else 152.0, duration_s=0.8)
    sim = step_response_metrics(r["t"], r["omega"], 152.0, 150.0, t_step=T_STEP)
    return lin, sim, r


@pytest.fixture(scope="module")
def disturbance_run():
    return run_dc_closed_loop("PI", 150.0, duration_s=1.0, load_step_nm=10.0, load_step_t_s=0.5)


@pytest.fixture(scope="module")
def mpc_step():
    r = run_dc_closed_loop("MPC", 150.0, duration_s=0.4)
    return r, step_response_metrics(r["t"], r["omega"], 150.0, 0.0)


class TestPiSmallSignalMatchesDesignModel:
    def test_step_is_unsaturated(self, metrics):
        _, _, r = metrics
        assert r["i"][r["t"] > T_STEP].max() < 0.8 * 210.0

    def test_rise_and_settling_time(self, metrics):
        lin, sim, _ = metrics
        assert sim.rise_time_s == pytest.approx(lin.rise_time_s, rel=0.10)
        assert sim.settling_time_s == pytest.approx(lin.settling_time_s, rel=0.10)

    def test_overshoot(self, metrics):
        """~15 % -- the PI zero, not the pole damping (0.9), sets it. The controller's docstring
        used to claim 'essentially no overshoot' (S-04, corrected)."""
        lin, sim, _ = metrics
        assert sim.overshoot_pct == pytest.approx(lin.overshoot_pct, abs=1.5)
        assert sim.overshoot_pct > 10.0

    def test_zero_steady_state_error(self, metrics):
        _, sim, _ = metrics
        assert sim.steady_state_error < 1e-6


class TestPiLoadDisturbance:
    def test_speed_dip_matches_design_model(self, disturbance_run):
        run = disturbance_run
        _, disturbance, _ = _linear_loop()
        _, y = _linear_step(disturbance, 10.0)
        dip_lin = -y.min()
        dip_sim = 150.0 - run["omega"][run["t"] >= 0.5].min()
        assert dip_sim == pytest.approx(dip_lin, rel=0.05)

    def test_rejects_the_disturbance_completely(self, disturbance_run):
        run = disturbance_run
        """Integral action: speed returns to the reference and the current rises by exactly
        10 Nm / psi_e."""
        psi = run["system"].electrical_motor.motor_parameter["psi_e"]
        assert abs(run["omega"][-1] - 150.0) < 1e-6
        i_before = run["i"][np.searchsorted(run["t"], 0.5) - 1]
        assert run["i"][-1] - i_before == pytest.approx(10.0 / psi, rel=1e-4)


class TestPiLargeStepIsCurrentLimited:
    def test_rise_time_equals_the_physical_bound(self):
        s = build_dc_system()
        psi, j = s.electrical_motor.motor_parameter["psi_e"], s.mechanical_load.j_total
        a, b = DC_LOAD_PARAMETER["a"], DC_LOAD_PARAMETER["b"]
        i_cap = 0.8 * s.limits[s.CURRENTS_IDX[0]]
        t = np.linspace(0, 1, 100001)
        w = (psi * i_cap - a) / b * (1 - np.exp(-b / j * t))  # accelerating at the current cap
        bound = t[np.argmax(w >= 135.0)] - t[np.argmax(w >= 15.0)]
        r = run_dc_closed_loop("PI", 150.0, duration_s=0.6)
        m = step_response_metrics(r["t"], r["omega"], 150.0, 0.0)
        assert m.rise_time_s >= bound - 2e-4  # cannot beat physics (2 samples of timing resolution)
        assert m.rise_time_s <= 1.05 * bound  # and uses the available current
        assert r["i"].max() <= i_cap + 1e-6 * i_cap + 1.0  # current cap honoured (1 A inner-loop transient)

    def test_no_overshoot_and_no_windup_on_a_saturated_step(self):
        r = run_dc_closed_loop("PI", 150.0, duration_s=0.6)
        m = step_response_metrics(r["t"], r["omega"], 150.0, 0.0)
        assert m.overshoot_pct < 2.0 and m.settling_time_s is not None


class TestPiRobustness:
    @pytest.mark.parametrize("override", [{"r_a": 0.024}, {"l_a": 38e-6}, {"psi_e": 0.132}, {"j_rotor": 0.075}])
    def test_detuned_plant_stays_stable_and_settles(self, override):
        """Controller tuned for the nominal motor, plant drifted (+50 % r_a, x2 l_a, -20 % psi_e,
        x3 rotor inertia). Requirement: stable, settles inside the 2 % band, no steady-state error
        left at the end of the window."""
        r = run_dc_closed_loop("PI", 150.0, duration_s=1.0, motor_parameter=override)
        m = step_response_metrics(r["t"], r["omega"], 150.0, 0.0)
        assert m.finite and m.settling_time_s is not None and m.settling_time_s < 0.9
        assert m.overshoot_pct < 5.0
        assert abs(r["omega"][-1] - 150.0) < 0.01


class TestTorqueMode:
    def test_current_tracks_torque_over_psi(self):
        r = run_dc_closed_loop("PI", 0.0, duration_s=0.05, torque_ref_nm=10.0)
        psi = r["system"].electrical_motor.motor_parameter["psi_e"]
        assert r["i"][-1] == pytest.approx(10.0 / psi, rel=0.01)


@pytest.mark.slow
class TestMpc:
    def test_large_step_is_current_limited_too(self, mpc_step):
        r, m = mpc_step
        assert m.overshoot_pct < 2.0 and m.settling_time_s is not None
        assert r["i"].max() < 1.10 * 0.85 * 210.0  # soft penalty around 0.85 x limit, ~5 % excursion measured

    def test_steady_state_offset_is_small_but_not_zero(self, mpc_step):
        """No integrator / disturbance model: friction leaves an offset (measured ~0.06 rad/s at
        150 rad/s). Required: < 0.2 % of the reference, i.e. 10x inside the 2 % settling band."""
        _, m = mpc_step
        assert m.steady_state_error < 0.002 * 150.0

    def test_load_step_offset_stays_bounded(self):
        r = run_dc_closed_loop("MPC", 150.0, duration_s=0.7, load_step_nm=10.0, load_step_t_s=0.4)
        assert np.all(np.isfinite(r["omega"]))
        assert abs(r["omega"][-1] - 150.0) < 0.002 * 150.0


class TestPiSetpointWeighting:
    """C2 / S-04: optional setpoint weight b on the speed loop's proportional term. Measured
    2026-10-20 on the 2 rad/s step at 150 rad/s: overshoot 15.1 % (b = 1), 3.1 % (b = 0.7),
    0.4 % (b = 0.5); rise time 9.3 / 16.2 / 24.1 ms; load-step dip unchanged (1.804 rad/s)."""

    @staticmethod
    def _step(weight):
        s = build_dc_system()
        c = PICascadeController(s)
        psi, j, b = c.psi_e, s.mechanical_load.j_total, DC_LOAD_PARAMETER["b"]
        lin_sys = signal.lti([psi * c.kp_omega * weight, psi * c.ki_omega], [j, b + psi * c.kp_omega, psi * c.ki_omega])
        t, y = _linear_step(lin_sys, 2.0)
        lin = step_response_metrics(t, 150.0 + y, 152.0, 150.0)
        r = run_dc_closed_loop("PI", lambda tt: 150.0 if tt < T_STEP else 152.0, duration_s=0.8, controller_kwargs=dict(setpoint_weight=weight))
        return lin, step_response_metrics(r["t"], r["omega"], 152.0, 150.0, t_step=T_STEP)

    @pytest.mark.parametrize("weight", [0.7, 0.5])
    def test_overshoot_follows_the_design_model_and_drops(self, weight):
        lin, sim = self._step(weight)
        assert sim.overshoot_pct == pytest.approx(lin.overshoot_pct, abs=1.5)
        assert sim.overshoot_pct < (5.0 if weight == 0.7 else 1.0)
        assert sim.steady_state_error < 1e-6

    def test_default_weight_is_the_original_controller(self, metrics):
        _, sim_default, r_default = metrics
        r = run_dc_closed_loop("PI", lambda tt: 150.0 if tt < T_STEP else 152.0, duration_s=0.8, controller_kwargs=dict(setpoint_weight=1.0))
        np.testing.assert_array_equal(r["omega"], r_default["omega"])

    def test_disturbance_rejection_does_not_depend_on_the_weight(self, disturbance_run):
        # The start-up from rest differs (that IS a reference step); once settled, the response to
        # the load step at 0.5 s must not depend on b.
        r = run_dc_closed_loop("PI", 150.0, duration_s=1.0, load_step_nm=10.0, load_step_t_s=0.5, controller_kwargs=dict(setpoint_weight=0.5))
        after = r["t"] >= 0.5
        np.testing.assert_allclose(r["omega"][after], disturbance_run["omega"][after], rtol=0, atol=1e-6)

    @pytest.mark.parametrize("weight", [-0.1, 1.5])
    def test_rejects_weights_outside_0_1(self, weight):
        with pytest.raises(ValueError):
            PICascadeController(build_dc_system(), setpoint_weight=weight)
