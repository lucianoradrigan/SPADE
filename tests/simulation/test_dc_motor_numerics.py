"""Numerical stability of the DC-motor plant (dc_perm_ex_v1) -- docs/SIMULATION_STABILITY_REPORT.md
Sec. 2. Every tolerance is derived from theory (order of the method, eigenvalues, analytic steady
state) or from the platform's own control criteria, never picked to make a run pass."""

import numpy as np
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.sim.solvers import ScipySolveIvpSolver
from driveflow.sim import EulerSolver
from driveflow.validation import (
    build_dc_system,
    dc_motor_linear_matrix,
    euler_spectral_radius,
    run_dc_closed_loop,
    run_dc_open_loop,
)
from driveflow.validation.benches import DC_LOAD_PARAMETER

pytestmark = pytest.mark.simulation

TAU = 1e-4


def _rk45():
    # Reference solution: adaptive RK45 at tolerances ~1e6x tighter than any error asserted below.
    return ScipySolveIvpSolver(method="RK45", rtol=1e-9, atol=1e-9)


def _linear_matrix():
    s = build_dc_system()
    mp = s.electrical_motor.motor_parameter
    return dc_motor_linear_matrix(mp["r_a"], mp["l_a"], mp["psi_e"], s.mechanical_load.j_total, DC_LOAD_PARAMETER["b"]), mp


def _analytic_open_loop_omega(u: float) -> float:
    """Steady state of u = r_a i + psi_e w, psi_e i = a + b w (w > 0)."""
    _, mp = _linear_matrix()
    a, b = DC_LOAD_PARAMETER["a"], DC_LOAD_PARAMETER["b"]
    return (mp["psi_e"] * u / mp["r_a"] - a) / (b + mp["psi_e"] ** 2 / mp["r_a"])


class TestBenchIsTheProductionPlant:
    def test_bench_trajectory_equals_datagen_trajectory(self):
        """The bench (selectable solver) must simulate exactly what datagen/runner.py does, or
        nothing measured on it says anything about the platform."""
        recs = run_scenario(Scenario(scenario_id="bench_eq", duration_s=0.05, seed=0))
        bench = run_dc_closed_loop("PI", 150.0, duration_s=0.05)
        np.testing.assert_allclose(bench["i"], [r["current_r"] for r in recs], rtol=0, atol=1e-9)
        np.testing.assert_allclose(bench["omega"] * 60 / (2 * np.pi), [r["rpm"] for r in recs], rtol=0, atol=1e-9)


@pytest.fixture(scope="module")
def errors():
    ref = run_dc_closed_loop("PI", 150.0, duration_s=0.2, ode_solver=_rk45())
    out = {}
    for n in (1, 2, 4):
        x = run_dc_closed_loop("PI", 150.0, duration_s=0.2, ode_solver=EulerSolver(nsteps=n))
        out[n] = (np.max(np.abs(x["omega"] - ref["omega"])), np.max(np.abs(x["i"] - ref["i"])))
    return out


class TestTimeStepConvergence:
    def test_observed_order_is_one(self, errors):
        """Forward Euler is first order: halving the integration step must halve the error
        (ratio 2). A ratio far from 2 means the error is NOT discretisation error (e.g. a bug in
        the substepping, or the RK45 reference itself being wrong)."""
        for k in (0, 1):
            for n in (1, 2):
                ratio = errors[n][k] / errors[2 * n][k]
                assert 1.8 < ratio < 2.2, (n, k, ratio)

    def test_production_step_error_is_below_control_resolution(self, errors):
        """At the production step (tau=1e-4, nsteps=1): speed error < 0.1 % of the reference --
        an order of magnitude below the 2 % band used for every settling-time criterion, so the
        integrator cannot flip a control PASS/FAIL. Current error < 2 % of the motor's current
        limit (210 A)."""
        e_omega, e_i = errors[1]
        assert e_omega < 0.001 * 150.0
        assert e_i < 0.02 * 210.0


class TestEulerStabilityRegion:
    def test_plant_itself_is_stable(self):
        a, _ = _linear_matrix()
        assert np.all(np.linalg.eigvals(a).real < 0)

    def test_production_step_is_inside_the_stability_region(self):
        a, _ = _linear_matrix()
        assert euler_spectral_radius(a, TAU) < 1.0

    def test_divergence_boundary_matches_eigenvalue_prediction(self):
        """A physically stable plant that diverges is a NUMERICAL instability: with real
        eigenvalues, forward Euler is stable iff tau < 2/|lambda|max (~2.58 ms here). Just
        inside it converges to the analytic steady state; just outside it blows up."""
        a, _ = _linear_matrix()
        tau_max = 2.0 / np.max(np.abs(np.linalg.eigvals(a)))
        w_ss = _analytic_open_loop_omega(0.5 * 65.0)
        inside = run_dc_open_loop(0.5, duration_s=0.4, tau=0.93 * tau_max)
        outside = run_dc_open_loop(0.5, duration_s=0.4, tau=1.05 * tau_max)
        assert inside["omega"][-1] == pytest.approx(w_ss, rel=1e-3)
        assert not np.all(np.abs(outside["omega"]) < 10 * w_ss)


class TestSteadyStateAgainstAnalytic:
    @pytest.mark.parametrize("solver", ["euler", "rk45"])
    def test_open_loop_steady_state(self, solver):
        r = run_dc_open_loop(0.5, duration_s=0.3, ode_solver=EulerSolver() if solver == "euler" else _rk45())
        assert r["omega"][-1] == pytest.approx(_analytic_open_loop_omega(0.5 * 65.0), rel=1e-4)

    def test_closed_loop_steady_state_current_balances_friction(self):
        """At constant speed the motor torque psi_e*i equals the load torque a + b*w exactly."""
        r = run_dc_closed_loop("PI", 150.0, duration_s=0.5)
        _, mp = _linear_matrix()
        expected_i = (DC_LOAD_PARAMETER["a"] + DC_LOAD_PARAMETER["b"] * 150.0) / mp["psi_e"]
        assert r["i"][-1] == pytest.approx(expected_i, rel=1e-6)


class TestReproducibility:
    def test_bitwise_identical_for_same_seed_including_vibration(self):
        kw = dict(scenario_id="rep", fault_type="outer_race", duration_s=0.05, seed=7)
        a, b = run_scenario(Scenario(**kw)), run_scenario(Scenario(**kw))
        for col in ("current_r", "rpm", "acc_x", "acc_y", "acc_z", "torque_nm"):
            assert [r[col] for r in a] == [r[col] for r in b], col

    def test_seed_only_changes_vibration_channels(self):
        """Documents AUDIT_ANSWERS Q15: the plant is deterministic; the seed reaches only Module B.
        If this ever fails, the seed started to vary the operating condition and Q15's caveat
        (and docs/AI_TECHNICAL_AUDIT.md AI-N07) must be revisited."""
        a = run_scenario(Scenario(scenario_id="s1", fault_type="inner_race", duration_s=0.05, seed=1))
        b = run_scenario(Scenario(scenario_id="s2", fault_type="inner_race", duration_s=0.05, seed=2))
        assert [r["current_r"] for r in a] == [r["current_r"] for r in b]
        assert [r["acc_x"] for r in a] != [r["acc_x"] for r in b]


@pytest.mark.slow
class TestLongSimulation:
    def test_twenty_seconds_no_drift_no_invalid_values(self):
        """200 000 steps: no NaN/inf, and the steady state after 20 s equals the one after 1 s to
        1e-9 (a drifting integrator or accumulating time error would show up here)."""
        r = run_dc_closed_loop("PI", 150.0, duration_s=20.0)
        assert np.all(np.isfinite(r["omega"])) and np.all(np.isfinite(r["i"]))
        k1 = int(1.0 / TAU) - 1
        assert abs(r["omega"][-1] - r["omega"][k1]) < 1e-9
        assert abs(r["i"][-1] - r["i"][k1]) < 1e-9


class TestInvalidInputsAreRejected:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(duration_s=0.0),
            dict(duration_s=-1.0),
            dict(omega_ref_rad_s=float("nan")),
            dict(omega_ref_rad_s=float("inf")),
            dict(motor_parameter_overrides={"l_a": 0.0}),
            dict(motor_parameter_overrides={"r_a": -0.016}),
            dict(motor_parameter_overrides={"j_rotor": -0.1}),
            dict(electrical_severity=-5.0, fault_type="outer_race"),
            dict(electrical_noise_pct=-10.0),
            dict(torque_ref_nm=float("nan")),
            dict(fault_type="outer_race", fault_order_override=0.0),
            dict(controller_type="DPC", plant_config_id="vsc_dpc_v1", load_resistance_ohm=0.0),
            dict(controller_type="DPC", plant_config_id="vsc_dpc_v1", load_resistance_ohm=-5.0),
            dict(controller_type="DPC", plant_config_id="vsc_dpc_v1", reference_magnitude_v=float("nan")),
        ],
    )
    def test_rejected_with_a_clear_error(self, kwargs):
        """Before S-03 every one of these ran silently and returned an empty list, an all-NaN
        trajectory, a non-physical one (r_a<0 -> 1.5e17 rpm) or a bare ZeroDivisionError."""
        with pytest.raises(ValueError, match="Scenario"):
            Scenario(scenario_id="bad", **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(omega_ref_rad_s=-50.0),  # one-quadrant converter cannot reverse: physical, not invalid
            dict(omega_ref_rad_s=1000.0),  # unreachable with 65 V: saturates physically
            dict(controller_type="DPC", plant_config_id="vsc_dpc_v1", load_resistance_ohm=2.0),  # physically unstable plant
        ],
    )
    def test_physically_legal_extremes_are_still_accepted(self, kwargs):
        Scenario(scenario_id="legal", **kwargs)

    def test_unreachable_speed_saturates_at_the_supply_limit_not_at_infinity(self):
        """omega_ref far above what 65 V can sustain: the speed must settle at the voltage-limited
        no-load point (u = r_a*i + psi_e*w with u = 65 V), not run away."""
        r = run_dc_closed_loop("PI", 1000.0, duration_s=2.5)
        assert np.all(np.isfinite(r["omega"]))
        assert r["omega"][-1] == pytest.approx(_analytic_open_loop_omega(65.0), rel=1e-3)


def test_denormalized_state_vector_is_finite_with_a_fault_injected():
    """S-06: the fault-phase state had limit = inf, so state * limits gave 0 * inf = NaN in that
    slot (and a RuntimeWarning on every step of every faulted run)."""
    from driveflow.datagen.runner import build_plant

    system = build_plant(Scenario(scenario_id="phase", fault_type="outer_race", extra_faults=[("inner_race", None)]))
    system.seed(np.random.SeedSequence(0))
    state = system.reset()
    for _ in range(50):
        state = system.simulate(np.array([0.5]))
    with np.errstate(invalid="raise"):
        physical = state * system.limits
    assert np.all(np.isfinite(physical))
