"""Tests for control/mpc/controller.py: the closed-loop MPC must actually track a speed/torque
reference against a real DcMotorSystem (not just "not crash"), mirroring test_pi_controller.py's
structure and tolerances so the two controllers are held to the same bar.

Step counts are deliberately smaller than test_pi_controller.py's where tracking already settles
well within them (verified interactively before writing this file, see docs/patch10_*): unlike PI
(an O(1) per-step update), MpcController solves a real QP every step (~6-7ms/step on this
machine), so a naive copy of PI's step counts would make this file the slowest in the suite for no
added rigor.
"""

import numpy as np
import pytest

from driveflow.control.classical import PICascadeController
from driveflow.control.mpc import MpcController
from driveflow.sim import (
    ContOneQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PolynomialStaticLoad,
)

TAU = 1e-4


def build_system():
    return DcMotorSystem(
        converter=ContOneQuadrantConverter(),
        motor=DcPermanentlyExcitedMotor(),
        load=PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)),
        supply=IdealVoltageSupply(u_nominal=65.0),  # motor is rated u=60V (see DcPermanentlyExcitedMotor limits)
        ode_solver=EulerSolver(),
        tau=TAU,
    )


class TestMpcController:
    def test_tracks_speed_step_reference(self):
        system = build_system()
        controller = MpcController(system)
        omega_ref = 100.0  # rad/s, well within the motor's nominal range (300 rad/s)

        state = system.reset()
        controller.reset()
        n_steps = 3000  # settles well within this (verified interactively: std=0 by 3000 steps)
        omega_trace = np.zeros(n_steps)
        for k in range(n_steps):
            action = controller.control(state, omega_ref)
            state = system.simulate(action)
            omega_trace[k] = (state * system.limits)[system.OMEGA_IDX]

        settled = omega_trace[-500:]
        assert np.mean(settled) == pytest.approx(omega_ref, rel=0.05)
        assert np.std(settled) < 0.05 * omega_ref

    def test_tracks_different_reference_levels(self):
        for omega_ref in (50.0, 200.0):
            system = build_system()
            controller = MpcController(system)
            state = system.reset()
            controller.reset()
            for _ in range(3000):
                action = controller.control(state, omega_ref)
                state = system.simulate(action)
            final_omega = (state * system.limits)[system.OMEGA_IDX]
            assert final_omega == pytest.approx(omega_ref, rel=0.08)

    def test_action_stays_within_converter_bounds(self):
        system = build_system()
        controller = MpcController(system)
        state = system.reset()
        controller.reset()
        for _ in range(1500):
            action = controller.control(state, omega_ref=300.0)
            assert 0.0 <= action[0] <= 1.0
            state = system.simulate(action)

    def test_current_stays_within_safety_margin(self):
        """The soft current penalty (q_current_penalty) is the MPC's only current-safety
        mechanism -- unlike PICascadeController, which clips an explicit i_ref, this controller
        never forms one, so this is the concrete regression check that the penalty actually
        holds the line under an aggressive step (0 -> near-max nominal speed)."""
        system = build_system()
        controller = MpcController(system)
        i_limit = system.limits[system.CURRENTS_IDX[0]]
        state = system.reset()
        controller.reset()
        max_i = 0.0
        for _ in range(1500):
            action = controller.control(state, omega_ref=300.0)
            state = system.simulate(action)
            max_i = max(max_i, abs((state * system.limits)[system.CURRENTS_IDX[0]]))
        assert max_i < 0.9 * i_limit, f"max |i|={max_i:.1f}A -- soft current penalty did not hold under the safety margin"

    def test_reset_clears_warm_start(self):
        system = build_system()
        controller = MpcController(system)
        state = system.reset()
        controller.reset()
        for _ in range(500):
            action = controller.control(state, omega_ref=200.0)
            state = system.simulate(action)
        assert controller._warm_start is not None
        controller.reset()
        assert controller._warm_start is None


class TestMpcControllerTorqueMode:
    """control_torque() tracks i_ref = torque_ref/psi_e directly instead of omega -- same
    contract as PICascadeController.control_torque(), see dashboard.py's "Torque (τ_ref)" option."""

    def test_tracks_torque_reference(self):
        system = build_system()
        controller = MpcController(system)
        torque_ref = 10.0  # Nm, within the motor's nominal range (16 Nm)
        psi_e = system.electrical_motor.motor_parameter["psi_e"]

        state = system.reset()
        controller.reset()
        n_steps = 1500  # current tracking settles much faster than speed tracking above
        i_trace = np.zeros(n_steps)
        for k in range(n_steps):
            action = controller.control_torque(state, torque_ref)
            state = system.simulate(action)
            i_trace[k] = (state * system.limits)[system.CURRENTS_IDX[0]]

        settled_torque = i_trace[-500:] * psi_e
        assert np.mean(settled_torque) == pytest.approx(torque_ref, rel=0.05)

    def test_action_stays_within_converter_bounds(self):
        system = build_system()
        controller = MpcController(system)
        state = system.reset()
        controller.reset()
        for _ in range(800):
            action = controller.control_torque(state, torque_ref=12.0)
            assert 0.0 <= action[0] <= 1.0
            state = system.simulate(action)

    def test_omega_is_free_running_not_tracking_a_setpoint(self):
        """Same invariant as PICascadeControllerTorqueMode's own test: different constant torques
        against the same speed-dependent load should settle at different equilibrium speeds, not
        converge to anything shared."""
        finals = []
        for torque_ref in (5.0, 15.0):
            system = build_system()
            controller = MpcController(system)
            state = system.reset()
            controller.reset()
            for _ in range(1500):
                action = controller.control_torque(state, torque_ref)
                state = system.simulate(action)
            finals.append((state * system.limits)[system.OMEGA_IDX])
        assert finals[0] != pytest.approx(finals[1], rel=0.05)


class TestPiVsMpcComparison:
    """The comparison docs/patch5_alcance_macrofase_B.md said never existed (DPC vs. PI/MPC,
    across incompatible physical domains -- VSC vs. DC motor). PI and MPC, by contrast, target
    the SAME plant/state/action here, so this comparison is genuinely meaningful and previously
    just didn't exist because MPC didn't (see docs/patch10_implementacion_mpc.md). This is not a
    claim that one is "better" in general -- only that both are real, working controllers for the
    same DcMotorSystem, checked here under identical conditions."""

    def test_both_controllers_track_the_same_reference_on_the_same_plant(self):
        omega_ref = 150.0
        results = {}
        for name, controller_cls in (("PI", PICascadeController), ("MPC", MpcController)):
            system = build_system()
            controller = controller_cls(system)
            state = system.reset()
            controller.reset()
            for _ in range(3000):
                action = controller.control(state, omega_ref)
                state = system.simulate(action)
            results[name] = (state * system.limits)[system.OMEGA_IDX]

        for name, final_omega in results.items():
            assert final_omega == pytest.approx(omega_ref, rel=0.08), f"{name} failed to track omega_ref={omega_ref} within tolerance"


class TestExactQpSolver:
    """C3: MpcController(qp_solver="exact") -- docs/audit/stability/mpc_exact_solver.json. Measured
    2026-10-20: 8-194x faster than the default L-BFGS-B on the four benchmark runs; the default is
    unchanged (bit-identical to the original controller)."""

    def test_box_qp_matches_a_reference_solver(self):
        from scipy.optimize import minimize

        from driveflow.control.mpc.controller import _box_qp

        rng = np.random.default_rng(0)
        for _ in range(50):
            a = rng.normal(size=(8, 8))
            h, g = a @ a.T + 0.1 * np.eye(8), rng.normal(size=8) * 5
            u = _box_qp(h, g, 0.0, 1.0)
            ref = minimize(lambda x: 0.5 * x @ h @ x + g @ x, np.zeros(8), jac=lambda x: h @ x + g, bounds=[(0, 1)] * 8,
                           method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12)).x
            assert np.all((u >= 0) & (u <= 1))
            assert 0.5 * u @ h @ u + g @ u <= 0.5 * ref @ h @ ref + g @ ref + 1e-7

    def test_exact_solver_tracks_and_respects_the_current_limit(self):
        from driveflow.validation import run_dc_closed_loop

        r = run_dc_closed_loop("MPC", 150.0, duration_s=0.4, controller_kwargs=dict(qp_solver="exact"))
        assert abs(r["omega"][-1] - 150.0) < 0.2
        assert r["i"].max() <= 210.0 * 0.85 + 0.1  # 178.5 A soft limit; the default solver reaches 178.8 A
        assert np.all((r["u"] >= -1e-9))

    def test_mostly_avoids_the_iterative_fallback(self):
        system = build_system()
        ctrl = MpcController(system, qp_solver="exact")
        system.seed(np.random.SeedSequence(0))
        state = system.reset()
        ctrl.reset()
        for _ in range(2000):
            state = system.simulate(ctrl.control(state, 150.0))
        assert ctrl.iterative_steps <= 0.01 * 2000

    def test_rejects_unknown_solver(self):
        with pytest.raises(ValueError):
            MpcController(build_system(), qp_solver="cvx")
