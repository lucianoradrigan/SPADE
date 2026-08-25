"""Tests for control/classical/pi_controller.py: the closed-loop cascade must actually track a
speed reference against a real DcMotorSystem (not just "not crash")."""

import numpy as np
import pytest

from driveflow.control.classical import PICascadeController
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


class TestPICascadeController:
    def test_tracks_speed_step_reference(self):
        system = build_system()
        controller = PICascadeController(system)
        omega_ref = 100.0  # rad/s, well within the motor's nominal range (300 rad/s)

        state = system.reset()
        controller.reset()
        n_steps = 5000  # 0.5 s
        omega_trace = np.zeros(n_steps)
        for k in range(n_steps):
            action = controller.control(state, omega_ref)
            state = system.simulate(action)
            omega_trace[k] = (state * system.limits)[system.OMEGA_IDX]

        settled = omega_trace[-500:]
        assert np.mean(settled) == pytest.approx(omega_ref, rel=0.05)
        assert np.std(settled) < 0.05 * omega_ref

    def test_tracks_different_reference_levels(self):
        for omega_ref in (50.0, 150.0, 250.0):
            system = build_system()
            controller = PICascadeController(system)
            state = system.reset()
            controller.reset()
            for _ in range(5000):
                action = controller.control(state, omega_ref)
                state = system.simulate(action)
            final_omega = (state * system.limits)[system.OMEGA_IDX]
            assert final_omega == pytest.approx(omega_ref, rel=0.08)

    def test_action_stays_within_converter_bounds(self):
        system = build_system()
        controller = PICascadeController(system)
        state = system.reset()
        controller.reset()
        for _ in range(2000):
            action = controller.control(state, omega_ref=300.0)
            assert 0.0 <= action[0] <= 1.0
            state = system.simulate(action)

    def test_reset_clears_integrators(self):
        system = build_system()
        controller = PICascadeController(system)
        state = system.reset()
        controller.reset()
        for _ in range(1000):
            action = controller.control(state, omega_ref=200.0)
            state = system.simulate(action)
        assert controller._omega_integral != 0.0 or controller._i_integral != 0.0
        controller.reset()
        assert controller._omega_integral == 0.0
        assert controller._i_integral == 0.0


class TestPICascadeControllerTorqueMode:
    """control_torque() bypasses the outer speed loop -- i_ref = torque_ref/psi_e feeds the same
    inner current loop control() uses. See dashboard.py's "Torque (τ_ref)" control-mode option."""

    def test_tracks_torque_reference(self):
        system = build_system()
        controller = PICascadeController(system)
        torque_ref = 10.0  # Nm, within the motor's nominal range (16 Nm)
        psi_e = system.electrical_motor.motor_parameter["psi_e"]

        state = system.reset()
        controller.reset()
        n_steps = 3000  # the current loop alone settles far faster than the outer speed loop did above
        i_trace = np.zeros(n_steps)
        for k in range(n_steps):
            action = controller.control_torque(state, torque_ref)
            state = system.simulate(action)
            i_trace[k] = (state * system.limits)[system.CURRENTS_IDX[0]]

        settled_torque = i_trace[-500:] * psi_e  # torque = psi_e * i_a for this motor (no reluctance term)
        assert np.mean(settled_torque) == pytest.approx(torque_ref, rel=0.05)

    def test_action_stays_within_converter_bounds(self):
        system = build_system()
        controller = PICascadeController(system)
        state = system.reset()
        controller.reset()
        for _ in range(1000):
            action = controller.control_torque(state, torque_ref=12.0)
            assert 0.0 <= action[0] <= 1.0
            state = system.simulate(action)

    def test_omega_is_free_running_not_tracking_a_setpoint(self):
        """Unlike control() (where different omega_ref values converge to those exact speeds),
        control_torque() has no speed reference at all -- different constant torques against the
        same load should settle at DIFFERENT equilibrium speeds (where applied torque balances the
        load's speed-dependent braking term), not converge to anything shared."""
        finals = []
        for torque_ref in (5.0, 15.0):
            system = build_system()
            controller = PICascadeController(system)
            state = system.reset()
            controller.reset()
            for _ in range(3000):
                action = controller.control_torque(state, torque_ref)
                state = system.simulate(action)
            finals.append((state * system.limits)[system.OMEGA_IDX])
        assert finals[0] != pytest.approx(finals[1], rel=0.05)
