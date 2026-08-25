"""Native cascaded PI speed+current controller for a DcMotorSystem.

NOT a port of gem_controllers. GEM's own PI/PID cascade (``gem_controllers.PICurrentController``
/ ``PISpeedController`` / ``TorqueController``, used via ``GemController.make(env, env_id, ...)``)
is built around the full gymnasium ``ElectricMotorEnvironment``: it reads ``env.unwrapped.
physical_system``, calls ``env.get_wrapper_attr(...)``, and dispatches motor-specific behavior by
parsing the registered ``env_id`` string (e.g. "Finite-SC-PermExDc-v0") via
``gem_controllers.utils.get_motor_type``. None of that exists for a bare, decoupled ``SCMLSystem``
(Principle of design #2 -- see A.2's note in the Macro-fase A closeout).

This instead implements the same well-known cascade-tuning method GEM's controllers use --
magnitude optimum ("Betragsoptimum") for the inner current loop, symmetric optimum for the outer
speed loop -- directly against a ``DcMotorSystem``, auto-tuned from the motor's own r_a/l_a/psi_e
and the load's j_total. This is standard DC-motor control theory (not GEM-specific code).
"""

import numpy as np


class PICascadeController:
    """Speed-outer / current-inner PI cascade, with back-EMF feedforward decoupling and simple
    conditional anti-windup (integration freezes while the corresponding output is saturated).

    Only supports DcMotorSystem (a single current/voltage channel, unipolar converter action in
    [0, 1]) -- matches the DC systems used throughout Macro-fase A's regression/vibration tests.
    """

    def __init__(
        self,
        system,
        speed_bandwidth_rad_s: float = 80.0,
        speed_damping: float = 0.9,
        current_safety_margin: float = 0.8,
    ):
        """
        Args:
            system: A DcMotorSystem instance (already constructed with its converter/motor/load/
                supply/solver -- this controller reads its parameters, it does not own the plant).
            speed_bandwidth_rad_s: Desired closed speed-loop natural frequency (rad/s). Must stay
                well below the current loop's implicit bandwidth (~1/(2*tau), i.e. tens of kHz for
                a typical tau=1e-4) for the "ideal inner loop" approximation below to hold.
            speed_damping: Desired closed speed-loop damping ratio (0.9 -> essentially no
                overshoot).
            current_safety_margin: Fraction of the current limit the speed loop is allowed to
                command as a reference (keeps the operating point away from the motor's absolute
                current limit).
        """
        motor_params = system.electrical_motor.motor_parameter
        r_a = motor_params["r_a"]
        l_a = motor_params["l_a"]
        self.psi_e = motor_params["psi_e"]
        j_total = system.mechanical_load.j_total
        self.tau = system.tau

        # Current loop, magnitude optimum: plant u -> i ~= (1/r_a) / (1 + s*l_a/r_a), treating tau
        # (the discrete control period) as the loop's unmodeled small delay. Controller zero
        # Tn_i = l_a/r_a cancels the electrical pole; the resulting loop gain is set for a
        # maximally-flat magnitude response (the standard "Betragsoptimum" result K*tau = 1/2).
        t_sigma = self.tau
        self.kp_i = l_a / (2 * t_sigma)
        self.ki_i = r_a / (2 * t_sigma)

        # Speed loop: pole placement, not a from-memory "symmetric optimum" formula (an earlier
        # attempt at that produced a wildly oversized ki_omega -- always-saturated current
        # reference, so the closed loop converged to the same speed regardless of the setpoint;
        # this version's poles are placed directly and were checked numerically to give sane gain
        # magnitudes before being adopted). The current loop's bandwidth (~1/(2*tau), tens of kHz)
        # is far above any sensible speed-loop bandwidth, so it is approximated as ideal
        # (i == i_ref) for this design step -- valid since l_a/r_a is ~100x smaller than the
        # mechanical time constant j_total*r_a/psi_e**2 for this motor. Plant (i_ref -> omega) is
        # then the pure integrator psi_e/(j_total*s); with controller C(s) = Kp*(1+s*Tn)/(s*Tn),
        # the closed loop's characteristic polynomial is s**2 + Kp*K2*s + Kp*K2/Tn (K2 =
        # psi_e/j_total), matched term-by-term to s**2 + 2*zeta*wn*s + wn**2.
        k2 = self.psi_e / j_total
        wn = speed_bandwidth_rad_s
        self.kp_omega = 2 * speed_damping * wn / k2
        tn_omega = self.kp_omega * k2 / wn**2
        self.ki_omega = self.kp_omega / tn_omega

        self.limits = system.limits
        self.omega_idx = system.OMEGA_IDX
        self.current_idx = system.CURRENTS_IDX[0]
        self.u_sup_idx = system.U_SUP_IDX[0]
        self.current_safety_margin = current_safety_margin

        self._omega_integral = 0.0
        self._i_integral = 0.0

    def reset(self):
        self._omega_integral = 0.0
        self._i_integral = 0.0

    def control(self, state: np.ndarray, omega_ref: float) -> np.ndarray:
        """
        Args:
            state: Normalized SCMLSystem state (as returned by reset()/simulate()).
            omega_ref: Target mechanical speed (rad/s).

        Returns:
            ndarray(float), shape (1,): duty-cycle action for a Cont*QuadrantConverter, in [0, 1].
        """
        physical = state * self.limits
        omega = physical[self.omega_idx]
        i = physical[self.current_idx]
        u_sup = physical[self.u_sup_idx]

        # Outer loop: speed error -> current reference.
        omega_error = omega_ref - omega
        i_max = self.limits[self.current_idx] * self.current_safety_margin
        i_ref_raw = self.kp_omega * omega_error + self._omega_integral
        i_ref = float(np.clip(i_ref_raw, -i_max, i_max))
        if i_ref == i_ref_raw:
            self._omega_integral += self.ki_omega * omega_error * self.tau

        return self._inner_current_loop(i_ref, i, omega, u_sup)

    def control_torque(self, state: np.ndarray, torque_ref: float) -> np.ndarray:
        """Torque-control variant: bypasses the outer speed loop entirely and derives i_ref
        directly from torque_ref (Nm) via i_ref = torque_ref / psi_e -- the same linear
        torque/current relation datagen/runner.py uses to report torque_nm for this
        permanently-excited motor (torque = psi_e * i_a, no reluctance term unlike the PMSM).
        The inner current loop below is identical to control()'s -- only how i_ref is computed
        differs, so the two modes share the same current-loop tuning/behavior.

        Args:
            state: Normalized SCMLSystem state (as returned by reset()/simulate()).
            torque_ref: Target torque (Nm). Positive/negative for motoring in either direction.

        Returns:
            ndarray(float), shape (1,): duty-cycle action for a Cont*QuadrantConverter, in [0, 1].
        """
        physical = state * self.limits
        omega = physical[self.omega_idx]
        i = physical[self.current_idx]
        u_sup = physical[self.u_sup_idx]

        i_max = self.limits[self.current_idx] * self.current_safety_margin
        i_ref = float(np.clip(torque_ref / self.psi_e, -i_max, i_max))

        return self._inner_current_loop(i_ref, i, omega, u_sup)

    def _inner_current_loop(self, i_ref: float, i: float, omega: float, u_sup: float) -> np.ndarray:
        """Current error -> voltage command, with back-EMF feedforward decoupling (same technique
        as GEM's EMFFeedforward stage -- standard cascade-control practice). Shared by control()
        and control_torque(): both ultimately just pick a different i_ref."""
        i_error = i_ref - i
        u_raw = self.kp_i * i_error + self._i_integral + self.psi_e * omega
        u = float(np.clip(u_raw, 0.0, u_sup))
        if u == u_raw:
            self._i_integral += self.ki_i * i_error * self.tau

        duty_cycle = float(np.clip(u / u_sup, 0.0, 1.0))
        self.last_i_ref = i_ref  # the commanded current reference, for diagnostics/plots
        return np.array([duty_cycle])
