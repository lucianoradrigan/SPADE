"""Electrical fault-injection path: BearingFaultLoad, per docs/addendum_vibracion_v1.md Sec. 3
("Camino electrico (MCSA)"): a bearing fault's characteristic frequency modulates the load
torque, which couples electromechanically into the simulated phase current(s) -- the classic
Motor Current Signature Analysis (MCSA) sidebands the paper_federative classifiers were designed
to detect from real current channels.

Reuses ``sim/vibration/bearing_frequencies.py`` for the characteristic-frequency math (shared
with the mechanical path, Module B) instead of duplicating it -- exactly the sharing point called
out in the addendum's Sec. 3, point 1.
"""

import numpy as np

from driveflow.sim.loads import MechanicalLoad
from driveflow.sim.vibration import bearing_frequencies as bf

#: Weights of the ripple's harmonic series (fundamental + 3 overtones of the fault order). Each
#: cos(k*phase) term is exactly zero-mean over one period, so their weighted sum is too -- no
#: systematic bias is introduced in the average load torque regardless of severity. The weights
#: taper off to make the waveform peakier ("impulse-like") than a pure sinusoid, closer to the
#: sharp periodic impacts a real localized bearing defect produces, without the discontinuities
#: of an actual impulse (which would be awkward for an ODE solver to integrate through).
_HARMONIC_WEIGHTS = (1.0, 0.35, 0.15, 0.07)


class BearingFaultLoad(MechanicalLoad):
    """Wraps a base MechanicalLoad and adds order-tracked torque ripple at a bearing fault's
    characteristic frequency.

    The fault phase is tracked as an extra ODE state (``fault_phase``, dphase/dt = order*omega),
    not as an externally-advanced accumulator -- this is what makes it safe to call from inside
    an adaptive-step ODE solver, which may evaluate ``mechanical_ode`` at arbitrary/repeated
    times within one control step (a phase accumulator driven by explicit dt substeps, like
    ``sim.vibration.fault_impulses.ImpulseTrainGenerator``, would desync under that access
    pattern -- that generator is designed for VibrationSynthesizer's own single-pass manual
    stepping loop instead, see modal_model.py).
    """

    HAS_JACOBIAN = False

    def __init__(
        self,
        base_load: MechanicalLoad,
        fault_type: str,
        geometry: bf.BearingGeometry,
        severity: float = 0.0,
        order_override: float = None,
        extra_faults: list = None,
    ):
        """
        Args:
            base_load: The underlying healthy-bearing load (e.g. PolynomialStaticLoad).
            fault_type: One of bearing_frequencies.FAULT_FREQUENCY_FUNCS' keys, or None for a
                healthy bearing (ripple identically zero, base_load's dynamics pass through
                unchanged). Any other non-None string is accepted (as a label) when
                order_override is given.
            geometry: Bearing geometry used to compute the fault's characteristic order (ignored
                when order_override is given).
            severity: Approximate peak amplitude (Nm) of the injected torque ripple.
            order_override: Characteristic frequency as a multiple of shaft speed, bypassing
                bearing_frequencies.fault_order()'s geometry-based lookup -- see
                sim/vibration/fault_impulses.py's FaultImpulseGenerator docstring for the same
                mechanism on the mechanical/vibration path (dashboard's "Custom fault types").
            extra_faults: Optional list of (fault_type, severity, order_override) triples for
                ADDITIONAL simultaneous faults, layered on top of the primary fault above --
                dashboard's "Combine with additional fault types" (compound/mixed-fault
                scenarios). Each triple is resolved to an order the same way the primary is
                (order_override if given, else a geometry lookup). Every active fault (primary +
                extras) gets its own independent ODE phase state, so faults at different orders
                don't interfere with each other's timing, and contributes additively to the total
                torque ripple. None/empty (default): behaves exactly as before this parameter
                existed -- a single "fault_phase" state, not one named "fault_phase_0".
        """
        self.extra_faults = list(extra_faults) if extra_faults else []
        n_phases = 1 if not self.extra_faults else 1 + len(self.extra_faults)
        # Single-fault case keeps the original "fault_phase" name (not "fault_phase_0") --
        # nothing about the common, original single-fault path changes when extras aren't used.
        phase_names = ["fault_phase"] if not self.extra_faults else [f"fault_phase_{i}" for i in range(n_phases)]
        super().__init__(state_names=list(base_load.state_names) + phase_names, j_load=base_load.j_total)
        # MechanicalLoad.__init__ starts self._limits/_nominal_values empty; the wrapped
        # base_load's own values (e.g. PolynomialStaticLoad(limits=dict(omega=150.0))) must be
        # carried over explicitly since we do not re-run base_load's __init__.
        self._limits.update(base_load.limits)
        self._nominal_values.update(base_load.nominal_values)
        self.base_load = base_load
        self.fault_type = fault_type
        self.geometry = geometry
        self.severity = severity
        self.order_override = order_override
        if order_override is not None:
            self._order = order_override
        else:
            self._order = bf.fault_order(fault_type, geometry) if fault_type is not None else None
        # (order, severity) per active fault, primary first -- only consulted by mechanical_ode's
        # multi-fault branch (self.extra_faults non-empty); the single-fault branch uses
        # self._order/self.severity directly, unchanged from before this feature existed.
        self._all_faults = [(self._order, self.severity)] + [
            (extra_oo if extra_oo is not None else (bf.fault_order(ft, geometry) if ft is not None else None), sev)
            for ft, sev, extra_oo in self.extra_faults
        ]

    def set_j_rotor(self, j_rotor):
        super().set_j_rotor(j_rotor)
        self.base_load.set_j_rotor(j_rotor)

    def get_state_space(self, omega_range):
        # fault_phase intentionally omitted here: it is an internal bookkeeping state (grows
        # unboundedly over a long run), not a physically-limited quantity like omega. Left out of
        # the returned dict, set_state_array leaves its Box bounds at the default [0, 0] -- this
        # never clips or otherwise affects the actual ODE integration (Box is descriptive/for
        # random-init interval logic only, which fault_phase never uses -- see reset()).
        return self.base_load.get_state_space(omega_range)

    def reset(self, state_space, state_positions, nominal_state, **__):
        self.next_generator()
        base_state = np.atleast_1d(self.base_load.reset(state_space, state_positions, nominal_state))
        n_phases = 1 if not self.extra_faults else 1 + len(self.extra_faults)
        return np.concatenate([base_state, np.zeros(n_phases)])

    def _ripple_torque(self, phase: float) -> float:
        """Ripple from the primary fault alone, at one phase. Used directly by mechanical_ode's
        single-fault branch (no extra_faults, the common/original case), and still meaningful
        standalone (e.g. in tests) even when extras are configured -- it only ever describes the
        primary; see _ripple_torque_for for the general per-fault version mechanical_ode's
        multi-fault branch uses for every active fault including this one."""
        if self._order is None:
            return 0.0
        waveform = sum(w * np.cos((k + 1) * phase) for k, w in enumerate(_HARMONIC_WEIGHTS))
        return self.severity * waveform

    @staticmethod
    def _ripple_torque_for(order, severity, phase: float) -> float:
        if order is None:
            return 0.0
        waveform = sum(w * np.cos((k + 1) * phase) for k, w in enumerate(_HARMONIC_WEIGHTS))
        return severity * waveform

    def mechanical_ode(self, t, mechanical_state, torque):
        n_base = len(self.base_load.state_names)
        base_state = mechanical_state[:n_base]
        omega = base_state[self.base_load.OMEGA_IDX]

        if not self.extra_faults:
            phase = mechanical_state[n_base]
            ripple = self._ripple_torque(phase)
            base_derivative = np.atleast_1d(self.base_load.mechanical_ode(t, base_state, torque - ripple))
            phase_derivative = (self._order if self._order is not None else 0.0) * omega
            return np.concatenate([base_derivative, [phase_derivative]])

        phases = mechanical_state[n_base:]
        ripple = sum(self._ripple_torque_for(order, severity, phase) for (order, severity), phase in zip(self._all_faults, phases))
        base_derivative = np.atleast_1d(self.base_load.mechanical_ode(t, base_state, torque - ripple))
        phase_derivatives = [(order if order is not None else 0.0) * omega for order, severity in self._all_faults]
        return np.concatenate([base_derivative, phase_derivatives])
