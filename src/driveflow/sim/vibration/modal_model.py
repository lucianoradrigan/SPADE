"""Module B: physical modal mass-spring-damper model of the bearing/housing structure.

Per ``docs/addendum_vibracion_v1.md`` Sec. 3, each mode ``k`` obeys::

    x_k''(t) + 2*zeta_k*omega_n,k*x_k'(t) + omega_n,k**2*x_k(t) = F(t) / m_k

This module integrates a bank of such modes per axis (x/y/z) and reports the resulting
acceleration. Calibration of ``{omega_n,k, zeta_k, gain_k}`` against real PSDs (KAt-DataCenter) is
``sim/vibration/calibration.py``; this module only needs the parameters, not the dataset.

Excitation F(t) is the sum of two independent, torque-free sources (``docs/patch3_mejora_modulo_B.md``):
``background_noise.BackgroundNoiseGenerator`` (always-on stochastic rolling-contact noise) and
``fault_impulses.FaultImpulseGenerator`` (order-tracked impulse train, only when a fault is set).
Earlier versions additionally coupled the excitation to torque ripple -- removed after Patch 3
found no exploitable torque->vibration relationship in KAt-DataCenter (broadband coherence 0.022;
see docs/patch2_retiro_modulo_C.md and docs/patch3_mejora_modulo_B.md for the full trail).

Numerical approach: each mode is a linear time-invariant 2-state system (position, velocity)
driven by F(t)/m. Rather than an explicit integrator (unstable for stiff/high-frequency modes
unless the step is very small), each mode is discretized *exactly* via the zero-order-hold matrix
exponential (``scipy.linalg.expm``) for the model's internal substep -- stable for any
(natural_freq, dt_sub) combination. ``ModalVibrationModel.step()`` is called once per
``SCMLSystem.simulate()`` (i.e. once per control tau) but internally oversamples into
``n_substeps`` to resolve mode frequencies and fault impulses far above 1/tau.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from . import bearing_frequencies
from .background_noise import BackgroundNoiseGenerator
from .fault_impulses import CompositeFaultImpulseGenerator, FaultImpulseGenerator


@dataclass(frozen=True)
class Mode:
    """One vibration mode.

    Args:
        natural_freq_hz: Undamped natural frequency omega_n,k / (2*pi).
        damping_ratio: zeta_k (dimensionless, 0 < zeta_k < 1 for an underdamped mode).
        gain: Modal participation factor, i.e. 1/m_k in the addendum's equation -- converts the
            shared excitation force F(t) into this mode's generalized force.
    """

    natural_freq_hz: float
    damping_ratio: float
    gain: float = 1.0

    def __post_init__(self):
        if self.natural_freq_hz <= 0:
            raise ValueError("natural_freq_hz must be positive")
        if not (0.0 < self.damping_ratio < 1.0):
            raise ValueError("damping_ratio must be in (0, 1) for an underdamped mode")


class ModalAxis:
    """A bank of modes contributing to the acceleration measured along one axis."""

    def __init__(self, modes):
        self.modes = list(modes)
        if not self.modes:
            raise ValueError("ModalAxis needs at least one Mode")
        self._state = np.zeros((len(self.modes), 2))
        self._cache = {}

    def reset(self):
        self._state[:] = 0.0

    def _discretize(self, dt_sub: float):
        key = round(dt_sub, 12)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        discretized = []
        for mode in self.modes:
            wn = 2.0 * np.pi * mode.natural_freq_hz
            augmented = np.zeros((3, 3))
            augmented[0, 1] = 1.0
            augmented[1, 0] = -(wn**2)
            augmented[1, 1] = -2.0 * mode.damping_ratio * wn
            augmented[1, 2] = mode.gain
            discrete = expm(augmented * dt_sub)
            a_d = discrete[:2, :2]
            b_d = discrete[:2, 2]
            discretized.append((a_d, b_d, wn))
        self._cache[key] = discretized
        return discretized

    def step(self, force_samples: np.ndarray, dt_sub: float) -> float:
        """Integrates all modes across the substeps of ``force_samples`` and returns the summed
        acceleration at the last sample.

        Args:
            force_samples: Excitation force F(t), substep-sampled, length n_substeps + 1.
            dt_sub: Substep duration (s).
        """
        force_samples = np.asarray(force_samples, dtype=float)
        discretized = self._discretize(dt_sub)
        accel = 0.0
        for i, (a_d, b_d, wn) in enumerate(discretized):
            mode = self.modes[i]
            x = self._state[i]
            for k in range(len(force_samples) - 1):
                x = a_d @ x + b_d * force_samples[k]
            self._state[i] = x
            accel += mode.gain * force_samples[-1] - 2.0 * mode.damping_ratio * wn * x[1] - wn**2 * x[0]
        return float(accel)


class ModalVibrationModel:
    """Module B: orchestrates the 3-axis modal bank plus the two excitation sources (background
    noise + fault impulses). This is the physical, calibrated-against-real-data half of the
    vibration layer -- Module C (data-driven residual correction) was retired, see
    ``docs/patch2_retiro_modulo_C.md``.
    """

    def __init__(
        self,
        axes: dict,
        geometry: bearing_frequencies.BearingGeometry,
        n_substeps: int = 20,
        background_gain: float = 1.0,
        seed=None,
    ):
        """
        Args:
            axes: {"x": ModalAxis, "y": ModalAxis, "z": ModalAxis}, one modal bank per axis.
            geometry: Bearing geometry used to translate fault types into characteristic orders.
            n_substeps: Internal oversampling factor per ``step()`` call (per control tau).
            background_gain: RMS scale of the background noise generator (see
                ``background_noise.BackgroundNoiseGenerator``, fit against healthy recordings).
            seed: Seed for the background noise generator's RNG (reproducibility).
        """
        missing = {"x", "y", "z"} - set(axes)
        if missing:
            raise ValueError(f"axes is missing entries for {missing}")
        self.axes = axes
        self.geometry = geometry
        self.n_substeps = n_substeps
        self.background_gen = BackgroundNoiseGenerator(gain=background_gain, seed=seed)
        self.fault_gen = FaultImpulseGenerator(geometry, fault_type=None, severity=0.0)
        self._prev_omega = 0.0

    def reset(self, seed=None):
        for axis in self.axes.values():
            axis.reset()
        self.background_gen.reset(seed=seed)
        self.fault_gen.reset()
        self._prev_omega = 0.0

    def set_fault(self, fault_type: str = None, severity: float = 0.0, order_override: float = None, extra_faults: list = None):
        """Configures (or clears, with ``fault_type=None`` and no extras) the bearing fault(s) to
        inject.

        Args:
            fault_type: One of ``bearing_frequencies.FAULT_FREQUENCY_FUNCS`` keys
                (``"outer_race"``, ``"inner_race"``, ``"ball"``, ``"cage"``), or None for a
                healthy bearing. Any other string is accepted (as a label) when order_override
                is given.
            severity: Impulse-train amplitude (0 = no fault).
            order_override: See ``FaultImpulseGenerator``'s docstring -- a user-defined fault
                type's characteristic order, bypassing the geometry-based lookup.
            extra_faults: Optional list of (fault_type, severity, order_override) triples for
                ADDITIONAL simultaneous faults, layered on top of the primary above -- see
                ``CompositeFaultImpulseGenerator``. None/empty (default): unchanged single-fault
                behavior (a plain FaultImpulseGenerator, not a composite of one).
        """
        if not extra_faults:
            self.fault_gen = FaultImpulseGenerator(self.geometry, fault_type=fault_type, severity=severity, order_override=order_override)
            return
        generators = [FaultImpulseGenerator(self.geometry, fault_type=fault_type, severity=severity, order_override=order_override)]
        generators += [FaultImpulseGenerator(self.geometry, fault_type=ft, severity=sev, order_override=oo) for ft, sev, oo in extra_faults]
        self.fault_gen = CompositeFaultImpulseGenerator(generators)

    def step(self, omega: float, dt: float) -> np.ndarray:
        """Advances the modal model by one control step and returns the synthesized acceleration.

        Args:
            omega: Mechanical angular speed (rad/s) at the end of this control step.
            dt: Control step duration (s), i.e. SCMLSystem's tau.

        Returns:
            ndarray(float), shape (3,): (acc_x, acc_y, acc_z).
        """
        dt_sub = dt / self.n_substeps
        omega_samples = np.linspace(self._prev_omega, omega, self.n_substeps + 1)
        self._prev_omega = omega

        excitation = self.background_gen.step(self.n_substeps + 1) + self.fault_gen.step(omega_samples, dt_sub)
        return np.array([self.axes[axis].step(excitation, dt_sub) for axis in ("x", "y", "z")])
