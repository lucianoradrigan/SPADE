"""Fault impulse train: shared by both fault paths -- mechanical (Module B, this module's
``FaultImpulseGenerator``, consumed by ``modal_model.py``) and electrical/MCSA
(``datagen/fault_injection.py``'s ``BearingFaultLoad``, which uses ``ImpulseTrainGenerator``
directly) -- one order-tracked impulse-train implementation, not duplicated, per
``docs/addendum_vibracion_v1.md`` Sec. 3, point 1.

Depends only on shaft speed (kinematic, calibration-free -- a fault's characteristic order is a
fixed multiple of the shaft frequency, see ``bearing_frequencies.py``) and severity, a scenario
parameter set by ``datagen`` -- not inferred from any electrical signal. Patch 3
(``docs/patch3_mejora_modulo_B.md``) confirmed this only *after* checking: coherence at each
bearing's own fault frequency (BPFO for an outer-race defect, BPFI for inner-race) is NOT
distinguishable from coherence at the *other*, wrong fault frequencies for that same bearing
(e.g. an outer-race-only defect (KA*) shows as much/more coherence at BSF, which it does not
have, as at its own BPFO) -- ruling out a fault-type-specific torque modulator, not just a
generic torque dependency.
"""

import numpy as np

from . import bearing_frequencies as bf


class ImpulseTrainGenerator:
    """Order-tracked impulse train for one bearing-fault characteristic frequency.

    Stateful across calls (the phase accumulator persists), so it must be driven with
    contiguous, chronologically-ordered sample windows -- exactly how ``FaultImpulseGenerator``
    (mechanical path) and ``BearingFaultLoad`` (electrical path) call it once per control step.
    """

    def __init__(self, order: float):
        """
        Args:
            order: Fault characteristic frequency expressed as a multiple of the shaft frequency
                (see ``bearing_frequencies.fault_order``).
        """
        if order <= 0:
            raise ValueError("order must be positive")
        self.order = float(order)
        self._phase = 0.0

    def reset(self):
        self._phase = 0.0

    def generate(self, omega_samples: np.ndarray, dt_sub: float) -> np.ndarray:
        """
        Args:
            omega_samples: Mechanical angular speed (rad/s) sampled at the substep times of the
                current integration window (length n_substeps + 1, the last sample belonging to
                the *next* window's start and only used to know the instantaneous rate there).
            dt_sub: Substep duration (s).

        Returns:
            ndarray(float), shape (len(omega_samples),): unit-area impulses (amplitude
            ``1/dt_sub`` where an impulse fires, 0 elsewhere).
        """
        omega_samples = np.asarray(omega_samples, dtype=float)
        f_r = omega_samples / (2.0 * np.pi)
        pulses = np.zeros_like(omega_samples)
        for k in range(len(omega_samples)):
            self._phase += self.order * abs(f_r[k]) * dt_sub
            if self._phase >= 1.0:
                self._phase -= 1.0
                pulses[k] = 1.0 / dt_sub
        return pulses


class LoadZoneModulator:
    """Amplitude-modulation envelope for the fault impulse train -- Patch 4
    (``docs/patch4_modulacion_zona_carga.md``), extending the plain order-tracked impulse train
    per the classic McFadden & Smith (1984) model: whether a rolling-element strike on the defect
    actually loads the structure depends on the defect's position relative to the (fixed) radial
    load direction, not just on shaft speed.

    - outer_race: the defect's position relative to the load zone does not change over time (both
      are fixed in the housing/bearing frame for a fixed radial load) -> impacts have roughly
      constant amplitude. This is the standard textbook explanation for why outer-race defects
      are the easier of the two to detect -- and matches what Patch 2's calibrated severities
      already found empirically (outer_race 0.05 vs inner_race 0.02) without knowing why.
    - inner_race: the defect rotates with the shaft through the (fixed) load zone once per shaft
      revolution -> amplitude modulated at the shaft frequency, producing sidebands at f_r around
      BPFI (the classic inner-race-defect signature in real spectra).
    - ball: the defective rolling element circles the cage (at FTF) carrying its defect through
      the load zone once per cage revolution -> amplitude modulated at FTF.
    - cage: no established load-zone modulation model found in the literature reviewed for this
      patch; left unmodulated (same as outer_race) rather than guessing at one.

    Modeled as a half-wave-rectified raised cosine (nonzero/near-1 only while the defect is
    within the loaded arc, ~0 outside it) -- ``sharpness`` controls how narrow that arc is.
    """

    def __init__(self, fault_type: str, geometry: bf.BearingGeometry, sharpness: float = 2.0):
        self.fault_type = fault_type
        self.sharpness = sharpness
        self._phase = 0.0
        if fault_type == "inner_race":
            self._mod_order = 1.0  # the shaft frequency itself
        elif fault_type == "ball":
            self._mod_order = bf.fault_order("cage", geometry)  # FTF, as a multiple of f_r
        else:
            self._mod_order = None  # outer_race, cage: no modulation (see class docstring)

    def reset(self):
        self._phase = 0.0

    def envelope(self, omega_samples: np.ndarray, dt_sub: float) -> np.ndarray:
        omega_samples = np.asarray(omega_samples, dtype=float)
        if self._mod_order is None:
            return np.ones_like(omega_samples)
        f_r = omega_samples / (2.0 * np.pi)
        env = np.ones_like(omega_samples)
        for k in range(len(omega_samples)):
            self._phase += self._mod_order * abs(f_r[k]) * dt_sub
            self._phase %= 1.0
            raised = max(0.0, np.cos(2.0 * np.pi * self._phase))
            env[k] = raised**self.sharpness
        return env


class FaultImpulseGenerator:
    """Module B's fault excitation: wraps ``ImpulseTrainGenerator`` (impulse timing) and
    ``LoadZoneModulator`` (impulse amplitude) with the bearing geometry -> order lookup and
    severity scaling, so ``modal_model.ModalVibrationModel`` only deals with
    ``(fault_type, severity)``, not raw orders.
    """

    def __init__(
        self,
        geometry: bf.BearingGeometry,
        fault_type: str = None,
        severity: float = 0.0,
        load_zone_modulation: bool = True,
        order_override: float = None,
    ):
        """
        Args:
            geometry: Bearing geometry used to compute the fault's characteristic order (ignored
                when order_override is given).
            fault_type: One of ``bearing_frequencies.FAULT_FREQUENCY_FUNCS`` keys, or None for a
                healthy bearing (step() then always returns zeros). Any non-None string is
                accepted (just a label) when order_override is given -- it does not need to be a
                recognized key in that case.
            severity: Impulse-train peak amplitude (0 = no fault, even if fault_type is set).
            load_zone_modulation: Whether to apply LoadZoneModulator (see its docstring). Exposed
                as a flag mainly so tests/experiments can isolate its effect by disabling it.
                Forced off when order_override is given (see below).
            order_override: Characteristic frequency as a multiple of shaft speed, bypassing
                bearing_frequencies.fault_order()'s geometry-based lookup entirely -- how a
                user-defined fault type (dashboard's "Custom fault types") is injected without
                needing real bearing geometry for it: gear-mesh orders, a different bearing's
                known BPFO, unbalance (order=1), misalignment (order=2), etc. LoadZoneModulator is
                never applied in this case: its amplitude envelope depends on where a defect sits
                relative to the fixed radial load direction (see its docstring), physics that
                doesn't exist for an arbitrary order with no known defect geometry -- constant
                amplitude instead, the same treatment already used for outer_race/cage.
        """
        self.geometry = geometry
        self.fault_type = fault_type
        self.severity = float(severity)
        self.order_override = order_override
        if order_override is not None:
            self._generator = ImpulseTrainGenerator(order_override)
            self._modulator = None
        elif fault_type is not None:
            self._generator = ImpulseTrainGenerator(bf.fault_order(fault_type, geometry))
            self._modulator = LoadZoneModulator(fault_type, geometry) if load_zone_modulation else None
        else:
            self._generator = None
            self._modulator = None

    def reset(self):
        if self._generator is not None:
            self._generator.reset()
        if self._modulator is not None:
            self._modulator.reset()

    def step(self, omega_samples: np.ndarray, dt_sub: float) -> np.ndarray:
        omega_samples = np.asarray(omega_samples, dtype=float)
        if self._generator is None or self.severity == 0.0:
            return np.zeros_like(omega_samples)
        pulses = self._generator.generate(omega_samples, dt_sub)
        if self._modulator is not None:
            pulses = pulses * self._modulator.envelope(omega_samples, dt_sub)
        return self.severity * pulses


class CompositeFaultImpulseGenerator:
    """Combines multiple independent FaultImpulseGenerator instances -- each its own fault_type/
    order/severity/modulation -- into one, by summing their excitation additively. This is how
    the mechanical/vibration path injects more than one simultaneous periodic fault (dashboard's
    "Combine with additional fault types", for compound/mixed-fault scenarios) -- built entirely
    out of unmodified FaultImpulseGenerator instances, each keeping its own phase state, so faults
    at different orders don't interfere with each other's timing. Duck-types the same
    reset()/step() interface as a plain FaultImpulseGenerator, so ModalVibrationModel.fault_gen
    doesn't need to know which one it holds.
    """

    def __init__(self, generators: list):
        self._generators = generators

    def reset(self):
        for g in self._generators:
            g.reset()

    def step(self, omega_samples: np.ndarray, dt_sub: float) -> np.ndarray:
        omega_samples = np.asarray(omega_samples, dtype=float)
        total = np.zeros_like(omega_samples)
        for g in self._generators:
            total = total + g.step(omega_samples, dt_sub)
        return total
