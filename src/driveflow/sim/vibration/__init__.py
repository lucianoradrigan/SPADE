"""Vibration synthesis layer -- Module B only. Module C (a data-driven residual corrector) was
tried and retired; see "Why it looks like this" below and docs/patch2_retiro_modulo_C.md.

HOW ESTIMATION WORKS, END TO END
---------------------------------
Called once per control step, right after SCMLSystem.simulate(), via
VibrationSynthesizer.step(omega) -- omega (rad/s) is the ONLY plant signal this layer consumes::

    omega (rad/s, from SCMLSystem)
      |
      +-----------------------------+-----------------------------+
      v                                                            v
    background_noise.py                                    fault_impulses.py
    BackgroundNoiseGenerator                                FaultImpulseGenerator
    - white noise, RMS-scaled by                            - order-tracked impulse train at the
      `gain` (fit vs. healthy Paderborn                        fault's characteristic frequency
      recordings, calibration.py::                             (BPFO/BPFI/BSF/FTF, from omega +
      fit_background_noise_gain)                                bearing geometry)
    - always on, independent of                             - amplitude MODULATED by
      omega/torque/current/fault                              LoadZoneModulator: constant for
                                                                outer_race (defect position fixed
                                                                vs. the load zone), modulated at
                                                                shaft freq for inner_race / cage
                                                                freq for ball (defect rotates
                                                                through the load zone)
      |                                                            |
      +-----------------------------+-----------------------------+
                                     |
                                     v   (sum = excitation force F(t), no torque/current involved)
                        modal_model.py :: ModalAxis (x3, one per output axis)
                        bank of {natural_freq_hz, damping_ratio, gain} modes, calibrated ONLY
                        against healthy recordings (calibration.py, Paderborn K001-K006),
                        integrated via *exact* zero-order-hold discretization (scipy.linalg.expm)
                        -- stable regardless of mode frequency vs. substep size.
                                     |
                                     v
                        (acc_x, acc_y, acc_z) for this control step

WHY IT LOOKS LIKE THIS (not the addendum's original B+C design)
------------------------------------------------------------------
1. Module C (Keras residual corrector conditioned on torque/i_d/i_q/omega) was implemented,
   tried with 3 different techniques (pointwise MLP, GRU+envelope target, phase-aware FRF fit),
   and retired: measured coherence between real vibration and every electrical/mechanical signal
   available in SCMLSystem is ~0 on real KAt-DataCenter recordings. There was nothing for any
   model, linear or not, to learn. -- docs/patch2_retiro_modulo_C.md
2. The same coherence check, done specifically AT each fault's own characteristic frequency on
   FAULTED bearings, also came back non-specific (BSF read as high as a bearing's own BPFO) --
   ruling out torque as even a generic *excitation* source for Module B, not just as a corrector.
   That's why background_noise.py/fault_impulses.py take omega and severity only, never torque
   or current. -- docs/patch3_mejora_modulo_B.md
3. `severity` is calibrated, not a round number picked by hand: datagen/scenario.py's
   CALIBRATED_MECHANICAL_SEVERITY (per fault type) was found by matching synthetic-vs-real AUC
   separability (experiments/verify_vibration_separability_auc.py) on windowed envelope-band-energy features.
4. inner_race/ball impulses are amplitude-modulated (LoadZoneModulator, fault_impulses.py)
   because those defects physically rotate through the load zone once per shaft/cage revolution
   -- outer_race defects don't (fixed position relative to a fixed radial load). This is *why*
   inner_race is calibrated to a lower severity than outer_race, and it improved the synthetic
   AUC's match to the real one at the SAME already-calibrated severity, not just a nicer
   explanation. -- docs/patch4_modulacion_zona_carga.md

Known, documented limitations (not bugs):
- Only ONE axis is calibrated against real data (KAt-DataCenter has a single accelerometer
  channel); y/z reuse the calibrated axis with a documented gain/frequency/damping assumption
  (see datagen/runner.py's _AXIS_PROFILE), not independently measured data.
- Module B reproduces *relative* spectral content (energy rises in the right band when a fault is
  injected), not *absolute* vibration amplitude at a given operating point -- see
  docs/patch2_retiro_modulo_C.md Sec. 3/9 for why that's out of scope for v1.

Sits between Sim (SCMLSystem) and Datagen: called manually, NOT as a gymnasium PhysicalSystemWrapper
(driveflow does not use GEM's Gym-Env layer, see sim/physical_system.py).
"""

from .background_noise import BackgroundNoiseGenerator
from .bearing_frequencies import (
    KAT_DATACENTER_6203_GEOMETRY,
    BearingGeometry,
    fault_frequencies_hz,
    fault_order,
)
from .fault_impulses import FaultImpulseGenerator, ImpulseTrainGenerator, LoadZoneModulator
from .modal_model import Mode, ModalAxis, ModalVibrationModel
from .vibration_synthesizer import VibrationSynthesizer

__all__ = [
    "BearingGeometry",
    "KAT_DATACENTER_6203_GEOMETRY",
    "fault_frequencies_hz",
    "fault_order",
    "Mode",
    "ModalAxis",
    "ModalVibrationModel",
    "BackgroundNoiseGenerator",
    "FaultImpulseGenerator",
    "ImpulseTrainGenerator",
    "LoadZoneModulator",
    "VibrationSynthesizer",
]
