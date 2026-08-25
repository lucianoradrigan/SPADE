"""Module B's stochastic background excitation -- rolling-contact noise.

Deliberately NOT a function of torque, current, speed or load. See
``docs/patch3_mejora_modulo_B.md`` for the full investigation: torque's broadband coherence with
real vibration is ~0.022 (healthy bearings); speed/force intra-run coherence is ~0.03-0.04
(same order); and even the 9.6-123 Hz band that reads elevated in FAULTED bearings (0.32-0.45,
Paso 0) is *already* elevated in healthy ones (0.168, Paso 3b) -- a generic rotation-harmonic
artifact present regardless of fault state, not something worth modeling via those signals here.

White noise, shaped by the SAME calibrated modal filter (``modal_model.ModalAxis``) that also
shapes the fault impulse train (``fault_impulses.py``) -- both excitations pass through the same
physical structure (housing/bearing) before reaching the sensor, so reusing one filter for both
is the physically correct choice (see ``calibration.py``'s module docstring, "Paso 2").
"""

import numpy as np


class BackgroundNoiseGenerator:
    """Zero-mean Gaussian white noise, RMS-scaled by ``gain``.

    ``gain`` is fit against healthy KAt-DataCenter recordings so that, once shaped by the
    calibrated modal filter, the resulting acceleration's RMS matches real healthy vibration (see
    ``calibration.py::fit_background_noise_gain`` / ``experiments/calibrate_module_b.py``).
    """

    def __init__(self, gain: float = 1.0, seed=None):
        self.gain = float(gain)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        """Re-seeds the generator. With seed=None, draws a fresh (non-reproducible) sequence --
        pass an explicit seed for reproducible runs, matching SCMLSystem.seed()'s convention."""
        self._rng = np.random.default_rng(seed)

    def step(self, n_samples: int) -> np.ndarray:
        """Returns n_samples of i.i.d. Gaussian white noise, scaled by gain."""
        return self.gain * self._rng.normal(0.0, 1.0, size=n_samples)
