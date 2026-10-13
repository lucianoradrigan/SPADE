"""Display-grade spectral analysis for the dashboard's Spectrum / FFT tab.

Kept in the library (not in viz/) so the numbers shown in the UI are unit-tested numerics, not
inline chart code. Single-sided amplitude spectrum with a Hann window and coherent-gain
correction: a sinusoid of amplitude A at a bin-centred frequency reads ~A in the result, in the
signal's own units (A for current, the synthetic acceleration unit for acc_*).
"""

from __future__ import annotations

import numpy as np


def amplitude_spectrum(signal, fs_hz: float, skip_fraction: float = 0.0, detrend: bool = True) -> tuple:
    """Returns (freqs_hz, amplitude). `skip_fraction` drops the leading part of the record (e.g.
    a start-up transient) before analysis. NaNs make the spectrum undefined -> ValueError."""
    x = np.asarray(signal, dtype=float)
    if not 0.0 <= skip_fraction < 1.0:
        raise ValueError("skip_fraction must be in [0, 1)")
    x = x[int(len(x) * skip_fraction):]
    if len(x) < 8:
        raise ValueError("need at least 8 samples for a spectrum")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN/Inf")
    if detrend:
        x = x - x.mean()
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    spec[0] /= 2.0
    if len(x) % 2 == 0:
        spec[-1] /= 2.0
    return np.fft.rfftfreq(len(x), d=1.0 / fs_hz), spec


def dominant_peak(freqs, amplitude, fmin_hz: float = 1.0) -> tuple:
    """(frequency_hz, amplitude) of the largest bin above fmin_hz, or (None, None)."""
    freqs, amplitude = np.asarray(freqs), np.asarray(amplitude)
    mask = freqs >= fmin_hz
    if not mask.any():
        return None, None
    i = int(np.argmax(np.where(mask, amplitude, -np.inf)))
    return float(freqs[i]), float(amplitude[i])
