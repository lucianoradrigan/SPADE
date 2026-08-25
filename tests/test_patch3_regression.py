"""Patch 3 regression test (docs/patch3_mejora_modulo_B.md): with severity=0, Module B's output
must be pure background noise -- no peaks at any bearing fault frequency. With a fault set, peaks
must appear at the correct (BPFO/BPFI/BSF/FTF) frequency, growing with severity.

Uses ModalVibrationModel directly at constant omega (not the full PI-controlled scenario loop) for
a clean, deterministic frequency-domain check, isolated from plant/controller dynamics.
"""

import numpy as np
import pytest

from driveflow.sim.vibration import bearing_frequencies as bf
from driveflow.sim.vibration.modal_model import Mode, ModalAxis, ModalVibrationModel

GEOMETRY = bf.KAT_DATACENTER_6203_GEOMETRY
TAU = 1e-4
OMEGA = 2 * np.pi * 25.0  # constant shaft speed, 25 Hz (~1500 rpm), matches Paderborn's own runs


def build_model(background_gain=0.02, seed=0):
    axes = {
        axis: ModalAxis([Mode(natural_freq_hz=f, damping_ratio=0.03) for f in (800.0, 2400.0, 5000.0)])
        for axis in ("x", "y", "z")
    }
    return ModalVibrationModel(axes, GEOMETRY, n_substeps=4, background_gain=background_gain, seed=seed)


def run_and_get_spectrum(model, n_steps=20000):
    acc = np.array([model.step(omega=OMEGA, dt=TAU)[0] for _ in range(n_steps)])  # axis x
    acc = acc - acc.mean()
    freqs = np.fft.rfftfreq(n_steps, d=TAU)
    spectrum = np.abs(np.fft.rfft(acc))
    return freqs, spectrum


def band_peak(freqs, spectrum, center_hz, half_width_hz=1.5):
    mask = np.abs(freqs - center_hz) <= half_width_hz
    return spectrum[mask].max() if mask.any() else 0.0


class TestSeverityZeroIsPureBackgroundNoise:
    def test_no_peaks_at_any_fault_frequency(self):
        model = build_model(seed=0)
        model.set_fault("outer_race", severity=0.0)  # fault_type set, but severity=0 -> no impulses
        model.reset(seed=0)
        freqs, spectrum = run_and_get_spectrum(model)

        f_r_hz = OMEGA / (2 * np.pi)
        fault_freqs = bf.fault_frequencies_hz(f_r_hz, GEOMETRY)

        # "no peak" = the band around each fault frequency is not distinguishable from the
        # general noise floor (median of the spectrum) beyond a generous multiple.
        noise_floor = np.median(spectrum)
        for name, f0 in fault_freqs.items():
            peak = band_peak(freqs, spectrum, f0)
            assert peak < 6 * noise_floor, f"unexpected peak at {name} ({f0:.1f} Hz) with severity=0"

    def test_healthy_fault_type_none_also_has_no_peaks(self):
        model = build_model(seed=1)
        model.set_fault(None)
        model.reset(seed=1)
        freqs, spectrum = run_and_get_spectrum(model)

        f_r_hz = OMEGA / (2 * np.pi)
        fault_freqs = bf.fault_frequencies_hz(f_r_hz, GEOMETRY)
        noise_floor = np.median(spectrum)
        for name, f0 in fault_freqs.items():
            peak = band_peak(freqs, spectrum, f0)
            assert peak < 6 * noise_floor, f"unexpected peak at {name} ({f0:.1f} Hz) for a healthy bearing"


class TestFaultProducesCorrectPeak:
    @pytest.mark.parametrize("fault_type", ["outer_race", "inner_race", "ball", "cage"])
    def test_band_energy_rises_at_expected_frequency(self, fault_type):
        """Checks a *local* rise in the fault's own band relative to severity=0, not that it's
        the spectrum's global maximum -- the calibrated structural resonances legitimately
        dominate the raw peak amplitude (same reasoning as A.5's closeout test / Sec. A.4-A.5
        report: real bearing-fault diagnostics read a band/envelope, not the raw global peak)."""
        f_r_hz = OMEGA / (2 * np.pi)
        expected_hz = bf.fault_order(fault_type, GEOMETRY) * f_r_hz

        healthy_model = build_model(background_gain=0.02, seed=0)
        healthy_model.set_fault(None)
        healthy_model.reset(seed=0)
        freqs_h, spectrum_h = run_and_get_spectrum(healthy_model)

        faulty_model = build_model(background_gain=0.02, seed=0)
        faulty_model.set_fault(fault_type, severity=5.0)
        faulty_model.reset(seed=0)
        freqs_f, spectrum_f = run_and_get_spectrum(faulty_model)

        healthy_band = band_peak(freqs_h, spectrum_h, expected_hz)
        faulty_band = band_peak(freqs_f, spectrum_f, expected_hz)
        assert faulty_band > 3 * healthy_band

    def test_peak_amplitude_grows_with_severity(self):
        f_r_hz = OMEGA / (2 * np.pi)
        expected_hz = bf.fault_order("outer_race", GEOMETRY) * f_r_hz

        peaks = []
        for severity in (0.0, 2.0, 5.0, 10.0):
            model = build_model(background_gain=0.02, seed=0)
            model.set_fault("outer_race", severity=severity)
            model.reset(seed=0)
            freqs, spectrum = run_and_get_spectrum(model)
            peaks.append(band_peak(freqs, spectrum, expected_hz))

        assert peaks == sorted(peaks)  # monotonically non-decreasing with severity
        assert peaks[-1] > 5 * peaks[0]  # severity=10 clearly above the severity=0 noise floor
