"""Tests for sim/vibration/calibration.py's fitting machinery.

These do NOT touch the real KAt-DataCenter dataset (it is a local, CC BY-NC download outside the
repo, see docs/addendum_vibracion_v1.md Sec. 9) -- they validate fit_modes_to_psd/psd_log_rmse
against a synthetic PSD generated from known Mode parameters, so the fit can be checked for
correctness (recovers the true modes) independently of how well any particular real recording
happens to calibrate.
"""

import numpy as np
import pytest

from driveflow.sim.vibration import calibration as cal
from driveflow.sim.vibration.modal_model import Mode


def synthetic_psd(freqs_hz, true_modes, noise_std=0.0, seed=0):
    params = cal._pack_params(true_modes)
    psd = cal._modal_bank_psd(freqs_hz, params, len(true_modes))
    if noise_std:
        rng = np.random.default_rng(seed)
        psd = psd * np.exp(rng.normal(0, noise_std, size=psd.shape))
    return psd


class TestFilenameParsing:
    def test_parses_standard_paderborn_filename(self):
        meta = cal.parse_filename("/some/path/N15_M07_F10_KA01_3.mat")
        assert meta == {
            "speed_code": "15",
            "load_code": "07",
            "force_code": "10",
            "bearing_code": "KA01",
            "run": "3",
        }

    def test_rejects_unexpected_filename(self):
        with pytest.raises(ValueError):
            cal.parse_filename("/some/path/not_a_paderborn_file.mat")


class TestModalBankPSD:
    def test_recovers_known_single_mode(self):
        true_modes = [Mode(natural_freq_hz=1000.0, damping_ratio=0.03, gain=0.001)]
        freqs = np.linspace(100, 5000, 2000)
        psd = synthetic_psd(freqs, true_modes)

        fitted = cal.fit_modes_to_psd(freqs, psd, n_modes=1, freq_bounds=(100, 5000), damping_bounds=(1e-3, 0.3))

        assert fitted[0].natural_freq_hz == pytest.approx(1000.0, rel=0.02)
        assert fitted[0].damping_ratio == pytest.approx(0.03, rel=0.1)
        assert fitted[0].gain == pytest.approx(0.001, rel=0.1)

    def test_recovers_known_modes_with_mild_noise(self):
        true_modes = [
            Mode(natural_freq_hz=800.0, damping_ratio=0.02, gain=0.002),
            Mode(natural_freq_hz=3000.0, damping_ratio=0.05, gain=0.0005),
        ]
        freqs = np.linspace(100, 8000, 4000)
        psd = synthetic_psd(freqs, true_modes, noise_std=0.05, seed=42)

        fitted = cal.fit_modes_to_psd(freqs, psd, n_modes=2, freq_bounds=(100, 8000), damping_bounds=(1e-3, 0.3))
        fitted_freqs = sorted(m.natural_freq_hz for m in fitted)
        true_freqs = sorted(m.natural_freq_hz for m in true_modes)
        for f_hat, f_true in zip(fitted_freqs, true_freqs):
            assert f_hat == pytest.approx(f_true, rel=0.05)

    def test_perfect_fit_has_near_zero_rmse(self):
        true_modes = [Mode(natural_freq_hz=1500.0, damping_ratio=0.02, gain=0.001)]
        freqs = np.linspace(100, 5000, 1000)
        psd = synthetic_psd(freqs, true_modes)
        rmse = cal.psd_log_rmse(freqs, psd, true_modes, freq_bounds=(100, 5000))
        assert rmse < 1e-6

    def test_wrong_modes_have_nonzero_rmse(self):
        true_modes = [Mode(natural_freq_hz=1500.0, damping_ratio=0.02, gain=0.001)]
        wrong_modes = [Mode(natural_freq_hz=2500.0, damping_ratio=0.1, gain=0.0003)]
        freqs = np.linspace(100, 5000, 1000)
        psd = synthetic_psd(freqs, true_modes)
        rmse = cal.psd_log_rmse(freqs, psd, wrong_modes, freq_bounds=(100, 5000))
        assert rmse > 0.5


class TestComputePsd:
    def test_welch_psd_peaks_near_injected_tone(self):
        fs = 20000.0
        t = np.arange(0, 2.0, 1 / fs)
        tone_hz = 1234.0
        signal = np.sin(2 * np.pi * tone_hz * t)
        freqs, psd = cal.compute_psd(signal, fs=fs, nperseg=4096)
        peak_freq = freqs[np.argmax(psd)]
        assert peak_freq == pytest.approx(tone_hz, abs=fs / 4096)
