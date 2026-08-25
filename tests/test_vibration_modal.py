"""Unit tests for the physics-only pieces of the vibration module (A.3/Patch 3):
bearing_frequencies, modal_model. None of these need the Paderborn dataset -- that is exercised
separately by calibration.py once the dataset is extracted.

fault_impulses.py and background_noise.py (the two excitation sources ModalVibrationModel
combines, per docs/patch3_mejora_modulo_B.md) have their own test files.
"""

import numpy as np
import pytest

from driveflow.sim.vibration import bearing_frequencies as bf
from driveflow.sim.vibration.modal_model import Mode, ModalAxis, ModalVibrationModel

GEOMETRY = bf.BearingGeometry(n_elements=9, element_diameter_m=7.5e-3, pitch_diameter_m=33.5e-3)


class TestBearingFrequencies:
    def test_shaft_frequency_from_omega(self):
        assert bf.shaft_frequency_hz(2 * np.pi) == pytest.approx(1.0)

    def test_bpfi_greater_than_bpfo_for_zero_contact_angle(self):
        # BPFI = (n/2) f_r (1 + ratio), BPFO = (n/2) f_r (1 - ratio); ratio in (0, 1) => BPFI > BPFO
        f_r = 25.0
        assert bf.bpfi(f_r, GEOMETRY) > bf.bpfo(f_r, GEOMETRY)

    def test_bpfo_plus_bpfi_equals_n_times_f_r(self):
        # (n/2)(1-r) + (n/2)(1+r) = n, independent of the ratio/contact angle.
        f_r = 25.0
        assert bf.bpfo(f_r, GEOMETRY) + bf.bpfi(f_r, GEOMETRY) == pytest.approx(GEOMETRY.n_elements * f_r)

    def test_frequencies_scale_linearly_with_shaft_speed(self):
        freqs_1x = bf.fault_frequencies_hz(10.0, GEOMETRY)
        freqs_2x = bf.fault_frequencies_hz(20.0, GEOMETRY)
        for name in freqs_1x:
            assert freqs_2x[name] == pytest.approx(2 * freqs_1x[name])

    def test_fault_order_matches_frequency_at_unit_shaft_speed(self):
        for fault_type, func in bf.FAULT_FREQUENCY_FUNCS.items():
            assert bf.fault_order(fault_type, GEOMETRY) == pytest.approx(func(1.0, GEOMETRY))

    def test_unknown_fault_type_raises(self):
        with pytest.raises(ValueError):
            bf.fault_order("not_a_fault", GEOMETRY)

    def test_rejects_invalid_geometry(self):
        with pytest.raises(ValueError):
            bf.BearingGeometry(n_elements=9, element_diameter_m=40e-3, pitch_diameter_m=33.5e-3)


class TestModalAxis:
    def test_free_decay_matches_theoretical_envelope(self):
        """With F=0 and a nonzero initial state, a single mode's displacement envelope must
        decay as exp(-zeta*omega_n*t) (standard underdamped free-vibration response)."""
        natural_freq_hz, zeta = 200.0, 0.02
        axis = ModalAxis([Mode(natural_freq_hz=natural_freq_hz, damping_ratio=zeta, gain=1.0)])
        axis._state[0] = [1.0, 0.0]  # unit initial displacement, zero velocity

        wn = 2 * np.pi * natural_freq_hz
        # Exact (matrix-exponential) discretization has no step-size stability constraint, so
        # dt_sub is chosen only to resolve the ~5 ms period well; n_steps covers ~5 time
        # constants (1/(zeta*wn) ~= 40 ms here) so real decay is observable.
        dt_sub = 1e-4
        n_steps = 2000
        zero_force = np.zeros(2)

        displacements = [1.0]
        for _ in range(n_steps):
            axis.step(zero_force, dt_sub)
            displacements.append(axis._state[0, 0])
        displacements = np.array(displacements)

        t = np.arange(n_steps + 1) * dt_sub
        # Exact envelope for x(0)=1, v(0)=0: x0 * exp(-zeta*wn*t) / sqrt(1 - zeta**2).
        expected_envelope = np.exp(-zeta * wn * t) / np.sqrt(1 - zeta**2)
        # The oscillation is bounded by its envelope; check it decays to near zero and never
        # exceeds the theoretical envelope (up to floating point slack).
        assert np.all(np.abs(displacements) <= expected_envelope + 1e-9)
        assert abs(displacements[-1]) < 0.05 * abs(displacements[0])

    def test_zero_modes_rejected(self):
        with pytest.raises(ValueError):
            ModalAxis([])

    def test_mode_rejects_invalid_damping_ratio(self):
        with pytest.raises(ValueError):
            Mode(natural_freq_hz=100.0, damping_ratio=1.5)


class TestModalVibrationModel:
    def _build_model(self, background_gain=0.5, seed=0):
        axes = {
            axis: ModalAxis([Mode(natural_freq_hz=freq, damping_ratio=0.02, gain=1.0) for freq in (800.0, 2400.0)])
            for axis in ("x", "y", "z")
        }
        return ModalVibrationModel(axes, GEOMETRY, n_substeps=20, background_gain=background_gain, seed=seed)

    def test_step_returns_three_axes(self):
        model = self._build_model()
        acc = model.step(omega=100.0, dt=1e-4)
        assert acc.shape == (3,)
        assert np.all(np.isfinite(acc))

    def test_healthy_bearing_has_no_fault_generator(self):
        model = self._build_model()
        model.set_fault(None)
        assert model.fault_gen._generator is None

    def test_set_fault_configures_impulse_generator_with_correct_order(self):
        model = self._build_model()
        model.set_fault("outer_race", severity=5.0)
        expected_order = bf.fault_order("outer_race", GEOMETRY)
        assert model.fault_gen._generator.order == pytest.approx(expected_order)

    def test_reset_clears_state(self):
        model = self._build_model()
        model.set_fault("inner_race", severity=3.0)
        for _ in range(10):
            model.step(omega=150.0, dt=1e-4)
        model.reset(seed=0)
        for axis in model.axes.values():
            assert np.all(axis._state == 0.0)
        assert model._prev_omega == 0.0

    def test_fault_injection_increases_response_energy(self):
        """A faulted bearing (extra impulsive excitation) should produce, on average, larger
        acceleration magnitude than a healthy one under the same operating point and the same
        background-noise seed (so the only difference is the fault impulses)."""
        healthy = self._build_model(background_gain=0.0, seed=1)  # background_gain=0 isolates the fault's effect
        faulty = self._build_model(background_gain=0.0, seed=1)
        faulty.set_fault("outer_race", severity=50.0)

        n_steps = 200
        healthy_energy = 0.0
        faulty_energy = 0.0
        for _ in range(n_steps):
            healthy_energy += np.sum(healthy.step(omega=120.0, dt=1e-4) ** 2)
            faulty_energy += np.sum(faulty.step(omega=120.0, dt=1e-4) ** 2)

        assert faulty_energy > healthy_energy
        assert healthy_energy == pytest.approx(0.0)  # no background, no fault -> silence

    def test_background_noise_alone_is_reproducible_with_seed(self):
        model_a = self._build_model(background_gain=1.0, seed=42)
        model_b = self._build_model(background_gain=1.0, seed=42)
        for _ in range(20):
            np.testing.assert_array_equal(model_a.step(omega=100.0, dt=1e-4), model_b.step(omega=100.0, dt=1e-4))

    def test_rejects_incomplete_axes(self):
        with pytest.raises(ValueError):
            ModalVibrationModel({"x": ModalAxis([Mode(100.0, 0.02)])}, GEOMETRY)
