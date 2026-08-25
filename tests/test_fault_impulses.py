"""Tests for sim/vibration/fault_impulses.py -- the order-tracked impulse train shared by Module
B's mechanical path (FaultImpulseGenerator) and the electrical/MCSA path
(datagen/fault_injection.py's BearingFaultLoad, tested separately, uses ImpulseTrainGenerator
directly)."""

import numpy as np
import pytest

from driveflow.sim.vibration import bearing_frequencies as bf
from driveflow.sim.vibration.fault_impulses import CompositeFaultImpulseGenerator, FaultImpulseGenerator, ImpulseTrainGenerator, LoadZoneModulator

GEOMETRY = bf.KAT_DATACENTER_6203_GEOMETRY


class TestImpulseTrainGenerator:
    def test_impulse_count_matches_expected_rate_at_constant_speed(self):
        order = 3.5  # e.g. a BPFO-like order
        f_r = 20.0  # Hz, constant shaft speed
        omega = 2 * np.pi * f_r
        dt_sub = 1e-5
        duration_s = 1.0
        n_samples = int(duration_s / dt_sub)

        generator = ImpulseTrainGenerator(order)
        omega_samples = np.full(n_samples, omega)
        pulses = generator.generate(omega_samples, dt_sub)

        n_impulses = np.count_nonzero(pulses)
        expected = order * f_r * duration_s
        assert n_impulses == pytest.approx(expected, abs=1)

    def test_impulse_amplitude_is_unit_area(self):
        generator = ImpulseTrainGenerator(order=1.0)
        dt_sub = 1e-4
        omega_samples = np.full(20000, 2 * np.pi * 10.0)  # 10 Hz shaft speed, order 1 -> ~10 pulses/s
        pulses = generator.generate(omega_samples, dt_sub)
        nonzero = pulses[pulses > 0]
        assert np.allclose(nonzero, 1.0 / dt_sub)

    def test_state_persists_across_calls(self):
        # Splitting one long window into two consecutive calls must give the same impulse count
        # as a single call over the concatenated window (continuity of the phase accumulator).
        order, f_r, dt_sub = 2.0, 15.0, 1e-5
        omega = np.full(20000, 2 * np.pi * f_r)

        gen_whole = ImpulseTrainGenerator(order)
        pulses_whole = gen_whole.generate(omega, dt_sub)

        gen_split = ImpulseTrainGenerator(order)
        first = gen_split.generate(omega[:10000], dt_sub)
        second = gen_split.generate(omega[10000:], dt_sub)
        pulses_split = np.concatenate([first, second])

        assert np.count_nonzero(pulses_whole) == np.count_nonzero(pulses_split)

    def test_rejects_nonpositive_order(self):
        with pytest.raises(ValueError):
            ImpulseTrainGenerator(order=0.0)


class TestFaultImpulseGenerator:
    def test_healthy_returns_zeros(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type=None, severity=0.0)
        omega_samples = np.full(21, 2 * np.pi * 25.0)
        out = gen.step(omega_samples, dt_sub=1e-5)
        assert np.all(out == 0.0)

    def test_zero_severity_returns_zeros_even_with_fault_type(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="outer_race", severity=0.0)
        omega_samples = np.full(21, 2 * np.pi * 25.0)
        out = gen.step(omega_samples, dt_sub=1e-5)
        assert np.all(out == 0.0)

    def test_order_matches_bearing_frequencies(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="inner_race", severity=1.0)
        expected_order = bf.fault_order("inner_race", GEOMETRY)
        assert gen._generator.order == pytest.approx(expected_order)

    def test_severity_scales_impulse_amplitude(self):
        omega_samples = np.full(2001, 2 * np.pi * 25.0)
        dt_sub = 1e-5
        gen1 = FaultImpulseGenerator(GEOMETRY, fault_type="ball", severity=1.0)
        gen2 = FaultImpulseGenerator(GEOMETRY, fault_type="ball", severity=3.0)
        out1 = gen1.step(omega_samples, dt_sub)
        out2 = gen2.step(omega_samples, dt_sub)
        assert np.max(out2) == pytest.approx(3.0 * np.max(out1))

    def test_reset_clears_phase(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="outer_race", severity=1.0)
        gen.step(np.full(1000, 2 * np.pi * 25.0), dt_sub=1e-5)
        assert gen._generator._phase != 0.0
        gen.reset()
        assert gen._generator._phase == 0.0

    def test_load_zone_modulation_can_be_disabled(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="inner_race", severity=1.0, load_zone_modulation=False)
        assert gen._modulator is None


class TestFaultImpulseGeneratorOrderOverride:
    """order_override is how dashboard.py's "Custom fault types" feature injects a user-defined
    fault (given only a characteristic order, no bearing geometry) -- see this class's docstring
    in fault_impulses.py."""

    def test_uses_the_override_order_not_a_geometry_lookup(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="gear_mesh", severity=1.0, order_override=7.25)
        assert gen._generator.order == pytest.approx(7.25)

    def test_accepts_a_fault_type_name_not_in_the_builtin_registry(self):
        """"gear_mesh" is not a bearing_frequencies.FAULT_FREQUENCY_FUNCS key -- without
        order_override this would raise ValueError (see test_order_matches_bearing_frequencies'
        counterpart for a real key); with it, construction must succeed."""
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="gear_mesh", severity=1.0, order_override=7.25)
        omega_samples = np.full(2001, 2 * np.pi * 25.0)
        out = gen.step(omega_samples, dt_sub=1e-5)
        assert np.any(out != 0.0)

    def test_load_zone_modulation_is_never_applied(self):
        """Even with load_zone_modulation=True (the default), a custom order gets no modulator --
        there's no defect-position physics for an arbitrary user-defined fault (see docstring)."""
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="gear_mesh", severity=1.0, order_override=7.25, load_zone_modulation=True)
        assert gen._modulator is None

    def test_impulse_count_matches_the_override_order(self):
        f_r = 20.0  # Hz
        omega_samples = np.full(int(1.0 / 1e-5), 2 * np.pi * f_r)
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="gear_mesh", severity=1.0, order_override=5.0)
        out = gen.step(omega_samples, dt_sub=1e-5)
        assert np.count_nonzero(out) == pytest.approx(5.0 * f_r * 1.0, abs=1)


class TestCompositeFaultImpulseGenerator:
    """How the mechanical/vibration path injects more than one simultaneous fault (dashboard's
    "Combine with additional fault types") -- see the class's own docstring."""

    def test_output_equals_sum_of_each_generator_alone(self):
        omega_samples = np.full(2001, 2 * np.pi * 25.0)
        dt_sub = 1e-5
        gen_a = FaultImpulseGenerator(GEOMETRY, fault_type="outer_race", severity=1.0)
        gen_b = FaultImpulseGenerator(GEOMETRY, fault_type="gear_mesh", severity=2.0, order_override=6.5)
        out_a = gen_a.step(omega_samples.copy(), dt_sub)
        out_b = gen_b.step(omega_samples.copy(), dt_sub)

        gen_a2 = FaultImpulseGenerator(GEOMETRY, fault_type="outer_race", severity=1.0)
        gen_b2 = FaultImpulseGenerator(GEOMETRY, fault_type="gear_mesh", severity=2.0, order_override=6.5)
        composite = CompositeFaultImpulseGenerator([gen_a2, gen_b2])
        out_composite = composite.step(omega_samples.copy(), dt_sub)

        assert np.allclose(out_composite, out_a + out_b)

    def test_reset_resets_every_sub_generator(self):
        gen_a = FaultImpulseGenerator(GEOMETRY, fault_type="outer_race", severity=1.0)
        gen_b = FaultImpulseGenerator(GEOMETRY, fault_type="inner_race", severity=1.0)
        composite = CompositeFaultImpulseGenerator([gen_a, gen_b])
        composite.step(np.full(1000, 2 * np.pi * 25.0), dt_sub=1e-5)
        assert gen_a._generator._phase != 0.0
        assert gen_b._generator._phase != 0.0
        composite.reset()
        assert gen_a._generator._phase == 0.0
        assert gen_b._generator._phase == 0.0

    def test_healthy_sub_generator_contributes_nothing(self):
        """A fault_type=None sub-generator (e.g. the "primary" slot when only extras are active,
        dashboard's "healthy" + additional fault types) must contribute exactly zero, not error."""
        omega_samples = np.full(2001, 2 * np.pi * 25.0)
        gen_healthy = FaultImpulseGenerator(GEOMETRY, fault_type=None, severity=0.0)
        gen_fault = FaultImpulseGenerator(GEOMETRY, fault_type="ball", severity=1.0)
        composite = CompositeFaultImpulseGenerator([gen_healthy, gen_fault])
        out_composite = composite.step(omega_samples, dt_sub=1e-5)
        out_fault_alone = FaultImpulseGenerator(GEOMETRY, fault_type="ball", severity=1.0).step(omega_samples, dt_sub=1e-5)
        assert np.allclose(out_composite, out_fault_alone)


class TestLoadZoneModulator:
    def test_outer_race_is_unmodulated(self):
        mod = LoadZoneModulator("outer_race", GEOMETRY)
        omega_samples = np.full(5000, 2 * np.pi * 25.0)
        env = mod.envelope(omega_samples, dt_sub=1e-4)
        assert np.all(env == 1.0)

    def test_cage_is_unmodulated(self):
        mod = LoadZoneModulator("cage", GEOMETRY)
        omega_samples = np.full(5000, 2 * np.pi * 25.0)
        env = mod.envelope(omega_samples, dt_sub=1e-4)
        assert np.all(env == 1.0)

    def test_inner_race_modulates_at_shaft_frequency(self):
        f_r = 25.0
        dt_sub = 1e-5
        n = int(round(1.0 / dt_sub))  # 1 second -> f_r cycles of modulation
        mod = LoadZoneModulator("inner_race", GEOMETRY)
        omega_samples = np.full(n, 2 * np.pi * f_r)
        env = mod.envelope(omega_samples, dt_sub)

        assert env.min() == pytest.approx(0.0, abs=1e-6)
        assert env.max() == pytest.approx(1.0, abs=1e-6)

        # count envelope peaks (load-zone crossings) and compare to the expected f_r*duration
        peak_mask = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > 0.9)
        n_peaks = np.count_nonzero(peak_mask)
        assert n_peaks == pytest.approx(f_r, abs=1)

    def test_ball_modulates_at_cage_frequency(self):
        f_r = 25.0
        dt_sub = 1e-5
        n = int(round(1.0 / dt_sub))
        mod = LoadZoneModulator("ball", GEOMETRY)
        omega_samples = np.full(n, 2 * np.pi * f_r)
        env = mod.envelope(omega_samples, dt_sub)

        expected_ftf_hz = bf.ftf(f_r, GEOMETRY)
        peak_mask = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > 0.9)
        n_peaks = np.count_nonzero(peak_mask)
        assert n_peaks == pytest.approx(expected_ftf_hz, abs=1)

    def test_sharpness_narrows_the_loaded_arc(self):
        omega_samples = np.full(4000, 2 * np.pi * 25.0)
        dt_sub = 1e-5
        narrow = LoadZoneModulator("inner_race", GEOMETRY, sharpness=8.0)
        wide = LoadZoneModulator("inner_race", GEOMETRY, sharpness=1.0)
        env_narrow = narrow.envelope(omega_samples, dt_sub)
        env_wide = wide.envelope(omega_samples, dt_sub)
        # a higher sharpness spends less time near full amplitude -> lower mean envelope
        assert env_narrow.mean() < env_wide.mean()

    def test_reset_restarts_modulation_phase(self):
        mod = LoadZoneModulator("inner_race", GEOMETRY)
        omega_samples = np.full(1000, 2 * np.pi * 25.0)
        mod.envelope(omega_samples, dt_sub=1e-5)
        assert mod._phase != 0.0
        mod.reset()
        assert mod._phase == 0.0


class TestFaultImpulseGeneratorWithModulation:
    def test_inner_race_amplitude_varies_over_time(self):
        """With load-zone modulation on, not every fired impulse has the same peak amplitude
        (unlike the pre-Patch-4 model, where every impulse had identical unit-area amplitude)."""
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="inner_race", severity=1.0)
        omega_samples = np.full(20000, 2 * np.pi * 25.0)
        pulses = gen.step(omega_samples, dt_sub=1e-5)
        nonzero = pulses[pulses > 0]
        assert len(nonzero) > 5  # sanity: several impulses fired in this window
        assert nonzero.std() > 0  # amplitudes are NOT all identical

    def test_outer_race_amplitude_stays_uniform(self):
        gen = FaultImpulseGenerator(GEOMETRY, fault_type="outer_race", severity=1.0)
        omega_samples = np.full(20000, 2 * np.pi * 25.0)
        pulses = gen.step(omega_samples, dt_sub=1e-5)
        nonzero = pulses[pulses > 0]
        assert len(nonzero) > 5
        assert nonzero.std() == pytest.approx(0.0, abs=1e-9)  # all impulses identical amplitude
