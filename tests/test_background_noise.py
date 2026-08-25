"""Tests for sim/vibration/background_noise.py -- Module B's always-on stochastic excitation,
deliberately independent of torque/speed/load (see docs/patch3_mejora_modulo_B.md)."""

import numpy as np
import pytest

from driveflow.sim.vibration.background_noise import BackgroundNoiseGenerator


class TestBackgroundNoiseGenerator:
    def test_zero_gain_is_silent(self):
        gen = BackgroundNoiseGenerator(gain=0.0, seed=0)
        assert np.all(gen.step(1000) == 0.0)

    def test_reproducible_with_seed(self):
        gen_a = BackgroundNoiseGenerator(gain=1.0, seed=7)
        gen_b = BackgroundNoiseGenerator(gain=1.0, seed=7)
        np.testing.assert_array_equal(gen_a.step(500), gen_b.step(500))

    def test_different_seeds_differ(self):
        gen_a = BackgroundNoiseGenerator(gain=1.0, seed=1)
        gen_b = BackgroundNoiseGenerator(gain=1.0, seed=2)
        assert not np.array_equal(gen_a.step(500), gen_b.step(500))

    def test_gain_scales_rms(self):
        gen1 = BackgroundNoiseGenerator(gain=1.0, seed=0)
        gen3 = BackgroundNoiseGenerator(gain=3.0, seed=0)
        rms1 = np.sqrt(np.mean(gen1.step(200000) ** 2))
        rms3 = np.sqrt(np.mean(gen3.step(200000) ** 2))
        assert rms3 == pytest.approx(3 * rms1, rel=0.02)

    def test_is_zero_mean(self):
        gen = BackgroundNoiseGenerator(gain=1.0, seed=0)
        samples = gen.step(200000)
        assert samples.mean() == pytest.approx(0.0, abs=0.02)

    def test_reset_with_seed_restarts_sequence(self):
        gen = BackgroundNoiseGenerator(gain=1.0, seed=5)
        first_run = gen.step(100)
        gen.reset(seed=5)
        second_run = gen.step(100)
        np.testing.assert_array_equal(first_run, second_run)

    def test_reset_without_seed_gives_a_fresh_sequence(self):
        gen = BackgroundNoiseGenerator(gain=1.0, seed=5)
        first_run = gen.step(100)
        gen.reset()
        second_run = gen.step(100)
        assert not np.array_equal(first_run, second_run)
