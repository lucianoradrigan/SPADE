"""Tests for the VibrationSynthesizer orchestration (A.3/Patch 3), including an end-to-end wiring
test against a real SCMLSystem, matching the usage sketched in docs/addendum_vibracion_v1.md
Sec. 4 (the actual datagen/runner.py loop is Macro-fase A.4).

Module C was retired and its code removed -- see docs/patch2_retiro_modulo_C.md.
"""

import numpy as np

from driveflow.sim import (
    ContOneQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PolynomialStaticLoad,
)
from driveflow.sim.vibration import BearingGeometry, Mode, ModalAxis, ModalVibrationModel, VibrationSynthesizer

GEOMETRY = BearingGeometry(n_elements=9, element_diameter_m=7.5e-3, pitch_diameter_m=33.5e-3)
TAU = 1e-4


def build_module_b(n_substeps=10, background_gain=0.5, seed=0):
    axes = {
        axis: ModalAxis([Mode(natural_freq_hz=freq, damping_ratio=0.02) for freq in (800.0, 2400.0)])
        for axis in ("x", "y", "z")
    }
    return ModalVibrationModel(axes, GEOMETRY, n_substeps=n_substeps, background_gain=background_gain, seed=seed)


def build_scml_system():
    return DcMotorSystem(
        converter=ContOneQuadrantConverter(),
        motor=DcPermanentlyExcitedMotor(),
        load=PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025)),
        supply=IdealVoltageSupply(u_nominal=560.0),
        ode_solver=EulerSolver(),
        tau=TAU,
    )


class TestVibrationSynthesizer:
    def test_vibration_source_is_always_synthetic_b(self):
        """Module C retired: vibration_source has no "synthetic_b_plus_c" alternative anymore."""
        synth = VibrationSynthesizer(build_module_b(), tau=TAU)
        assert synth.vibration_source == "synthetic_b"

    def test_step_matches_underlying_modal_model_with_same_seed(self):
        module_b_a = build_module_b(seed=3)
        module_b_b = build_module_b(seed=3)
        synth = VibrationSynthesizer(module_b_a, tau=TAU)

        for _ in range(5):
            vib_synth = synth.step(omega=100.0)
            vib_b_only = module_b_b.step(omega=100.0, dt=TAU)
            np.testing.assert_allclose(vib_synth, vib_b_only)

    def test_set_fault_delegates_to_modal_model(self):
        synth = VibrationSynthesizer(build_module_b(), tau=TAU)
        synth.set_fault("ball", severity=10.0)
        assert synth.modal_model.fault_gen.fault_type == "ball"
        assert synth.modal_model.fault_gen.severity == 10.0

    def test_reset_reseeds_background_noise(self):
        synth = VibrationSynthesizer(build_module_b(background_gain=1.0, seed=9), tau=TAU)
        synth.reset(seed=9)
        first = [synth.step(omega=100.0) for _ in range(10)]
        synth.reset(seed=9)
        second = [synth.step(omega=100.0) for _ in range(10)]
        np.testing.assert_allclose(first, second)

    def test_end_to_end_with_real_scml_system(self):
        """Mirrors the addendum's Sec. 4 runner fragment, updated per Patch 3 (no i_d/i_q/torque
        into vibration_synth.step()): state = scml_system.simulate(action); omega =
        extract_omega(...); acc = vibration_synth.step(omega)."""
        system = build_scml_system()
        synth = VibrationSynthesizer(build_module_b(), tau=TAU)

        state = system.reset()
        synth.reset(seed=0)
        synth.set_fault("outer_race", severity=20.0)

        omega_idx = system.OMEGA_IDX
        limits = system.limits

        accelerations = []
        for k in range(20):
            action = np.array([0.5])
            state = system.simulate(action)
            omega = (state * limits)[omega_idx]
            acc_x, acc_y, acc_z = synth.step(omega)
            accelerations.append((acc_x, acc_y, acc_z))

        accelerations = np.array(accelerations)
        assert accelerations.shape == (20, 3)
        assert np.all(np.isfinite(accelerations))
