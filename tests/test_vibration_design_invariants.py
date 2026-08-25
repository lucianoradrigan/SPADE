"""Negative tests for design decisions already investigated and refuted (docs/patch2_retiro_modulo_C.md,
docs/patch3_mejora_modulo_B.md) -- these must fail loudly if any of the discarded designs gets
silently reintroduced, not just verify today's happy-path behavior.

Written for Patch 8 Sec. 4. Note on one deviation from the patch's literal wording: it names a
dict ``DEFAULT_VIBRATION_FAULT_AMPLITUDE`` that does not exist anywhere in this codebase. The
closest real dict, ``driveflow.datagen.scenario.CALIBRATED_MECHANICAL_SEVERITY``, is NOT what
``FaultImpulseGenerator`` validates fault_type against -- it's a UI-convenience default consulted
via ``.get(fault_type, 0.0)`` (a silent fallback, by design, for the dashboard's slider default,
never touched by ``FaultImpulseGenerator`` at all). ``FaultImpulseGenerator`` actually validates
fault_type against ``bearing_frequencies.FAULT_FREQUENCY_FUNCS`` (via ``fault_order()``), which
DOES raise explicitly. Sec. 4's third check below tests that real, present behavior instead of the
non-existent dict named in the patch -- see the Patch 8 summary for this discrepancy.
"""

import inspect
from pathlib import Path

import pytest

import driveflow.sim.vibration as vibration_pkg
from driveflow.sim.vibration.background_noise import BackgroundNoiseGenerator
from driveflow.sim.vibration.bearing_frequencies import KAT_DATACENTER_6203_GEOMETRY
from driveflow.sim.vibration.fault_impulses import FaultImpulseGenerator


def _param_names(func) -> set:
    return set(inspect.signature(func).parameters) - {"self"}


class TestNoTorqueCoupling:
    """docs/patch3_mejora_modulo_B.md: torque was investigated as a Module B excitation input and
    found to carry no fault-type-specific signal (coherence at the wrong fault frequency is as
    high or higher than at the bearing's own) -- it was removed from the excitation path entirely,
    not just left unused. If it (or an equivalent renamed parameter, e.g. "torque_nm") reappears in
    either generator's signature, that reintroduces a design already investigated and discarded."""

    def _assert_no_torque_param(self, func, owner_name: str):
        names = _param_names(func)
        offending = {n for n in names if "torque" in n.lower()}
        assert not offending, (
            f"{owner_name} accepts torque-like parameter(s) {offending} -- Patch 3 "
            "(docs/patch3_mejora_modulo_B.md) found torque carries no fault-type-specific "
            "coupling to vibration and removed it from Module B's excitation entirely. "
            "Reintroducing a torque input here silently un-does that finding -- see the patch "
            "before adding this back."
        )

    def test_fault_impulse_generator_init_has_no_torque_param(self):
        self._assert_no_torque_param(FaultImpulseGenerator.__init__, "FaultImpulseGenerator.__init__")

    def test_fault_impulse_generator_step_has_no_torque_param(self):
        self._assert_no_torque_param(FaultImpulseGenerator.step, "FaultImpulseGenerator.step")

    def test_background_noise_generator_init_has_no_torque_param(self):
        self._assert_no_torque_param(BackgroundNoiseGenerator.__init__, "BackgroundNoiseGenerator.__init__")

    def test_background_noise_generator_step_has_no_torque_param(self):
        self._assert_no_torque_param(BackgroundNoiseGenerator.step, "BackgroundNoiseGenerator.step")


class TestModuleCStaysRetired:
    """docs/patch2_retiro_modulo_C.md: the data-driven residual corrector (Module C) was
    investigated and retired -- no exploitable torque/speed/force -> vibration signal exists in
    this dataset to condition it on. Checks for absence of the FILE, not just absence of an
    import: a re-added file with dead imports would still represent silent reintroduction of
    retired scope, and grep-for-import alone would miss that."""

    def test_no_residual_model_file(self):
        vibration_dir = Path(vibration_pkg.__file__).parent
        residual_model_path = vibration_dir / "residual_model.py"
        assert not residual_model_path.exists(), (
            f"{residual_model_path} exists -- Module C (the data-driven residual corrector) was "
            "retired in docs/patch2_retiro_modulo_C.md after finding no exploitable signal to "
            "condition it on (torque/speed/force coherence with real vibration all near zero, "
            "see docs/patch2_retiro_modulo_C.md Sec. 1). Re-adding this file reintroduces "
            "discarded scope -- see the patch before restoring it."
        )


class TestUnknownFaultTypeRaisesExplicitly:
    """FaultImpulseGenerator must fail loudly on an unrecognized fault_type, not silently fall
    back to some default amplitude/order -- a silent default would mask a typo'd or newly-added
    fault_type never wired into bearing_frequencies.FAULT_FREQUENCY_FUNCS, producing a
    plausible-looking but physically meaningless impulse train instead of an error."""

    def test_unknown_fault_type_raises(self):
        with pytest.raises(ValueError, match="[Uu]nknown fault_type"):
            FaultImpulseGenerator(KAT_DATACENTER_6203_GEOMETRY, fault_type="not_a_real_fault_type", severity=1.0)

    def test_known_fault_types_do_not_raise(self):
        """Sanity check on the test above: a real fault_type must NOT raise, so the check is
        actually discriminating unknown-vs-known, not just always raising."""
        for fault_type in ("outer_race", "inner_race", "ball", "cage"):
            FaultImpulseGenerator(KAT_DATACENTER_6203_GEOMETRY, fault_type=fault_type, severity=1.0)
