"""AI-N28 repair round (docs/audit/assistant/esp32_repair_preregistration.md): the new options keep today's
defaults -- export_int8 stays per channel, and the held-out generator still builds the original E2a."""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_export_int8_default_is_per_channel():
    from driveflow.ai.tflite_export import export_int8

    assert inspect.signature(export_int8).parameters["per_channel"].default is True


def test_heldout_generator_defaults_reproduce_the_original_e2a():
    import generate_ai_n27_heldout as H

    a, b = H.build_jobs("3class"), H.build_jobs("3class", H.RNG_SEED, H.FIRST_RUN_SEED)
    assert a == b and a[0]["seed"] == 60001 and len(a) == 45
    new = H.build_jobs("3class", 20261108, 70001)
    assert [j["seed"] for j in new] == list(range(70001, 70046))
    assert {(j["speed"], j["severity_scale"], j["fault"]) for j in new} == {(j["speed"], j["severity_scale"], j["fault"]) for j in a}
    assert [j["electrical_noise_pct"] for j in new] != [j["electrical_noise_pct"] for j in a]  # new draws
