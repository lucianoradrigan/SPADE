"""B4 C2000 visit kits (D-B4-4 option B): fresh kits with fs_hz at the data's 10 kHz (AI-N26); the October kits untouched."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KITS = {"esp32_tiny": ("esp32_tiny/2026-10-19_run01", 256, 25.6), "esp32_b4": ("esp32_dscnn/2026-10-18_run01", 512, 51.2)}


@pytest.mark.parametrize("name", sorted(KITS))
def test_visit_kit_manifest_at_10khz(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location("c2000_visit_kit", REPO / "scripts" / "edge_benchmark" / "c2000_visit_kit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source, samples, budget = KITS[name]
    pre = mod.preflight(REPO / "docs" / "audit" / "b4" / "c2000_kit" / name)
    assert pre["source_run"] == source and pre["input_samples"] == samples
    assert pre["fs_ok"] and pre["c5_budget_ms"] == pytest.approx(budget) and pre["memory"]["fits"]


@pytest.mark.parametrize("name", sorted(KITS))
def test_visit_kit_same_weights_as_october_kit(name):
    source = KITS[name][0]
    block = "regressors" if source.startswith("esp32_tiny") else "classifiers"
    old = json.loads((REPO / "configs" / block / source / "c2000" / "manifest.json").read_text())
    new = json.loads((REPO / "docs" / "audit" / "b4" / "c2000_kit" / name / "manifest.json").read_text())
    assert new["model_id"] == old["model_id"]
    assert "fs_hz" not in old or old.get("fs_hz") is None  # the October kit was not re-exported in place
