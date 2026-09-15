"""Phase 1 scaffolding smoke tests for ModelLoader -- Phase 2 replaces the NotImplementedError
assertions with real behavior tests once loader.py is actually implemented."""

import pytest

from driveflow.ai.transfer.loader import ModelLoader


def test_loader_initialization():
    loader = ModelLoader(registry_path="configs/registry.yaml", domain="dc_motor", block="classifier", tier="pc")
    assert loader.domain == "dc_motor"
    assert loader.block == "classifier"
    assert loader.tier == "pc"


def test_loader_missing_implementation():
    loader = ModelLoader(registry_path="configs/registry.yaml", domain="dc_motor", block="classifier")
    with pytest.raises(NotImplementedError):
        loader.load_latest()
