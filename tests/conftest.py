"""Suite-wide fixtures.

Every test gets its own SPADE workspace (driveflow.runs.store.workspace_root reads
$SPADE_WORKSPACE), so pages that persist runs or jobs never write into the repository's
./workspace and tests never see each other's records."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_spade_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_WORKSPACE", str(tmp_path / "spade_workspace"))


def pytest_addoption(parser):
    parser.addoption("--run-llm", action="store_true", default=False,
                     help="run the @pytest.mark.llm tests against the local Ollama (assistant, phase 1)")


def pytest_collection_modifyitems(config, items):
    """@pytest.mark.llm tests need a real local model runtime: skipped unless --run-llm (B4)."""
    if config.getoption("--run-llm"):
        return
    skip = pytest.mark.skip(reason="needs the local model runtime; run with --run-llm")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
