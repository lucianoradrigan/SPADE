"""Suite-wide fixtures.

Every test gets its own SPADE workspace (driveflow.runs.store.workspace_root reads
$SPADE_WORKSPACE), so pages that persist runs or jobs never write into the repository's
./workspace and tests never see each other's records."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_spade_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("SPADE_WORKSPACE", str(tmp_path / "spade_workspace"))
