"""Installation smoke checks (scripts/smoke_test.py) run under pytest -- the job the CI matrix
runs on ubuntu-latest, windows-latest and macos-latest (.github/workflows/tests.yml)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.compat

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("spade_smoke_test", ROOT / "scripts" / "smoke_test.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.RESULTS.clear()
    return mod


@pytest.mark.parametrize("name", ["check_imports", "check_modules", "check_simulation", "check_inference", "check_app"])
def test_smoke_check_passes(smoke, name):
    before = len(smoke.RESULTS)
    getattr(smoke, name)()
    label, ok, detail, _ = smoke.RESULTS[before]
    assert ok, f"{label}: {detail}"


def test_repository_paths_are_platform_neutral():
    """No hard-coded POSIX-only absolute path or backslash path in the package (Windows/macOS)."""
    offenders = []
    for py in (ROOT / "src" / "driveflow").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for needle in ('"/home/', "'/home/", '"/tmp/', "'/tmp/", '"C:\\\\', "'C:\\\\"):
            if needle in text:
                offenders.append(f"{py.relative_to(ROOT)}: {needle}")
    assert offenders == []
