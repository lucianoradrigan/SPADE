"""B8 (round PLECS/OM): the licence review stopped the Release, so `python -m driveflow.fmu fetch`
is disabled and says why, and the guide to build the FMUs locally exists."""

import subprocess
import sys
from pathlib import Path

from driveflow.fmu import FETCH_DISABLED
from driveflow.fmu.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_is_disabled_with_the_reason(capsys):
    assert main(["fetch", "--model", "BuckAveraged"]) == 2
    err = capsys.readouterr().err
    assert "disabled" in err and "b8_license_review.md" in err and "BUILD_FMUS_LOCALLY.md" in err


def test_fetch_as_a_module_downloads_nothing(tmp_path):
    r = subprocess.run([sys.executable, "-m", "driveflow.fmu", "fetch"], cwd=tmp_path, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert r.returncode == 2 and FETCH_DISABLED in r.stderr and not any(tmp_path.iterdir())


def test_guide_and_review_exist_and_cross_reference():
    guide = (ROOT / "docs" / "guides" / "BUILD_FMUS_LOCALLY.md").read_text()
    review = (ROOT / "docs" / "audit" / "plecs_om" / "b8_license_review.md").read_text()
    assert "scripts/fmu/build_fmus.sh" in guide and "b8_license_review.md" in guide
    assert "not clear" in review.lower() and "Publishing the Release still needs your confirmation" in review
