"""FE-2: avatar layers, their own license (not MIT), the persona block, and the silhouette fallback."""

from pathlib import Path

from driveflow.assistant import config as acfg

ROOT = Path(__file__).resolve().parents[2]


def test_persona_and_assets_present_and_small():
    a = acfg.avatar_assets()
    assert a["ok"] and a["missing"] == [] and a["name"] == "PhD. Radrigan" and a["tagline"] == "AI assistant · local model"
    total = sum(p.stat().st_size for p in a["files"].values())
    assert total < 1_000_000  # FE-3 (e): resources < 1 MB
    for p in a["files"].values():
        assert p.read_bytes()[:4] == b"RIFF" and p.read_bytes()[8:12] == b"WEBP"


def test_missing_directory_falls_back_without_failing(tmp_path, caplog):
    cfg = {**acfg.load_config(), "persona": {"name": "X", "avatar_dir": str(tmp_path)}}
    a = acfg.avatar_assets(cfg)
    assert not a["ok"] and set(a["missing"]) == {"base.webp", "hand.webp"} and "silhouette" in caplog.text


def test_avatar_license_is_separate_and_mit_untouched():
    lic = (ROOT / "licenses" / "avatar" / "LICENSE").read_text()
    assert "All rights reserved" in lic and "Not covered by the MIT license" in lic
    assert "MIT" in (ROOT / "LICENSE").read_text()
    assert "licenses/avatar/" in (ROOT / "README.md").read_text()


def test_no_original_caricature_in_the_repository():
    import subprocess

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    # Golden-set photos of motor rating plates (round C, T8): CC0 / CC BY Wikimedia files, each listed with its author and
    # licence in tests/assistant/golden/photos/ATTRIBUTION.md. They are excluded by name only; any other JPG fails.
    golden = Path("tests/assistant/golden/photos")
    attributed = {f"{golden}/{line.split('`')[1]}" for line in (ROOT / golden / "ATTRIBUTION.md").read_text().splitlines() if line.startswith("| W") and "`" in line}
    assert not [f for f in tracked if f.lower().endswith((".jpg", ".jpeg")) and "assistant" in f.lower() and f not in attributed]
