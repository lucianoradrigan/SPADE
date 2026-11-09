"""T8 (round C, decision C10): the golden set of phase 2 is catalogued and its expected values are complete.
No model runs on it (phase 2 has not started); these tests check the catalogue, not an extractor."""

import json
from pathlib import Path

import pytest
import yaml

from tests.assistant.golden import conventions

GOLDEN = Path(__file__).resolve().parent / "golden"
MANIFEST = yaml.safe_load((GOLDEN / "manifest.yaml").read_text())
FIELD_KEYS = {"field", "value_printed", "unit_printed", "label_printed", "basis", "waveform", "temperature_c", "condition", "variant"}


def test_documents_are_catalogued_and_never_committed():
    docs = MANIFEST["documents"]
    assert {d["id"] for d in docs} >= {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "I1", "I2", "R1"}
    for d in docs:
        assert d["committed"] is False
        assert d["status"] in ("downloaded", "unavailable")
        if d["status"] == "downloaded":
            assert len(d["sha256"]) == 64 and d["url"].startswith("http") and d["consulted_utc"]
        else:
            assert d["unavailable_reason"]
        assert isinstance(d["simulable_today"], bool) or d["id"] == "R1"


def test_at_least_twelve_documents_have_hand_written_expected_values():
    written = [d for d in MANIFEST["documents"] if (GOLDEN / "expected" / f"{d['id']}.json").exists()]
    assert len(written) >= 12


@pytest.mark.parametrize("path", sorted((GOLDEN / "expected").glob("[DPI][0-9]*.json")), ids=lambda p: p.stem)
def test_document_expected_values_follow_the_schema(path):
    e = json.loads(path.read_text())
    assert e["fields"] and isinstance(e["must_ask"], list) and isinstance(e["must_warn"], list)
    for f in e["fields"]:
        assert set(f) == FIELD_KEYS
        assert f["basis"] in ("phase", "line-line", "unknown", "not_applicable")
        assert f["waveform"] in ("rms", "peak", "dc_equivalent", "unknown", "not_applicable")


def test_photos_follow_their_licence_rule():
    for ph in MANIFEST["photos"]:
        committed_path = Path(ph["path"]).parts[:1] == ("tests",)
        assert ph["committed"] == committed_path
        assert ph["committed"] == (ph["licence_class"] in ("cc0", "cc_by")), ph["id"]
        if ph["committed"]:
            assert (GOLDEN.parents[2] / ph["path"]).exists()
            assert ph["id"] in (GOLDEN / "photos" / "ATTRIBUTION.md").read_text()
        assert (GOLDEN / "expected" / f"{ph['id']}.json").exists()


def test_only_open_licence_photos_are_in_git():
    on_disk = {p.name for p in (GOLDEN / "photos").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    allowed = {Path(ph["path"]).name for ph in MANIFEST["photos"] if ph["committed"]}
    assert on_disk == allowed


def test_the_ratio_checks_report_without_failing():
    """Reports which convention of the Kollmorgen white paper fits P1-P5 (not a pass/fail test)."""
    report = conventions.ratio_checks()
    print(json.dumps(report, indent=1, ensure_ascii=False))
    assert {r["doc"] for r in report} == {"P1", "P2", "P3", "P4", "P5"}
