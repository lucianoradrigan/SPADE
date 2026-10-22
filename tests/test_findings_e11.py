"""E11a (round 2026-10-22): the IGBT data audit's recorded conclusion (docs/AI_TECHNICAL_AUDIT.md §11,
AI-N23). Read from the recorded report so a regenerated audit updates it."""

import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "ai" / "e11a_igbt_data_audit.json"


def test_ai_n23_fewer_than_three_run_to_failure_devices_on_a_comparable_protocol():
    rep = json.loads(AUDIT.read_text())
    assert rep["max_run_to_failure_on_one_comparable_protocol"] < 3
    assert rep["conclusion"] == "NEGATIVE" and "a validated RUL model is not feasible" in rep["finding"]


def test_every_device_in_the_archive_is_accounted_for():
    rep = json.loads(AUDIT.read_text())
    names = {d["device"] for d in rep["devices"]}
    assert {"Device 2", "Device 3", "Device 4", "Device 5", "A17 (square gate)", "DC-gate (20080429T135531)"} == names
    assert all(d["source"] for d in rep["devices"])
