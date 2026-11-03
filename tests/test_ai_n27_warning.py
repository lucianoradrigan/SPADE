"""AI-N27 (2026-11-02): the three promoted dc_motor classifiers carry the open warning about the S-21
simulator artifact in their training data, shown by Model Registry next to each run. The models and
their promotion are unchanged."""

import yaml

from driveflow.ai.audit_findings import load_audit_findings
from driveflow.viz.adapters import registry_adapter

PROMOTED = ("dc_motor/esp32/classifier", "dc_motor/pc/classifier", "dc_motor/rpi5/classifier")


def test_promoted_dc_motor_classifiers_carry_ai_n27():
    entries = yaml.safe_load(open("configs/registry.yaml"))["entries"]
    for key in PROMOTED:
        findings = load_audit_findings(entries[key])
        ai = [f for f in findings if f.get("id") == "AI-N27"]
        assert len(ai) == 1 and "open" in ai[0]["summary"] and ai[0]["severity"] == "high", key
        assert "unchanged" in ai[0]["summary"] and "affects" in ai[0]["summary"]


def test_model_registry_lists_it_with_the_run():
    """registry_adapter.list_runs feeds the Model Registry view, which shows each audit finding."""
    from pathlib import Path

    entries = yaml.safe_load(open("configs/registry.yaml"))["entries"]
    rows = registry_adapter.list_runs()
    for key in PROMOTED:
        target = Path(entries[key]).resolve()
        match = [r for r in rows if Path(r["run_dir"]).resolve() == target]
        assert match, key
        assert any(f.get("id") == "AI-N27" for f in match[0]["audit_findings"]), key
