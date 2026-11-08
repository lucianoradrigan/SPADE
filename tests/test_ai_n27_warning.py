"""AI-N27: every promoted dc_motor classifier carries the AI-N27 finding about the S-21 simulator
artifact, shown by Model Registry next to the run. Since 2026-11-04 (option 2) the PC and RPi5 entries
are runs trained on the corrected (dcm x1000) data, where it is mitigated; the ESP32 entry is still
the run trained on data with the artifact, where it stays open."""

import yaml

from driveflow.ai.audit_findings import load_audit_findings
from driveflow.viz.adapters import registry_adapter

PROMOTED = ("dc_motor/esp32/classifier", "dc_motor/pc/classifier", "dc_motor/rpi5/classifier")


def test_promoted_dc_motor_classifiers_carry_ai_n27():
    entries = yaml.safe_load(open("configs/registry.yaml"))["entries"]
    for key in PROMOTED:
        findings = load_audit_findings(entries[key])
        ai = [f for f in findings if f.get("id") == "AI-N27"]
        assert len(ai) == 1, key
        if key == "dc_motor/esp32/classifier":
            assert "open" in ai[0]["summary"] and ai[0]["severity"] == "high"
            assert "trained on data that contain" in ai[0]["summary"] and "AI-N28" in ai[0]["summary"]
        else:
            assert "mitigated" in ai[0]["summary"] and "dc_perm_ex_v1" in ai[0]["summary"]
            assert any(f.get("id") == "dcm-trained" for f in findings), key
            assert entries[key].endswith("2026-11-04_run01"), key


def test_previous_rpi5_does_not_pass_d1_on_corrected_physics():
    findings = load_audit_findings("configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01")
    sup = [f for f in findings if f.get("id") == "superseded-dcm"]
    assert len(sup) == 1 and "0.870" in sup[0]["summary"] and "D1" in sup[0]["summary"]


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


def test_promoted_esp32_records_the_failed_repair_round():
    """AI-N28 (decision of 2026-11-08): the promoted ESP32 says it was trained on data with the artifact
    (AI-N27) and that the pre-registered repair round did not meet its criterion."""
    entries = yaml.safe_load(open("configs/registry.yaml"))["entries"]
    assert entries["dc_motor/esp32/classifier"].endswith("esp32_dscnn/2026-10-18_run01")
    f = [x for x in load_audit_findings(entries["dc_motor/esp32/classifier"]) if x.get("id") == "AI-N28"]
    assert len(f) == 1 and f[0]["severity"] == "high"
    s = f[0]["summary"]
    assert "did not meet" in s and "0.560 -> 0.717" in s and "0.994 -> 0.687" in s and "+19 points" in s and "AI-N27" in s and "B4" in s
