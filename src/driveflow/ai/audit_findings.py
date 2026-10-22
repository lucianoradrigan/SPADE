"""Measured audit findings attached to a run directory (AUDIT Q15).

A run's metrics.json is its training-time record and is never rewritten. When a later, reproducible
experiment shows that record to be misleading, the finding is stored next to it as
`audit_findings.json` -- {"findings": [{"id", "severity", "summary", "evidence", "date"}]} -- and
every view that shows the recorded metric must show the finding too.
"""

from __future__ import annotations

import json
from pathlib import Path

FINDINGS_FILE = "audit_findings.json"


def load_audit_findings(run_dir) -> list:
    path = Path(run_dir) / FINDINGS_FILE
    if not path.exists():
        return []
    try:
        findings = json.loads(path.read_text()).get("findings", [])
    except (json.JSONDecodeError, AttributeError):
        return [{"id": "unreadable", "severity": "high", "summary": f"{FINDINGS_FILE} is unreadable -- treat the recorded metrics as unverified.", "evidence": str(path), "date": None}]
    return [f for f in findings if isinstance(f, dict)]


#: Classes whose fault amplitude is a placeholder, not calibrated against measured data
#: (CALIBRATED_MECHANICAL_SEVERITY in datagen/scenario.py: KAt has no clean single-defect code for them).
SYNTHETIC_CLASSES = ("ball", "cage")
SYNTHETIC_LABEL = "ball/cage: synthetic-only, uncalibrated amplitude"
SYNTHETIC_FINDING = {
    "id": "A2-synthetic-ball-cage",
    "severity": "high",
    "summary": SYNTHETIC_LABEL + ". The ball and cage classes were trained on simulated faults whose mechanical amplitude "
               "(CALIBRATED_MECHANICAL_SEVERITY in datagen/scenario.py) is a placeholder: KAt has no clean single-defect code for these faults, "
               "so it cannot be calibrated without real data.",
    "evidence": "docs/AI_TECHNICAL_AUDIT.md Sec. 8.4-8.5",
}


def is_synthetic_class(label) -> bool:
    return label in SYNTHETIC_CLASSES


def ensure_synthetic_class_finding(run_dir, classes, date: str | None = None) -> bool:
    """Adds SYNTHETIC_FINDING to run_dir/audit_findings.json when `classes` include ball or cage and it
    is not there yet (every run that can answer ball/cage must carry it). Returns True if written."""
    if not any(is_synthetic_class(c) for c in (classes or [])):
        return False
    path = Path(run_dir) / FINDINGS_FILE
    data = json.loads(path.read_text()) if path.exists() else {"findings": []}
    if any(SYNTHETIC_LABEL in str(f.get("summary", "")) for f in data.get("findings", [])):
        return False
    import datetime as dt

    data.setdefault("findings", []).append({**SYNTHETIC_FINDING, "date": date or dt.date.today().isoformat()})
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True
