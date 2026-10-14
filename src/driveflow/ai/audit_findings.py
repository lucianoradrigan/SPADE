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
