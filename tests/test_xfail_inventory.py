"""T4 (round 2026-10-22): every xfail in the suite is strict and cites a finding ID, so it cannot
outlive the defect silently and it can be traced to docs/AI_TECHNICAL_AUDIT.md or
docs/SIMULATION_STABILITY_REPORT.md. Inventory: docs/audit/expansion/xfail_inventory.md."""

import ast
import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent
FINDING_ID = re.compile(r"\b(AI-N\d+|S-\d+)\b")


def _xfail_markers():
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "xfail":
                kw = {k.arg: k.value for k in node.keywords}
                yield path.relative_to(TESTS).as_posix(), node.lineno, kw


def test_every_xfail_is_strict_and_cites_a_finding():
    found = list(_xfail_markers())
    assert found, "the scan found no xfail at all"
    bad = []
    for rel, line, kw in found:
        strict = isinstance(kw.get("strict"), ast.Constant) and kw["strict"].value is True
        reason = kw["reason"].value if isinstance(kw.get("reason"), ast.Constant) else ""
        if not strict or not FINDING_ID.search(str(reason)):
            bad.append(f"{rel}:{line} strict={strict} reason={reason!r}")
    assert not bad, "xfail without strict=True or without a finding ID:\n" + "\n".join(bad)


def test_cited_findings_exist_in_the_audit_documents():
    docs = TESTS.parent / "docs"
    text = (docs / "AI_TECHNICAL_AUDIT.md").read_text() + (docs / "SIMULATION_STABILITY_REPORT.md").read_text()
    for rel, line, kw in _xfail_markers():
        for fid in FINDING_ID.findall(str(kw["reason"].value)):
            assert fid in text, f"{rel}:{line} cites {fid}, which no audit document defines"
