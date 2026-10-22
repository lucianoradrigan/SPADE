"""Re-evaluates the candidates documented in docs/audit/ai/custom_training_tl.json against the D1
acceptance floors (2026-10-22: macro-F1 >= 0.88, recall >= 0.70 for every class). Uses the metrics
recorded there -- nothing is retrained. A recall that was not recorded is reported as such.

Usage: python experiments/report_tl_gate_floors.py  ->  docs/audit/ai/tl_gate_floors.{md,json}
"""

import json
from pathlib import Path

from driveflow.ai.transfer.validator import floor_checks, resolve_floors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "audit" / "ai" / "custom_training_tl.json"
OUT = ROOT / "docs" / "audit" / "ai"


def _row(name, macro, recalls):
    floors = resolve_floors(None)
    checks = {c.name: (c.status.value, c.detail) for c in floor_checks(macro, recalls or {}, floors)}
    if recalls is None:
        checks["min_per_class_recall"] = ("NOT RECORDED", "per-class recall not stored in custom_training_tl.json")
    blocked = any(v[0] != "PASS" for v in checks.values())
    return {"candidate": name, "macro_f1": macro, "recall_per_class": recalls, "checks": checks,
            "floors_verdict": "BLOCKED" if blocked else "PASS (still needs human approval)"}


def main():
    src = json.loads(SRC.read_text())
    a = src["A_custom_training"]
    b = next(r for r in src["B_full_vs_feature"]["results"] if r["strategy"] == "new_head")
    c = src["C_new_class"]
    rows = [
        _row("A · GUI-trained 4-class model (custom training, not a TL candidate)", a["test_macro_f1"], a["test_per_class_recall"]),
        _row("B · NEW_HEAD from the promoted 5-class model, shifted condition", b["candidate_macro_f1"], None),
        _row("C · NEW_HEAD adding 'unbalance' to pc_server/2026-10-18_run01", c["candidate_macro_f1"], c["candidate_recall_per_class"]),
    ]
    out = {"command": "python experiments/report_tl_gate_floors.py", "source": str(SRC.relative_to(ROOT)), "floors": resolve_floors(None), "candidates": rows}
    (OUT / "tl_gate_floors.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# D1 floors applied to the documented candidates (round 2026-10-22, T5)", "",
             f"Command: `{out['command']}`. Source metrics: [`custom_training_tl.json`](custom_training_tl.json), recorded on 2026-10-21; nothing was retrained. "
             "Floors: macro-F1 ≥ 0.88 and recall ≥ 0.70 for every class (default D1 2026-10-22).", "",
             "| Candidate | macro-F1 | min_macro_f1 | min_per_class_recall | Floors verdict |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['macro_f1']:.3f} | {r['checks']['min_macro_f1'][0]} | {r['checks']['min_per_class_recall'][0]}: {r['checks']['min_per_class_recall'][1]} | {r['floors_verdict']} |")
    lines += ["", "A is not a transfer-learning candidate (no gate applies to it); it is listed because the request asked for it. Under the same floors it would be blocked.",
              "Every candidate also needs a named reviewer's approval: the floors never approve on their own.", ""]
    (OUT / "tl_gate_floors.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
