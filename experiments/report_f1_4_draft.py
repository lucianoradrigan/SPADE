"""Phase 1, F1.4: the assistant's draft with provenance, and the no-invention rule (C3). No language model.

1. A configuration request written by hand as the patches a model would send: the draft, its
   status() and the Scenario list it expands to.
2. The comparison rule on explicit cases: user values that must be accepted and invented values
   that must be refused (false positives / false negatives of the rule).
3. Bit-for-bit: a draft that touches no field expands to the Scenario a caller writes by hand.

    python experiments/report_f1_4_draft.py
Writes docs/audit/assistant/f1_4_draft.{md,json}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from driveflow.assistant.draft import Draft, Ledger
from driveflow.datagen import Scenario

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f1_4_draft"
REQUEST = "Simula el motor DC de stock con PI a 1500 rpm, 2 s por corrida, con carga del 20 al 100 % cada 20 %."
PATCH = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"},
         {"op": "set", "field": "omega_ref_rad_s", "value": 1500, "unit": "rpm"}, {"op": "set", "field": "duration_s", "value": 2, "unit": "s"},
         {"op": "vary", "field": "load_torque_nm", "range": {"start": 20, "stop": 100, "step": 20}, "unit": "%"}]
# (user text, value, unit, field, expected accepted)
CASES = [
    ("a 3000 rpm", 3000, "rpm", "omega_ref_rad_s", True),
    ("a 3000 rpm", 3000 * 2 * math.pi / 60, "rad/s", "omega_ref_rad_s", True),
    ("a 3000 rpm", 314, "rad/s", "omega_ref_rad_s", False),
    ("a 3000 rpm", 314.16, "rad/s", "omega_ref_rad_s", False),
    ("a 300 rpm", 300, "rad/s", "omega_ref_rad_s", False),
    ("a 1.500 rpm", 1500, "rpm", "omega_ref_rad_s", False),
    ("a 1.500 rpm", 1.5, "rpm", "omega_ref_rad_s", False),
    ("a 1.500,5 rpm", 1500.5, "rpm", "omega_ref_rad_s", True),
    ("alimentación de 220 V", 220, "V", "dc_supply_v", True),
    ("alimentación de 220 V", 230, "V", "dc_supply_v", False),
    ("tres segundos", 3, "s", "duration_s", True),
    ("un motor", 1, "s", "duration_s", False),
    ("del 20 al 100 % de carga", 60, "%", "load_torque_nm", False),
    ("del 20 al 100 % de carga", 20, "%", "load_torque_nm", True),
]


def main() -> None:
    led = Ledger()
    led.add_user_message(1, REQUEST)
    d = Draft()
    r = d.patch(PATCH, led)
    st = d.status()
    rows = []
    for text, value, unit, fld, expected in CASES:
        lg = Ledger()
        lg.add_user_message(1, text)
        dd = Draft()
        dd.patch([{"op": "set_plant", "plant": "dc_perm_ex_v1"}], lg)
        got = dd.patch([{"op": "set", "field": fld, "value": value, "unit": unit}], lg)["ok"]
        rows.append({"text": text, "value": value, "unit": unit, "field": fld, "expected": expected, "accepted": got, "pass": got == expected})
    blank = Draft()
    blank.patch([{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"}], led)
    identical = blank.expand()[0] == Scenario("assistant_000", controller_type="PI", plant_config_id="dc_perm_ex_v1")
    result = {"command": "python experiments/report_f1_4_draft.py", "llm": "none (patches written by hand)", "request": REQUEST, "patch": PATCH,
              "patch_ok": r["ok"], "draft": d.to_dict(), "status": st, "conditions": d.conditions(), "comparison_cases": rows,
              "untouched_draft_equals_hand_written_scenario": identical}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    lines = [
        "# F1.4 — the assistant's draft with provenance",
        "",
        "**No language model involved:** the patches below are written by hand, as a model would send them. Simulations are not run here (only `Scenario` validation).",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f1_4_draft.json`](f1_4_draft.json). Tests: `tests/assistant/test_assistant_draft.py`, `tests/assistant/test_assistant_numbers.py`.",
        "- **Code:** `src/driveflow/assistant/draft.py` (draft, ledger of evidence, atomic patches, `status()`, `expand()`, `hash()`, `audit_trace()`), `src/driveflow/assistant/numbers.py` (quantities in the user's text, unit table, comparison rule).",
        "",
        "## 1. A configuration request",
        "",
        f"> {REQUEST}",
        "",
        "| Field | Value | Unit | Origin | Source |",
        "|---|---|---|---|---|",
    ]
    for path, v in [*d.preset.items(), *d.base.items()]:
        lines.append(f"| `{path}` | {v.value} | {v.unit} | {v.origin} | {v.source} |")
    for path, vs in d.vary.items():
        for v in vs:
            lines.append(f"| `{path}` (varied) | {v.value} | {v.unit} | {v.origin} | {v.source} |")
    lines += [
        "",
        "Load torque SPADE derives from the percentages: " + ", ".join(f"{x['value']:g} N·m ({x['formula']})" for x in st["derived"]) + f"; rated torque from: {st['derived'][0]['rated_torque_from']}.",
        f"Status: ready **{st['ready']}**, {st['experiments']} experiments, {st['cost']['sim_seconds']:g} simulated s, about {st['cost']['est_cpu_s']:.0f} CPU s ({st['cost']['basis']}). "
        f"Important fields left at their default (reported, not asked): {', '.join(m['field'] for m in st['missing_important'])}.",
        "",
        "## 2. Comparison rule (C3): explicit cases",
        "",
        "| User wrote | Patch value | Expected | Result |",
        "|---|---|---|---|",
    ] + [f"| {x['text']} | {x['value']:g} {x['unit']} → `{x['field']}` | {'accept' if x['expected'] else 'refuse'} | {'ok' if x['pass'] else '**WRONG**'} |" for x in rows] + [
        "",
        f"{sum(x['pass'] for x in rows)}/{len(rows)} cases behave as specified. Rules: SI conversion with one unit table, relative tolerance 1e-9, decimal comma or point, "
        "a lone separator before three digits (\"1.500\") is ambiguous and never a user value, \"un\"/\"una\" are articles, a range yields only its endpoints (intermediate values need the user's step and are then `derived`), Hz never converts to rad/s.",
        "",
        "## 3. Untouched fields",
        "",
        f"A draft with only plant and controller expands to exactly `Scenario('assistant_000', controller_type='PI', plant_config_id='dc_perm_ex_v1')`: **{identical}**. "
        "Defaults accepted with `use_defaults` are shown with origin `default` but not passed to `Scenario`, so the run stays the untouched one (passing the stock motor's values as overrides would take another code path, e.g. the 0b-D1 load rule).",
        "",
        "## Not executed",
        "",
        "- The no-invention check over recorded cassettes: no ejecutado todavía (no cassettes exist; they arrive with F1.2). `audit_trace()` and its test are in place and will run on every cassette.",
        "- Origin `datasheet` (phase 2) and fault fields (phase 4) are refused by design.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
