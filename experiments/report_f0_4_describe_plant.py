"""Round Assistant, task F0.4: the output of describe_plant() for the four native plants.

    python experiments/report_f0_4_describe_plant.py
Writes docs/audit/assistant/f0_4_describe_plant.{md,json}.
"""

from __future__ import annotations

import json
from pathlib import Path

from driveflow.datagen.plant_catalog import NATIVE_PLANTS, describe_plant

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f0_4_describe_plant"


def _range(r) -> str:
    if r is None:
        return "—"
    if r == {}:
        return "any finite"
    lo = "" if r.get("min") is None else f"{'[' if r.get('min_inclusive', True) else '('}{r['min']:g}"
    hi = "" if r.get("max") is None else f"{r['max']:g}{']' if r.get('max_inclusive', True) else ')'}"
    return f"{lo or '(-∞'}, {hi or '∞)'}"


def _cell(v) -> str:
    return "—" if v in (None, "") else str(v).replace("|", "\\|")


def main() -> None:
    out = {p: describe_plant(p) for p in NATIVE_PLANTS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    lines = [
        "# F0.4 — `describe_plant()`: what each native plant lets you configure",
        "",
        "- **Command:** `python experiments/report_f0_4_describe_plant.py`. Full output: [`f0_4_describe_plant.json`](f0_4_describe_plant.json).",
        "- **Source of truth:** `configs/plant_catalog.yaml` (units, conventions, ranges, importance, descriptions). Defaults are read from the code, not from the YAML.",
        "- **Guard:** `tests/test_assistant_f0_plant_catalog.py`:",
        "  - every `Scenario` field is described or marked `not_applicable` for each of the 4 plants;",
        "  - every described field exists;",
        "  - every range is probed against `Scenario.__post_init__` just inside and just outside each bound (also each key of the dict fields).",
        "  - A mutation check (three ranges altered in memory) was detected by the test each time.",
        "",
        "Range notation: `[` inclusive, `(` exclusive; \"any finite\" = only finiteness is checked; \"—\" = no numeric range (string, dict or unchecked field).",
    ]
    for plant, d in out.items():
        lines += ["", f"## `{plant}` — {d['title']}", "", f"Controllers: {', '.join(d['controllers'])}.", "",
                  "| Field | Type | Unit | Range | Default | Importance | Convention / description |", "|---|---|---|---|---|---|---|"]
        for f in d["fields"]:
            desc = f["description"] + (f" Convention: {f['convention']}." if f.get("convention") else "") + (f" Allowed: {', '.join(f['allowed'])}." if f.get("allowed") else "") \
                + (f" ({f['range_note']})" if f.get("range_note") else "")
            lines.append(f"| `{f['name']}` | {f['type']} | {_cell(f['unit'])} | {_range(f['range'])} | {_cell(f['default'])} | {f['importance']} | {_cell(desc)} |")
            for k, v in (f.get("keys") or {}).items():
                lines.append(f"| ↳ `{k}` | {'int' if v.get('integer') else 'float'} | {_cell(v.get('unit'))} | {_range(v['range'])} | {_cell(v['default'])} | | {_cell(v.get('convention'))} |")
        lines.append(f"\nNot applicable here: {', '.join(f'`{n}`' for n in d['not_applicable'])}.")
        if "stock_motor" in d:
            s = d["stock_motor"]
            pi, mpc = s["controller_design_check"]["PI"], s["controller_design_check"]["MPC"]
            lines += ["", "**Stock motor:** "
                      f"parameters {s['motor_parameter']}; limits {s['limits']}; nominal {s['nominal_values']}; supply {s['supply_v']:g} V; load {s['load']}; "
                      f"j_total {s['j_total_kg_m2']:g} kg·m²; control period {s['control_period_s']:g} s.",
                      "", f"**Design check:** PI **{pi['status']}** (τ_a {pi['tau_a_s'] * 1e3:.3g} ms, τ_m {pi['tau_m_s'] * 1e3:.3g} ms, τ_m/τ_a {pi['tau_m_over_tau_a']:.1f}, "
                      f"bandwidth ratio {pi['bandwidth_ratio']:.1f}); MPC **{mpc['status']}** (horizon/τ_a {mpc['horizon_over_tau_a']:.2f})."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
