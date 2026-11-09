"""Ratio checks of configs/assistant_conversions.yaml (Kollmorgen WP RevD, p. 8) on hand-written golden expected values.
A helper for tests and for experiments/report_golden_set.py; it reports which convention fits, it never decides one."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONVERSIONS = yaml.safe_load((ROOT / "configs" / "assistant_conversions.yaml").read_text())
EXPECTED = Path(__file__).resolve().parent / "expected"

#: printed unit -> factor to volts per 1000 rpm
KE_UNITS = {"v/1000*min¯¹": 1.0, "v/krpm": 1.0, "vp-p/krpm": 1.0, "vrms/krpm": 1.0, "mv/rpm": 1.0, "v/rpm": 1000.0}
#: (doc, variant) -> (voltage-constant field, torque-constant field) in the expected files
PAIRS = {"P1": ("voltage_constant", "torque_constant"), "P2": ("back_emf_constant", "torque_constant"),
         "P3": ("voltage_constant", "torque_constant"), "P4": ("back_emf_constant", "torque_constant"), "P5": ("back_emf_constant", "torque_constant")}


def _number(text: str) -> float:
    """The SI value of a printed number: '0.41' in '57.6 [0.41]' (the bracket is the SI value on B23), decimal comma accepted."""
    m = re.search(r"\[([\d.,]+)\]", text) or re.search(r"[\d.,]+", text)
    return float(m.group(1 if m.re.pattern.startswith("\\[") else 0).replace(",", "."))


def ratio_checks() -> list[dict]:
    checks = {c["id"]: c for c in CONVERSIONS["checks"]}
    out = []
    for doc, (ke_name, kt_name) in PAIRS.items():
        path = EXPECTED / f"{doc}.json"
        if not path.exists():
            out.append({"doc": doc, "status": "not executed: document unavailable (no expected values)"})
            continue
        fields = json.loads(path.read_text())["fields"]
        variants = sorted({f["variant"] for f in fields if f["field"] in (ke_name, kt_name)}, key=str)
        for v in variants:
            ke = next(f for f in fields if f["field"] == ke_name and f["variant"] == v)
            kt = next(f for f in fields if f["field"] == kt_name and f["variant"] == v)
            unit = (ke["unit_printed"] or "").lower().replace(" ", "")
            ke_v = _number(ke["value_printed"]) * KE_UNITS[unit]
            kt_v = _number(kt["value_printed"])
            ratio = ke_v / kt_v
            fits = [cid for cid, c in checks.items() if c["low"] <= ratio <= c["high"]]
            out.append({"doc": doc, "variant": v, "ke_printed": f"{ke['value_printed']} {ke['unit_printed']}", "kt_printed": f"{kt['value_printed']} {kt['unit_printed']}",
                        "ratio": round(ratio, 2), "fits": fits or ["none"],
                        "reading": {"kb_over_kt_sine_rms": "both RMS (sinusoidal commutation)", "ke_over_kt_six_step": "both 6-step / peak"}.get(fits[0], "") if fits
                        else "no convention of the white paper fits: ask the user which constant to trust"})
    return out
