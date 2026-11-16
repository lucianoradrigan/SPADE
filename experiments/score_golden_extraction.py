"""Phase 2 scorer (R4, 2026-11-15; decision R-D6): compares an extraction output with the golden expected file of the
same document (tests/assistant/golden/expected/<id>.json). No LLM and no extractor is run here: this only measures.

    python experiments/score_golden_extraction.py --outputs <dir with <id>.json> [--ids P1 W01 ...] [--rtol 0.02] [--json out.json]

Extraction output (one JSON per document; the same field records as the expected files, plus what the agent asked and
warned; a field may declare a conversion):

    {"id": "P1", "simulable_today": true,
     "fields": [{"field": "kt", "label_printed": "...", "value": 1.4, "unit": "Nm/A", "basis": "line-line", "waveform": "rms",
                 "variant": null, "converted_by": null}, ...],
     "asked": ["free text of each question the agent asked"],
     "warned": ["free text of each warning the agent gave"]}

A field is matched to the expected field with the same (field or label_printed, variant); a key printed several times pairs by order. Metrics, per document and in total:

- values: a printed, legible expected value counts as correct only if the output value equals it within the relative
  tolerance AND the unit, the basis (phase / line-line) and the waveform (rms / peak / dc_equivalent) agree. A different
  unit, basis or waveform is wrong even if the number is right. A converted value (e.g. Ke peak -> Kb rms) is accepted
  only if the field declares "converted_by": <rule id of configs/assistant_conversions.yaml> and the value equals the
  printed one times that rule's factor; an undeclared conversion is wrong.
- invention (severe, separate): any value given where the expected one is null, "illegible" or "ambiguous: must ask".
- questions: share of must_ask items the agent asked (text match, see _matches), and the questions that ask for a
  value that is printed and legible (unneeded questions).
- warnings: share of must_warn items the agent gave.
- simulable: whether the output's simulable_today equals the expected one.

Text matching of questions and warnings is lenient by design (the agent paraphrases): normalised exact match, or a share
of the expected item's content words present in the answer >= --text-threshold (default 0.6). The threshold is a
proposal, not a decision (docs/audit/assistant/phase2_scoring_proposal.md, DRAFT).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EXPECTED_DIR = REPO / "tests" / "assistant" / "golden" / "expected"
CONVERSIONS = REPO / "configs" / "assistant_conversions.yaml"
NOT_GIVEN = (None, "illegible", "ambiguous: must ask")
SPECIFIC_BASIS = ("phase", "line-line")
SPECIFIC_WAVE = ("rms", "peak", "dc_equivalent")
STOP = set("the a an of and or to in on is are be with for by at from as it its this that not no if than then into per same".split())


# ----------------------------------------------------------------------------------- parsing
def _number(tok: str):
    t = tok.strip().replace(" ", "")
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", t):  # 3,000 / 1,234.5: thousands separators
        t = t.replace(",", "")
    elif re.fullmatch(r"[-+]?\d+,\d+", t):  # 0,8: decimal comma
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def parse_value(v):
    """A number, a tuple of numbers ("230/400"), or a normalised string; None stays None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    parts = re.split(r"\s*/\s*", s)
    nums = [_number(p) for p in parts]
    if all(n is not None for n in nums):
        return nums[0] if len(nums) == 1 else tuple(nums)
    return re.sub(r"\s+", " ", s).lower()


def _norm_unit(u) -> str | None:
    if u is None:
        return None
    s = str(u).strip().lower().replace("·", "").replace("*", "").replace(" ", "")
    return {"nm": "nm", "n.m": "nm"}.get(s, s)


def values_equal(a, b, rtol: float) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(math.isclose(x, y, rel_tol=rtol, abs_tol=1e-12) for x, y in zip(a, b))
    return a == b


def _words(s: str) -> list:
    return [w for w in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", s.lower()) if w not in STOP]


def _matches(expected_item: str, said: list, threshold: float) -> bool:
    e = re.sub(r"\s+", " ", expected_item.strip().lower())
    ew = set(_words(expected_item))
    for s in said:
        if re.sub(r"\s+", " ", str(s).strip().lower()) == e:
            return True
        if ew and len(ew & set(_words(str(s)))) / len(ew) >= threshold:
            return True
    return False


# ----------------------------------------------------------------------------------- scoring
def _key(f: dict):
    return (f.get("field") or f.get("label_printed"), f.get("variant"))


def load_rules(path: Path = CONVERSIONS) -> dict:
    return {r["id"]: float(r["factor"]) for r in (yaml.safe_load(path.read_text()) or {}).get("rules", [])}


def score_document(expected: dict, output: dict | None, rtol: float = 0.02, text_threshold: float = 0.6, rules: dict | None = None) -> dict:
    rules = load_rules() if rules is None else rules
    output = output or {}
    # A label can be printed more than once on a plate (W03, W10): the n-th expected field with a key pairs with the n-th
    # output field with the same key.
    out_fields, seen = {}, {}
    for f in output.get("fields") or []:
        if isinstance(f, dict):
            k = _key(f)
            seen[k] = seen.get(k, 0) + 1
            out_fields[(k, seen[k])] = f
    exp_seen = {}
    res = {"id": expected.get("id"), "values_correct": 0, "values_total": 0, "value_errors": [], "inventions": [],
           "must_ask_hit": 0, "must_ask_total": len(expected.get("must_ask") or []), "missed_questions": [], "unneeded_questions": [],
           "must_warn_hit": 0, "must_warn_total": len(expected.get("must_warn") or []), "missed_warnings": [],
           "simulable_expected": expected.get("simulable_today"), "simulable_output": output.get("simulable_today")}
    printed_terms = []
    for ef in expected.get("fields") or []:
        key = _key(ef)
        exp_seen[key] = exp_seen.get(key, 0) + 1
        exp_raw = ef.get("value_printed")
        got = out_fields.get((key, exp_seen[key]))
        if exp_raw in NOT_GIVEN:
            if got is not None and got.get("value") not in NOT_GIVEN:
                res["inventions"].append({"field": key[0], "variant": key[1], "expected": exp_raw, "given": got.get("value")})
            continue
        res["values_total"] += 1
        printed_terms += [t for t in (ef.get("field"), ef.get("label_printed")) if t]
        why = _value_error(ef, got, rtol, rules)
        if why is None:
            res["values_correct"] += 1
        else:
            res["value_errors"].append({"field": key[0], "variant": key[1], "why": why})
    asked = [str(a) for a in output.get("asked") or []]
    for item in expected.get("must_ask") or []:
        if _matches(item, asked, text_threshold):
            res["must_ask_hit"] += 1
        else:
            res["missed_questions"].append(item)
    for a in asked:
        if not any(_matches(m, [a], text_threshold) for m in expected.get("must_ask") or []):
            low = a.lower()
            hit = [t for t in printed_terms if len(str(t)) > 2 and str(t).lower().replace("_", " ") in low.replace("_", " ")]
            if hit:
                res["unneeded_questions"].append({"question": a, "printed": hit[0]})
    warned = [str(w) for w in output.get("warned") or []]
    for item in expected.get("must_warn") or []:
        if _matches(item, warned, text_threshold):
            res["must_warn_hit"] += 1
        else:
            res["missed_warnings"].append(item)
    res["simulable_match"] = res["simulable_output"] == res["simulable_expected"] and res["simulable_output"] is not None
    return res


def _value_error(ef: dict, got: dict | None, rtol: float, rules: dict) -> str | None:
    if got is None or got.get("value") in (None, ""):
        return "missing"
    exp, val = parse_value(ef.get("value_printed")), parse_value(got.get("value"))
    rule = got.get("converted_by")
    if rule:
        if rule not in rules:
            return f"unknown conversion rule {rule!r}"
        if not (isinstance(exp, float) and isinstance(val, float) and math.isclose(val, exp * rules[rule], rel_tol=rtol)):
            return f"declared conversion {rule} does not reproduce the value"
        return None  # a declared conversion changes unit / basis / waveform by definition
    if not values_equal(exp, val, rtol):
        return "value"
    if _norm_unit(ef.get("unit_printed")) != _norm_unit(got.get("unit")):
        return "unit"
    if ef.get("basis") in SPECIFIC_BASIS and got.get("basis") != ef.get("basis"):
        return "basis"
    if ef.get("basis") == "unknown" and got.get("basis") in SPECIFIC_BASIS:
        return "basis (claimed where the document does not say)"
    if ef.get("waveform") in SPECIFIC_WAVE and got.get("waveform") != ef.get("waveform"):
        return "waveform"
    if ef.get("waveform") == "unknown" and got.get("waveform") in SPECIFIC_WAVE:
        return "waveform (claimed where the document does not say)"
    return None


def summarize(docs: list) -> dict:
    def ratio(a, b):
        return None if b == 0 else a / b
    t = {k: sum(d[k] for d in docs) for k in ("values_correct", "values_total", "must_ask_hit", "must_ask_total", "must_warn_hit", "must_warn_total")}
    return {"documents": len(docs), "values": ratio(t["values_correct"], t["values_total"]), "values_correct": t["values_correct"], "values_total": t["values_total"],
            "inventions": sum(len(d["inventions"]) for d in docs), "must_ask": ratio(t["must_ask_hit"], t["must_ask_total"]),
            "unneeded_questions": sum(len(d["unneeded_questions"]) for d in docs), "must_warn": ratio(t["must_warn_hit"], t["must_warn_total"]),
            "simulable": ratio(sum(d["simulable_match"] for d in docs), len(docs))}


def perfect_output(expected: dict) -> dict:
    """The output an ideal extractor would give: every printed, legible value as printed, nothing invented, every
    must_ask asked and every must_warn given (used by the tests and as the format example)."""
    fields = [{"field": f.get("field"), "label_printed": f.get("label_printed"), "variant": f.get("variant"), "value": f.get("value_printed"),
               "unit": f.get("unit_printed"), "basis": f.get("basis"), "waveform": f.get("waveform")}
              for f in expected.get("fields") or [] if f.get("value_printed") not in NOT_GIVEN]
    return {"id": expected.get("id"), "simulable_today": expected.get("simulable_today"), "fields": fields,
            "asked": list(expected.get("must_ask") or []), "warned": list(expected.get("must_warn") or [])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, required=True, help="directory with one <id>.json extraction output per document")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--rtol", type=float, default=0.02)
    ap.add_argument("--text-threshold", type=float, default=0.6)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    ids = args.ids or sorted(p.stem for p in EXPECTED_DIR.glob("*.json"))
    rules = load_rules()
    docs = []
    for i in ids:
        exp = json.loads((EXPECTED_DIR / f"{i}.json").read_text())
        p = args.outputs / f"{i}.json"
        try:
            out = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}
        except json.JSONDecodeError:
            out = {}
        docs.append(score_document(exp, out, args.rtol, args.text_threshold, rules))
    res = {"rtol": args.rtol, "text_threshold": args.text_threshold, "total": summarize(docs), "documents": docs}
    if args.json:
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(res["total"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
