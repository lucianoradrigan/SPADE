"""R4 (2026-11-15): the Phase 2 scorer (experiments/score_golden_extraction.py), tested without any LLM or extractor.
The expected scored against itself is perfect; each controlled perturbation lowers the right metric and only that one;
an empty or broken output does not crash."""

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import score_golden_extraction as S  # noqa: E402

EXPECTED = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(S.EXPECTED_DIR.glob("*.json"))}
METRICS = ("values", "inventions", "must_ask", "unneeded_questions", "must_warn", "simulable")


def _summary(exp, out):
    return S.summarize([S.score_document(exp, out)])


def _numeric_field(exp, unit=True):
    for i, f in enumerate(exp["fields"]):
        v = S.parse_value(f.get("value_printed"))
        if f.get("value_printed") not in S.NOT_GIVEN and isinstance(v, float) and v != 0 and (f.get("unit_printed") or not unit):
            return i, f
    return None, None


@pytest.mark.parametrize("doc", sorted(EXPECTED))
def test_the_expected_against_itself_is_perfect(doc):
    exp = EXPECTED[doc]
    s = _summary(exp, S.perfect_output(exp))
    assert s["values"] in (1.0, None) and s["inventions"] == 0 and s["unneeded_questions"] == 0
    assert s["must_ask"] in (1.0, None) and s["must_warn"] in (1.0, None) and s["simulable"] == 1.0


def _changed(base, pert):
    return {k for k in METRICS if base[k] != pert[k]}


def test_a_changed_unit_lowers_only_values():
    exp = EXPECTED["P1"]
    i, f = _numeric_field(exp)
    out = S.perfect_output(exp)
    j = next(k for k, g in enumerate(out["fields"]) if S._key(g) == S._key(f))
    out["fields"][j]["unit"] = "mNm" if f["unit_printed"] != "mNm" else "Nm"
    base, pert = _summary(exp, S.perfect_output(exp)), _summary(exp, out)
    assert _changed(base, pert) == {"values"} and pert["values"] < 1.0
    assert S.score_document(exp, out)["value_errors"][0]["why"] == "unit"


def test_an_undeclared_sqrt2_lowers_only_values_and_a_declared_conversion_is_accepted():
    exp = EXPECTED["P1"]
    i, f = _numeric_field(exp)
    out = S.perfect_output(exp)
    j = next(k for k, g in enumerate(out["fields"]) if S._key(g) == S._key(f))
    out["fields"][j]["value"] = S.parse_value(f["value_printed"]) * math.sqrt(2)
    base, pert = _summary(exp, S.perfect_output(exp)), _summary(exp, out)
    assert _changed(base, pert) == {"values"}
    declared = copy.deepcopy(out)
    declared["fields"][j]["value"] = S.parse_value(f["value_printed"]) / math.sqrt(2)
    declared["fields"][j]["converted_by"] = "ke_peak_to_kb_rms"  # factor 1/sqrt(2), declared
    assert _summary(exp, declared)["values"] == 1.0
    wrong_rule = copy.deepcopy(declared)
    wrong_rule["fields"][j]["converted_by"] = "phase_to_line_voltage_constant"  # sqrt(3): does not reproduce the value
    assert _summary(exp, wrong_rule)["values"] < 1.0


def test_a_basis_or_waveform_claimed_differently_is_wrong():
    exp = {"id": "X", "simulable_today": True, "must_ask": [], "must_warn": [],
           "fields": [{"field": "ke", "value_printed": "89.0", "unit_printed": "V/krpm", "basis": "line-line", "waveform": "rms", "variant": None},
                      {"field": "kt", "value_printed": "1.4", "unit_printed": "Nm/A", "basis": "unknown", "waveform": "unknown", "variant": None}]}
    good = S.perfect_output(exp)
    assert _summary(exp, good)["values"] == 1.0
    bad = copy.deepcopy(good)
    bad["fields"][0]["basis"] = "phase"
    bad["fields"][1]["waveform"] = "peak"  # claimed where the document does not say
    r = S.score_document(exp, bad)
    assert r["values_correct"] == 0 and {e["why"].split(" ")[0] for e in r["value_errors"]} == {"basis", "waveform"}


def test_an_invented_value_counts_only_as_invention():
    exp = next(e for e in EXPECTED.values() if any(f.get("value_printed") == "illegible" for f in e["fields"]))
    f = next(f for f in exp["fields"] if f.get("value_printed") == "illegible")
    out = S.perfect_output(exp)
    out["fields"].append({"field": f.get("field"), "label_printed": f.get("label_printed"), "variant": f.get("variant"), "value": "123", "unit": f.get("unit_printed")})
    base, pert = _summary(exp, S.perfect_output(exp)), _summary(exp, out)
    assert _changed(base, pert) == {"inventions"} and pert["inventions"] == 1


def test_ambiguous_must_ask_values_are_inventions_if_given():
    exp = next((e for e in EXPECTED.values() if any(f.get("value_printed") == "ambiguous: must ask" for f in e["fields"])), None)
    if exp is None:
        pytest.skip("no 'ambiguous: must ask' field in the golden set")
    f = next(f for f in exp["fields"] if f.get("value_printed") == "ambiguous: must ask")
    out = S.perfect_output(exp)
    out["fields"].append({"field": f.get("field"), "label_printed": f.get("label_printed"), "variant": f.get("variant"), "value": 1.0})
    assert _summary(exp, out)["inventions"] == 1


def test_an_omitted_question_lowers_only_questions():
    exp = next(e for e in EXPECTED.values() if e.get("must_ask"))
    out = S.perfect_output(exp)
    out["asked"] = out["asked"][1:]
    base, pert = _summary(exp, S.perfect_output(exp)), _summary(exp, out)
    assert _changed(base, pert) == {"must_ask"} and pert["must_ask"] < 1.0


def test_asking_for_a_printed_value_is_counted_as_unneeded():
    exp = EXPECTED["P1"]
    i, f = _numeric_field(exp)
    out = S.perfect_output(exp)
    out["asked"].append(f"What is the {f.get('label_printed')}?")
    pert = _summary(exp, out)
    assert pert["unneeded_questions"] == 1 and _changed(_summary(exp, S.perfect_output(exp)), pert) == {"unneeded_questions"}


def test_a_wrong_simulable_conclusion_lowers_only_simulable():
    exp = EXPECTED["P1"]
    out = S.perfect_output(exp)
    out["simulable_today"] = not exp["simulable_today"]
    assert _changed(_summary(exp, S.perfect_output(exp)), _summary(exp, out)) == {"simulable"}


@pytest.mark.parametrize("out", [{}, None, {"fields": None, "asked": None}, {"fields": ["not a dict"]}])
def test_an_empty_or_broken_output_does_not_crash(out):
    s = _summary(EXPECTED["P1"], out)
    assert s["inventions"] == 0 and s["simulable"] == 0.0 and (s["values"] or 0.0) == 0.0


def test_the_cli_on_an_empty_file_and_a_missing_file(tmp_path):
    (tmp_path / "P1.json").write_text("", encoding="utf-8")
    out = tmp_path / "res.json"
    S.main(["--outputs", str(tmp_path), "--ids", "P1", "W01", "--json", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["total"]["documents"] == 2 and r["total"]["inventions"] == 0


def test_numbers_are_parsed_as_printed():
    assert S.parse_value("3,000") == 3000.0 and S.parse_value("0,8") == 0.8 and S.parse_value("230/400") == (230.0, 400.0)
    assert S.parse_value("MS 711-2") == "ms 711-2" and S.parse_value(None) is None
