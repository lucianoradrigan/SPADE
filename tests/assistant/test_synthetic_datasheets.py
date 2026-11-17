"""B6 (2026-11-16): the synthetic datasheet generator. No LLM and no extractor is run here (or anywhere in B6)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gen_syn", ROOT / "experiments" / "generate_synthetic_datasheets.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
sys.path.insert(0, str(ROOT / "experiments"))
import score_golden_extraction as scorer  # noqa: E402


def test_same_seed_same_sheets(tmp_path):
    a, b = gen.generate(tmp_path / "a", n=12), gen.generate(tmp_path / "b", n=12)
    assert [x["html_sha256"] for x in a] == [x["html_sha256"] for x in b]
    assert [x["html_sha256"] for x in gen.generate(tmp_path / "c", seed=gen.SEED + 1, n=12)] != [x["html_sha256"] for x in a]


def test_every_style_and_the_closed_reason_list(tmp_path):
    log = gen.generate(tmp_path, n=gen.N_SHEETS)
    assert {x["style"] for x in log} == set(gen.STYLES)
    assert gen.REASONS == scorer.REASONS
    for p in (tmp_path / "expected").glob("*.json"):
        e = json.loads(p.read_text(encoding="utf-8"))
        assert e["schema"] == "golden-expected/1" and e["synthetic"] is True
        assert all(set(a) >= {"field", "reason", "item", "text"} and a["reason"] in scorer.REASONS for a in e["must_ask"])
        assert [a["item"] for a in e["must_ask"]] == list(range(len(e["must_ask"])))
        r = scorer.score_document(e, scorer.perfect_output(e))  # the scorer reads these files: a perfect output scores perfectly
        assert r["values_correct"] == r["values_total"] and r["must_ask_hit"] == r["must_ask_total"] and not r["inventions"]


def test_every_logged_trap_is_visible_in_the_expected_file(tmp_path):
    gen.generate(tmp_path, n=gen.N_SHEETS)
    for p in (tmp_path / "expected").glob("*.json"):
        e = json.loads(p.read_text(encoding="utf-8"))
        reasons = {a["reason"] for a in e["must_ask"]}
        for t in (x["trap"] for x in e["traps"]):
            visible = {"kt_ke_inconsistent": "inconsistent" in reasons, "wrong_label": "ambiguous_meaning" in reasons,
                       "unitless_inertia": "ambiguous_meaning" in reasons, "missing_inductance": "missing" in reasons,
                       "manufacturer_unnamed": "manufacturer_unnamed" in reasons, "multiple_ratings": "variant_unspecified" in reasons,
                       "ke_waveform_undeclared": "ambiguous_waveform" in reasons}.get(t, bool(e["must_warn"]) or bool(e["must_ask"]))
            assert visible, (e["id"], t)


def test_printed_values_follow_the_truth(tmp_path):
    gen.generate(tmp_path, n=gen.N_SHEETS)
    for p in (tmp_path / "expected").glob("*.json"):
        e = json.loads(p.read_text(encoding="utf-8"))
        t, traps = e["truth_si"], {x["trap"] for x in e["traps"]}
        r = next(f for f in e["fields"] if f["field"] == "resistance")
        printed = float(r["value_printed"].replace(",", "."))
        expected_r = t["r_ohm"] * (2 if "line_line_resistance" in traps else 1)
        assert math.isclose(printed, expected_r, rel_tol=0.01), (e["id"], printed, expected_r)
        assert r["basis"] == ("line-line" if "line_line_resistance" in traps else ("not_applicable" if t["machine"] == "dc" else "phase"))
        if t["machine"] == "pmsm":
            assert math.isclose(t["kt_nm_per_a"], 1.5 * t["pole_pairs"] * t["psi_v_s"], rel_tol=1e-3)


def test_sheets_say_they_are_synthetic_and_name_no_real_maker(tmp_path):
    gen.generate(tmp_path, n=gen.N_SHEETS)
    for h in tmp_path.glob("S*.html"):
        text = h.read_text(encoding="utf-8")
        assert "SYNTHETIC SHEET" in text and "Not a real product" in text
        assert not any(m in text for m in ("Siemens", "maxon", "WEG", "Baldor", "Yaskawa", "EMRAX", "Bodine", "Parvalux", "Motenergy"))
