"""F1.7: the assistant's catalog covers SPADE and stays true (plan §6c)."""

import importlib
import re
from pathlib import Path

import pytest

from driveflow.assistant import catalog as C
from driveflow.assistant.core.policy import SessionState

ROOT = Path(__file__).resolve().parents[2]


def entries():
    return C.load_catalog()["entries"]


def test_every_page_plant_controller_and_job_has_an_entry():
    from driveflow.datagen.plant_catalog import NATIVE_PLANTS
    from driveflow.datagen.scenario import Scenario
    from driveflow.jobs.handlers import HANDLERS
    from driveflow.viz.app import PAGES

    e = entries().values()
    pages = {x["page"] for x in e if x["kind"] == "page"}
    assert pages == {p for items in PAGES.values() for p, _t, _i in items}
    assert {x["plant_config_id"] for x in e if x["kind"] == "plant"} == set(NATIVE_PLANTS)
    assert {x["controller_type"] for x in e if x["kind"] == "controller"} == {c for c, _p in Scenario._VALID_PAIRS}
    assert {x["job_kind"] for x in e if x["kind"] == "job"} == set(HANDLERS)


def test_every_algorithm_family_is_referenced():
    referenced = {g for x in entries().values() for g in x.get("algorithm_guide", [])}
    assert set(C.load_guide()) <= referenced


@pytest.mark.parametrize("key", sorted(C.load_catalog()["entries"]))
def test_entry_is_true(key):
    e = entries()[key]
    for mod in e.get("implemented_by", []):
        importlib.import_module(mod)
    for doc in e.get("docs", []):
        assert (ROOT / doc).exists(), doc
    if e.get("page"):
        assert (ROOT / "src" / "driveflow" / "viz" / e["page"]).exists(), e["page"]
    for g in e.get("algorithm_guide", []):
        assert g in C.load_guide(), g
    text = " ".join(str(v) for v in e.values())
    sim = (ROOT / "docs" / "SIMULATION_STABILITY_REPORT.md").read_text()
    ai = (ROOT / "docs" / "AI_TECHNICAL_AUDIT.md").read_text()
    for fid in re.findall(r"\b(S-\d+|AI-N\d+)\b", text):
        source = sim if fid.startswith("S-") else ai
        assert re.search(rf"\|\s*{re.escape(fid)}\s*\|", source), f"{key} cites {fid}, which is not a finding"


@pytest.mark.parametrize("topic, expected", [("Tuning Studio", "tuning_studio"), ("tuning_studio", "tuning_studio"), ("MPC", "controller_MPC"),
                                             ("dc_pmg132_v1", "plant_dc_pmg132_v1"), ("controller_tuning", "job_controller_tuning"), ("Model Registry", "model_registry")])
def test_explain_spade_finds_the_entry(topic, expected):
    r = C.explain_spade(SessionState(), topic)
    assert r["documented"] and r["entries"][0]["id"] == expected


def test_explain_spade_returns_the_guide_entries_and_says_when_it_does_not_know():
    r = C.explain_spade(SessionState(), "Tuning Studio")
    assert r["entries"][0]["algorithm_guide_entries"][0]["id"] == "controller_tuning"
    r = C.explain_spade(SessionState(), "quantum annealing")
    assert r["documented"] is False and "not documented" in r["note"]
    r = C.explain_spade(SessionState(), "induction motor")
    assert r["documented"] is False and r["not_simulable"][0]["topic"].startswith("induction")


def test_search_docs_cites_file_and_section():
    r = C.search_docs(SessionState(), "one-quadrant converter brake")
    assert r["hits"] and all({"file", "section", "snippet"} <= set(h) for h in r["hits"])
    assert any("SIMULATION_STABILITY_REPORT" in h["file"] for h in r["hits"])
