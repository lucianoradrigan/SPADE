"""AI-N41 (corrected 2026-11-16, decision D7): a finding named by its ID never drops out of the assistant's glossary, even
when every word of its heading is frequent among the findings. The case that happened: the first wording of the AI-N40
heading ("the dc_motor classifiers are confidently wrong above their training grid") made all of AI-N29's heading words
frequent, AI-N29 lost its last term and `search_docs('AI-N29 ...')` returned nothing."""

from driveflow.assistant import glossary
from driveflow.assistant.catalog import search_docs
from driveflow.assistant.core.loop import SessionState

AI_N40_FIRST_HEADING = "### 13.13 AI-N40 — the dc_motor classifiers are confidently wrong above their training grid · **High** · **Mitigated**\n"


def _with_extra_heading(monkeypatch, extra: str):
    real = glossary._read

    def fake(path):
        text = real(path)
        return text + "\n" + extra if path == "docs/AI_TECHNICAL_AUDIT.md" else text

    monkeypatch.setattr(glossary, "_read", fake)
    for f in (glossary.build_glossary, getattr(glossary, "_glossary_cached", None)):
        if f is not None and hasattr(f, "cache_clear"):
            f.cache_clear()


def test_the_finding_survives_a_heading_that_makes_its_words_frequent(monkeypatch):
    _with_extra_heading(monkeypatch, AI_N40_FIRST_HEADING)
    ids = {e["id"] for e in glossary.build_glossary()}
    assert "AI-N29" in ids
    assert any(g["id"] == "AI-N29" for g in glossary.lookup("AI-N29 clasificador PC"))


def test_search_docs_finds_the_named_finding_in_that_case(monkeypatch):
    _with_extra_heading(monkeypatch, AI_N40_FIRST_HEADING)
    hits = search_docs(SessionState(), "AI-N29 clasificador PC")["hits"]
    assert hits and hits[0]["file"] == "docs/AI_TECHNICAL_AUDIT.md"


def test_every_finding_heading_of_the_audit_is_in_the_glossary():
    import re

    heads = set(re.findall(r"^#{2,4} (?:[\d.]+ )?(AI-N\d+) — ", glossary._read("docs/AI_TECHNICAL_AUDIT.md"), re.M))
    assert heads and heads <= {e["id"] for e in glossary.build_glossary()}
