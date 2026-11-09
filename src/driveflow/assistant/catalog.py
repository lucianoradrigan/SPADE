"""Questions about SPADE (task F1.7, capability C8): answers come ONLY from the repository.

- ``explain_spade(topic)``: the entry of configs/assistant_catalog.yaml that matches the topic (a page,
  plant, controller, job kind or algorithm family), with the algorithm-guide entries it references.
  Nothing matching -> "not documented" (the assistant must say so instead of answering from memory).
- ``search_docs(query)``: plain-text search over docs/ and the README; returns snippets with their
  file and section, for "why" questions (findings, decisions).
"""

from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

from driveflow.assistant.glossary import ID_RE
from driveflow.assistant.core.loop import Tool, ToolFailure
from driveflow.assistant.core.policy import SessionState

REPO = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO / "configs" / "assistant_catalog.yaml"
GUIDE_PATH = REPO / "configs" / "algorithm_guide.yaml"
DOC_ROOTS = ("docs", "README.md")
MAX_HITS = 5
#: The assistant's own benchmark reports quote the model's wrong answers verbatim; searching them would
#: feed those errors back (F1.1 round 3).
#: AI-N30 (2026-11-08): the harness-fix evidence quotes F1.11's requests too.
#: docs/archive/: superseded documents (decision C4, 2026-11-09) must not be cited as SPADE's current state.
EXCLUDED_DOCS = re.compile(r"docs/audit/assistant/(f1_1|ai_n30_)|docs/archive/")


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text())


@lru_cache(maxsize=1)
def load_guide() -> dict:
    return {t["id"]: t for t in yaml.safe_load(GUIDE_PATH.read_text())["tasks"]}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def _names(key: str, e: dict) -> set:
    names = {key, e.get("title", ""), e.get("plant_config_id", ""), e.get("controller_type", ""), e.get("job_kind", "")}
    if e.get("page"):
        names.add(Path(e["page"]).stem)
    return {_norm(n) for n in names if n}


def explain_spade(state: SessionState, topic: str) -> dict:
    cat, guide = load_catalog(), load_guide()
    t = _norm(topic)
    if not t:
        raise ToolFailure("give a topic, e.g. 'Tuning Studio', 'MPC' or 'controller_tuning'")
    matches = [k for k, e in cat["entries"].items() if t in _names(k, e)]
    if not matches:
        matches = [k for k, e in cat["entries"].items() if any(t in n or n in t for n in _names(k, e) if len(n) > 2)]
    guide_hits = [g for g in guide if _norm(g) == t or t in _norm(guide[g]["label"])]
    blocked = [x for x in cat.get("not_simulable", []) if _norm(x["topic"]).split()[0] in t]
    if not matches and not guide_hits:
        out = {"documented": False, "note": "not documented in SPADE's catalog; say so, do not answer from general knowledge. search_docs may find a mention."}
        if blocked:
            out["not_simulable"] = blocked
        return out
    entries = []
    for k in matches[:3]:
        e = dict(cat["entries"][k])
        e["id"] = k
        e["algorithm_guide_entries"] = [{kk: guide[g][kk] for kk in ("id", "label", "family", "why", "assumptions", "limitations", "page")} for g in e.get("algorithm_guide", []) if g in guide]
        entries.append(e)
    for g in guide_hits:
        if not any(g in e.get("algorithm_guide", []) for e in entries):
            entries.append({"id": g, "kind": "algorithm", **{kk: guide[g][kk] for kk in ("label", "family", "why", "assumptions", "limitations", "page")}})
    return {"documented": True, "entries": entries, "source": "configs/assistant_catalog.yaml, configs/algorithm_guide.yaml",
            "note": "answer only from these entries and cite the page, file or finding ID"}


@lru_cache(maxsize=1)
def _doc_paragraphs() -> list:
    out = []
    for root in DOC_ROOTS:
        p = REPO / root
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*.md") if not EXCLUDED_DOCS.match(str(f.relative_to(REPO))))
        for f in files:
            section = ""
            for para in re.split(r"\n\s*\n", f.read_text(errors="replace")):
                head = next((ln for ln in para.splitlines() if ln.startswith("#")), None)
                if head:
                    section = head.lstrip("# ").strip()
                out.append((str(f.relative_to(REPO)), section, para.strip()))
    return out


#: AI-N30 fix (2026-11-08, designed by reading the F1.11 conversations -- declared): words that do not make a
#: query specific. Before this, any paragraph with ANY query word longer than two letters ("the", "que",
#: "spade", "motor") was a hit, so search_docs almost never came back empty and the "not documented after two
#: empty searches" rule could not be met (F1.11: 0 %).
GENERIC_WORDS = frozenset("""
the and for with what which does how why can could use used using this that these are was were has have from into about
its their there here when where who whom whose all any some more most such not but only also than then them they you your our
el la los las lo un una unos unas de del al que qué como cómo para por con sin sobre segun según dice dicen sirve es son esta este estos
hay tiene tienen cual cuál cuales donde dónde cuando cuándo porque qué entre desde hasta muy mas más ser fue hace hacer puede pueden
spade finding hallazgo option opcion opción model modelo page pagina página plant planta plants plantas controller controllers controlador
controladores type types tipo tipos allowed compatible available disponible support supports soporta documented documentado documentation
documentacion documentación
""".split())
#: share of a query's distinctive words a paragraph must contain (all of them when there are two or fewer)
COVERAGE = 0.75


def distinctive_terms(query: str) -> list:
    out = []
    for w in _norm(query).split():
        if len(w) > 2 and w not in GENERIC_WORDS and w not in out:
            out.append(w)
    return out


def _covers(text: str, terms: list) -> bool:
    if not terms:
        return False
    low = _norm(text)
    found = sum(1 for w in terms if re.search(rf"\b{re.escape(w)}\b", low))
    need = len(terms) if len(terms) <= 2 else math.ceil(COVERAGE * len(terms))
    return found >= need


def search_docs(state: SessionState, query: str) -> dict:
    terms = distinctive_terms(query)
    ids = set(ID_RE.findall(query or ""))
    if not terms and not ids:
        raise ToolFailure("give two or more specific words to search for")
    scored = []
    for path, section, para in _doc_paragraphs():
        if not _covers(para, terms):
            continue
        low = _norm(para)
        score = sum(low.count(w) for w in terms)
        scored.append((score, path, section, para))
    scored.sort(key=lambda x: -x[0])
    hits = [{"file": p, "section": s, "snippet": para[:600]} for _sc, p, s, para in scored[:MAX_HITS]]
    # F1.1 round 3: generated glossary of aliases (glossary.py). A finding or catalog entry whose
    # distinctive words are in the query is returned, and its own paragraph comes first.
    from driveflow.assistant import glossary

    matches = glossary.lookup(query)
    first = []
    for g in matches:
        named = g["id"] in ids  # a finding / entry named by its ID is always returned (if it exists)
        if g["file"].startswith("docs/"):
            in_file = [(pth, sec, txt) for pth, sec, txt in _doc_paragraphs() if pth == g["file"] and re.search(rf"\b{re.escape(g['id'])}\b", txt)]
            own = rf"(^|\n)(\| |#+ (?:[\d.]+ )?){re.escape(g['id'])}\b"  # the finding's own table row or heading
            para = next((x for x in in_file if re.search(own, x[2])), in_file[0] if in_file else None)
            if para and (named or _covers(para[2], terms)):
                m = re.search(own, para[2])
                start = m.start() + (1 if m and para[2][m.start()] == "\n" else 0) if m else 0
                h = {"file": para[0], "section": para[1], "snippet": para[2][start:start + 600]}
                if h not in first:
                    first.append(h)
    hits = (first + [h for h in hits if h not in first])[:MAX_HITS]
    out = {"query": query, "hits": hits, "note": "cite file and section (a docs/ path) or the finding ID; if nothing relevant is here, run one more, different query before saying SPADE does not document it"}
    if matches:
        out["glossary"] = [g for g in matches if g["id"] in ids or any(h["file"] == g["file"] for h in first)]
    out["distinctive_terms"] = terms
    return out


def _schema(props: dict, required: list) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


CATALOG_TOOLS = [
    Tool("explain_spade", "What a SPADE page, plant, controller, job or algorithm family does, its assumptions and limits, from SPADE's own catalog.",
         _schema({"topic": {"type": "string", "description": "e.g. 'Tuning Studio', 'MPC', 'dc_pmg132_v1', 'controller_tuning'"}}, ["topic"]), explain_spade),
    Tool("search_docs", "Search SPADE's documentation (docs/, README) for a question such as why something is limited; returns snippets with file and section.",
         _schema({"query": {"type": "string"}}, ["query"]), search_docs),
]
