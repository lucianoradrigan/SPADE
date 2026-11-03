"""Questions about SPADE (task F1.7, capability C8): answers come ONLY from the repository.

- ``explain_spade(topic)``: the entry of configs/assistant_catalog.yaml that matches the topic (a page,
  plant, controller, job kind or algorithm family), with the algorithm-guide entries it references.
  Nothing matching -> "not documented" (the assistant must say so instead of answering from memory).
- ``search_docs(query)``: plain-text search over docs/ and the README; returns snippets with their
  file and section, for "why" questions (findings, decisions).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

from driveflow.assistant.core.loop import Tool, ToolFailure
from driveflow.assistant.core.policy import SessionState

REPO = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO / "configs" / "assistant_catalog.yaml"
GUIDE_PATH = REPO / "configs" / "algorithm_guide.yaml"
DOC_ROOTS = ("docs", "README.md")
MAX_HITS = 5


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
        files = [p] if p.is_file() else sorted(p.rglob("*.md"))
        for f in files:
            section = ""
            for para in re.split(r"\n\s*\n", f.read_text(errors="replace")):
                head = next((ln for ln in para.splitlines() if ln.startswith("#")), None)
                if head:
                    section = head.lstrip("# ").strip()
                out.append((str(f.relative_to(REPO)), section, para.strip()))
    return out


def search_docs(state: SessionState, query: str) -> dict:
    terms = [w for w in _norm(query).split() if len(w) > 2]
    if not terms:
        raise ToolFailure("give two or more words to search for")
    scored = []
    for path, section, para in _doc_paragraphs():
        low = _norm(para)
        score = sum(low.count(w) for w in terms) + 5 * sum(1 for w in terms if re.search(rf"\b{re.escape(w)}\b", low)) * (1 if all(w in low for w in terms) else 0)
        if score:
            scored.append((score, path, section, para))
    scored.sort(key=lambda x: -x[0])
    hits = [{"file": p, "section": s, "snippet": para[:600]} for _sc, p, s, para in scored[:MAX_HITS]]
    return {"query": query, "hits": hits, "note": "cite file and section; if nothing relevant is here, say SPADE does not document it"}


def _schema(props: dict, required: list) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


CATALOG_TOOLS = [
    Tool("explain_spade", "What a SPADE page, plant, controller, job or algorithm family does, its assumptions and limits, from SPADE's own catalog.",
         _schema({"topic": {"type": "string", "description": "e.g. 'Tuning Studio', 'MPC', 'dc_pmg132_v1', 'controller_tuning'"}}, ["topic"]), explain_spade),
    Tool("search_docs", "Search SPADE's documentation (docs/, README) for a question such as why something is limited; returns snippets with file and section.",
         _schema({"query": {"type": "string"}}, ["query"]), search_docs),
]
