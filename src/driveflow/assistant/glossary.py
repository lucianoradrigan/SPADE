"""Glossary of aliases for questions about SPADE, and the citation check (F1.1 round 3, decided by
Luciano 2026-11-04 after round 2 found false statements in every configuration).

The glossary is GENERATED, not written by hand: from the finding rows of docs/VALIDATION_SUMMARY.md
and docs/SIMULATION_STABILITY_REPORT.md, the finding headings of docs/AI_TECHNICAL_AUDIT.md and the
entries of configs/assistant_catalog.yaml. Each entry keeps the distinctive words of its title (words
of 5+ letters that occur in at most ``MAX_DF`` entries), so "abstention threshold" finds AI-N12 and
VALIDATION_SUMMARY although the question names neither.

- ``lookup(text)``: the entries whose distinctive words occur in the text (search_docs expands the
  query with them; the loop's look-up nudge also fires on them);
- ``verified_citation(text)``: the text cites a docs/ path that exists or a finding ID that exists.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
MAX_DF = 3
MIN_LEN = 5
ID_RE = re.compile(r"\b(AI-N\d+|AI-\d+|S-\d+)\b")
DOCS_PATH_RE = re.compile(r"\b(docs/[A-Za-z0-9_./-]+\.(?:md|json|txt))")
STOP = set("""about above after again against because before being below between could doing during every first found further having
into itself might model models other ought should since their there these those through under until using value values which while
would without where while para sobre entre desde hasta donde cuando porque tiene tienen puede pueden segun según mismo misma otros
otras stock shipped runs simulated simulation simulations round phase measured fixed open mitigated still second third first pass
default defaults change changed changes check checked""".split())


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def _words(s: str) -> set:
    return {w for w in _norm(s).split() if len(w) >= MIN_LEN and w not in STOP and not w.isdigit()}


def _read(rel: str) -> str:
    p = REPO / rel
    return p.read_text(errors="replace") if p.exists() else ""


@lru_cache(maxsize=1)
def build_glossary() -> list:
    raw = []
    for line in _read("docs/VALIDATION_SUMMARY.md").splitlines():
        m = re.match(r"^\| ((?:AI-N?\d+|S-\d+)) — ([^|]+)\|[^|]*\|([^|]*)\|", line)
        if m:
            raw.append({"id": m.group(1), "file": "docs/VALIDATION_SUMMARY.md", "title": m.group(2).strip(), "text": m.group(2) + " " + m.group(3)[:200]})
    for line in _read("docs/AI_TECHNICAL_AUDIT.md").splitlines():
        m = re.match(r"^#{2,4} (?:[\d.]+ )?(AI-N\d+) — (.+?)(?: ·|$)", line)
        if m:
            raw.append({"id": m.group(1), "file": "docs/AI_TECHNICAL_AUDIT.md", "title": m.group(2).strip(), "text": m.group(2)})
    for line in _read("docs/SIMULATION_STABILITY_REPORT.md").splitlines():
        m = re.match(r"^\| (S-\d+) \|[^|]*\|([^|]*)\|([^|]*)\|", line)
        if m:
            title = re.split(r"(?<=[.:])\s", m.group(3).strip(), maxsplit=1)[0][:160]
            raw.append({"id": m.group(1), "file": "docs/SIMULATION_STABILITY_REPORT.md", "title": title, "text": m.group(2) + " " + title})
    cat = yaml.safe_load(_read("configs/assistant_catalog.yaml") or "{}") or {}
    for key, e in (cat.get("entries") or {}).items():
        what = str(e.get("what", ""))
        raw.append({"id": key, "file": "configs/assistant_catalog.yaml", "title": str(e.get("title", key)), "text": f"{key} {e.get('title', '')} {what[:160]}"})
    df: dict = {}
    for r in raw:
        for w in _words(r["text"]):
            df[w] = df.get(w, 0) + 1
    out = []
    for r in raw:
        terms = sorted(w for w in _words(r["text"]) if df[w] <= MAX_DF)
        if terms:
            out.append({"id": r["id"], "file": r["file"], "title": r["title"], "terms": terms})
    return out


def lookup(text: str, limit: int = 3) -> list:
    words = _words(text)
    ids = set(ID_RE.findall(text or ""))
    scored = []
    for e in build_glossary():
        hit = [t for t in e["terms"] if t in words]
        if e["id"] in ids:
            scored.append((100, e["id"], e))  # a finding named by its ID comes first
        elif hit:
            scored.append((len(hit), e["id"], e))
    scored.sort(key=lambda x: (-x[0], x[1]))
    seen, out = set(), []
    for _n, _id, e in scored:
        if (e["id"], e["file"]) not in seen:
            seen.add((e["id"], e["file"]))
            out.append({"id": e["id"], "file": e["file"], "title": e["title"]})
        if len(out) >= limit:
            break
    return out


@lru_cache(maxsize=1)
def known_ids() -> frozenset:
    ids = set()
    for f in (REPO / "docs").glob("*.md"):
        ids |= set(ID_RE.findall(f.read_text(errors="replace")))
    return frozenset(ids)


def verified_citation(text: str) -> bool:
    """A docs/ path that exists in the repository, or a finding ID that exists in docs/."""
    for p in DOCS_PATH_RE.findall(text or ""):
        if (REPO / p.rstrip(".,;:)")).exists():
            return True
    return any(i in known_ids() for i in ID_RE.findall(text or ""))
