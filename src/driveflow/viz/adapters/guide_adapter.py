"""Algorithm Guide data (expansion task E12; spec Sec. 8.6): the recommendation table in
configs/algorithm_guide.yaml, validated on load against the pages the app actually registers."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path

import yaml

GUIDE_PATH = Path(__file__).resolve().parents[4] / "configs" / "algorithm_guide.yaml"
NOT_AVAILABLE = "Not available in this version"
_FIELDS = ("id", "phase", "label", "family", "why", "assumptions", "limitations", "page")


@dataclasses.dataclass(frozen=True)
class Recommendation:
    task_id: str
    task: str
    family: str
    why: str
    assumptions: tuple
    limitations: tuple
    page: str | None
    page_title: str | None
    unavailable_reason: str | None


def registered_pages() -> dict:
    """path -> title of every page in viz/app.py's navigation."""
    from driveflow.viz.app import PAGES

    return {path: title for items in PAGES.values() for path, title, _ in items}


def validate(guide: dict, pages: dict) -> list[str]:
    """Problems with the table (empty = valid): missing fields, unknown phases/sources, pages that
    are not registered, a task with no page that still lists sources."""
    problems = []
    phases, sources = set(guide.get("phases", {})), set(guide.get("data_sources", {}))
    for key in ("authors", "title", "venue", "year"):
        if not (guide.get("citation") or {}).get(key):
            problems.append(f"citation.{key} is missing")
    if set(guide.get("data_source_notes", {})) != sources:
        problems.append("data_source_notes must have one entry per data source")
    seen = set()
    for t in guide.get("tasks", []):
        tid = t.get("id", "?")
        problems += [f"{tid}: missing '{f}'" for f in _FIELDS if f not in t]
        if tid in seen:
            problems.append(f"{tid}: duplicate id")
        seen.add(tid)
        if t.get("phase") not in phases:
            problems.append(f"{tid}: unknown phase {t.get('phase')!r}")
        if t.get("page") is not None and t["page"] not in pages:
            problems.append(f"{tid}: page {t['page']!r} is not registered in viz/app.py")
        if t.get("page") is not None and not set(t.get("accepts", [])) <= sources:
            problems.append(f"{tid}: accepts unknown data source(s)")
        if t.get("page") is not None and not t.get("accepts"):
            problems.append(f"{tid}: a task with a page must list the data sources it accepts")
    problems += [f"phase {p!r} has no task" for p in phases if not any(t.get("phase") == p for t in guide.get("tasks", []))]
    return problems


@lru_cache(maxsize=1)
def load_guide(path: Path = GUIDE_PATH) -> dict:
    guide = yaml.safe_load(Path(path).read_text())
    problems = validate(guide, registered_pages())
    if problems:
        raise ValueError("configs/algorithm_guide.yaml is invalid: " + "; ".join(problems))
    return guide


def tasks_for(phase: str, guide: dict | None = None) -> list[tuple[str, str]]:
    g = guide or load_guide()
    return [(t["id"], t["label"]) for t in g["tasks"] if t["phase"] == phase]


def recommend(phase: str, task_id: str, data_source: str, guide: dict | None = None) -> Recommendation:
    g = guide or load_guide()
    if data_source not in g["data_sources"]:
        raise KeyError(f"unknown data source {data_source!r}")
    t = next((t for t in g["tasks"] if t["id"] == task_id and t["phase"] == phase), None)
    if t is None:
        raise KeyError(f"no task {task_id!r} in phase {phase!r}")
    notes = g["data_source_notes"][data_source]
    page, reason = t["page"], None
    if page is None:
        reason = NOT_AVAILABLE
    elif data_source not in t["accepts"]:
        page, reason = None, f"{NOT_AVAILABLE} for {g['data_sources'][data_source].lower()}"
    return Recommendation(task_id=t["id"], task=t["label"], family=t["family"], why=t["why"], assumptions=(t["assumptions"], notes["assumptions"]),
                          limitations=(t["limitations"], notes["limitations"]), page=page, page_title=registered_pages().get(page) if page else None,
                          unavailable_reason=reason)


def citation_text(guide: dict | None = None) -> str:
    c = (guide or load_guide())["citation"]
    return f"{c['authors']}, \"{c['title']},\" {c['venue']}, {c['year']}."
