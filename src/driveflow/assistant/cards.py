"""SPADE data shown verbatim (F1.1 round 3, decided by Luciano 2026-11-04).

Round 2 found that the false statements came from the model paraphrasing SPADE's data (plant
descriptions, nominal values, findings). The results of ``explain_spade``, ``search_docs`` and
``describe_plant`` are therefore shown to the user AS THEY ARE, as a card with its source, by the
interface (CLI and Assistant page); the model only adds one or two sentences. Plants named in the
answer get a card painted from the catalog (title, nominal values), never from the model's text.

``card_from_tool`` builds a card from a tool result; ``plant_card`` from the catalog;
``render_text`` prints cards for the CLI; the page renders the same dicts.
"""

from __future__ import annotations

import re

CARD_TOOLS = ("explain_spade", "search_docs", "describe_plant")
SHOWN_NOTE = ("This result is shown to the user verbatim, as a card with its source. Do not restate or paraphrase it: "
              "add at most two sentences and cite the source (a docs/ path or a finding ID).")
PLANT_SOURCE = "configs/plant_catalog.yaml (driveflow.datagen.plant_catalog)"


def _fmt(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "; ".join(_fmt(x) for x in v)
    return str(v)


def plant_card(plant: str) -> dict | None:
    from driveflow.datagen.plant_catalog import NATIVE_PLANTS, describe_plant

    if plant not in NATIVE_PLANTS:
        return None
    d = describe_plant(plant)
    lines = [d["title"], f"Controllers: {_fmt(d.get('controllers', []))}"]
    sm = d.get("stock_motor")
    if sm:
        lines += [f"Nominal values: {_fmt(sm['nominal_values'])}", f"Limits: {_fmt(sm['limits'])}", f"Supply: {sm['supply_v']} V"]
    return {"tool": "catalog", "title": f"Plant {plant}", "source": PLANT_SOURCE, "lines": lines}


def card_from_tool(name: str, arguments: dict, result: dict) -> list:
    if not isinstance(result, dict) or result.get("ok") is False or "error" in result:
        return []
    if name == "describe_plant":
        c = plant_card(str(result.get("plant") or (arguments or {}).get("plant", "")))
        return [c] if c else []
    if name == "explain_spade":
        if not result.get("documented"):
            return [{"tool": name, "title": f"Not in SPADE's catalog: {(arguments or {}).get('topic', '')}", "source": "configs/assistant_catalog.yaml",
                     "lines": [result.get("note", "")]}]
        cards = []
        for e in result.get("entries", []):
            lines = [f"{k}: {_fmt(e[k])}" for k in ("what", "when", "page", "limitations", "family", "why", "assumptions", "docs") if e.get(k)]
            cards.append({"tool": name, "title": str(e.get("title") or e.get("label") or e.get("id")), "source": result.get("source", "configs/assistant_catalog.yaml"),
                          "lines": lines})
        return cards
    if name == "search_docs":
        cards = [{"tool": name, "title": f"{g['id']} — {g['title']}", "source": g["file"], "lines": ["(glossary match)"]} for g in result.get("glossary", [])]
        for h in result.get("hits", []):
            cards.append({"tool": name, "title": h.get("section") or h["file"], "source": h["file"], "lines": [h.get("snippet", "")]})
        if not cards:
            cards.append({"tool": name, "title": f"No documentation found for: {result.get('query', '')}", "source": "docs/, README.md", "lines": []})
        return cards
    return []


def plants_named(text: str) -> list:
    from driveflow.datagen.plant_catalog import NATIVE_PLANTS

    return [p for p in NATIVE_PLANTS if re.search(rf"\b{re.escape(p)}\b", text or "")]


def render_text(cards: list) -> str:
    out = []
    for c in cards:
        out.append(f"┌ {c['title']}  [source: {c['source']}]")
        out += [f"│ {ln}" for ln in c["lines"] if ln]
        out.append("└")
    return "\n".join(out)
