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

CARD_TOOLS = ("explain_spade", "search_docs", "describe_plant", "describe_tunable", "propose_experiments", "propose_tuning", "summarize_results", "check_requirements")
SHOWN_NOTE = ("This result is shown to the user verbatim, as a card with its source. Do not restate or paraphrase it, "
              "do not repeat its values: add at most one sentence.")
PLANT_SOURCE = "configs/plant_catalog.yaml (driveflow.datagen.plant_catalog)"
MAX_SEARCH_CARDS = 3
SNIPPET_CHARS = 450


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
        # At most MAX_SEARCH_CARDS: the glossary's own rows come first (search_docs puts them first).
        cards = []
        for h in result.get("hits", [])[:MAX_SEARCH_CARDS]:
            cards.append({"tool": name, "title": h.get("section") or h["file"], "source": h["file"], "lines": [h.get("snippet", "")[:SNIPPET_CHARS]]})
        if not cards:
            cards.append({"tool": name, "title": f"No documentation found for: {result.get('query', '')}", "source": "docs/, README.md", "lines": []})
        return cards
    if name == "describe_tunable":
        lines = [f"{p['name']}: range {p.get('range')}, baseline {p.get('baseline')}" for p in result.get("parameters", [])]
        lines += [f"objective {k}: {v}" for k, v in (result.get("objectives") or {}).items()]
        lines += [f"algorithms: {_fmt(result.get('algorithms', []))}"]
        return [{"tool": name, "title": f"Tunable parameters ({result.get('system', '')})", "source": "driveflow.assistant.objective.describe_tunable", "lines": lines}]
    if name == "propose_experiments":
        lines = [", ".join(f"{k} {v}" for k, v in e.items() if k != "scenario_id") for e in result.get("experiments", [])]
        cost = result.get("cost") or {}
        if cost:
            lines.append(f"estimated CPU: {cost.get('est_cpu_s')} s ({cost.get('basis', '')})")
        lines += [f"assumption: {a}" for a in result.get("assumptions", [])]
        return [{"tool": name, "title": f"Proposal: {len(result.get('experiments', []))} experiment(s) on {result.get('plant')} ({result.get('controller_type')})",
                 "source": "propose_experiments (draft)", "lines": lines}]
    if name == "propose_tuning":
        # Demo fixes (2026-11-08): the requirements and their values are shown HERE, one line each,
        # not by the model's text (answer.strip_draft_numbers removes them from it).
        ev = result.get("evaluations")
        ev_txt = f"{ev.get('value')} ({ev.get('origin')})" if isinstance(ev, dict) else _fmt(ev)
        motor = result.get("base_motor")
        lines = [f"plant: {result.get('plant')}", f"base motor: {motor if isinstance(motor, str) else 'custom (' + ', '.join(sorted(motor)) + ')'}",
                 f"tunes: {result.get('tunes')}", f"objective: {result.get('objective')}", f"algorithm: {result.get('algorithm')}",
                 f"evaluations: {ev_txt}", f"seed: {result.get('seed')}"]
        lines += [f"requirement: {r.get('metric')} {r.get('op')} {r.get('value')}{' ' + r['unit'] if r.get('unit') else ''}" for r in result.get("requirements", [])]
        lines += [f"verification: {result.get('verification')}"] if result.get("verification") else []
        return [{"tool": name, "title": "Tuning proposal", "source": "propose_tuning (draft)", "lines": lines}]
    if name == "summarize_results":
        def flat(kpis: dict, prefix: str = "") -> list:
            out = []
            for k, v in (kpis or {}).items():
                if isinstance(v, dict):
                    out += flat(v, f"{prefix}{k}.")
                elif not isinstance(v, (list, bool)) and k != "units":
                    out.append(f"{prefix}{k} {v}")
            return out

        lines = [f"{r.get('condition', {}).get('scenario_id', r.get('index'))}: {r.get('status')}; " + ", ".join(flat(r.get("kpis")))
                 for r in result.get("rows", [])]
        return [{"tool": name, "title": f"Results ({result.get('origin', 'SIMULATED')}), run {result.get('run_id')}", "source": "summarize_results (run store)", "lines": lines}]
    if name == "check_requirements":
        lines = [f"{k}: {_fmt(v)}" for k, v in result.items() if k not in ("note",)]
        return [{"tool": name, "title": "Requirements check (SIMULATED)", "source": "check_requirements (run store)", "lines": lines[:12]}]
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
