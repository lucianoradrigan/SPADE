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

CARD_TOOLS = ("explain_spade", "search_docs", "describe_plant", "describe_tunable", "propose_experiments", "propose_tuning", "summarize_results", "check_requirements",
              "suggest_value", "get_job_status", "run_experiments", "run_tuning")
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
        if c:
            c["lines"] += warning_lines(result)
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
    if name == "get_job_status":
        # F1.12 fix: a job's state is shown only here (the model's sentences about it are removed)
        p = result.get("progress") or {}
        lines = [f"state: {result.get('state')}", f"progress: {p.get('done')}/{p.get('total')}" + (f" ({p.get('phase')})" if p.get("phase") else "")]
        if result.get("run_id"):
            lines.append(f"run: {result['run_id']} ({result.get('run_status')})")
        if result.get("error"):
            lines.append(f"error: {result['error']}")
        lines.append("a SPADE worker must process queued jobs (Process queued jobs button, /worker, or python -m driveflow.jobs.worker)")
        return [{"tool": name, "title": f"Job {result.get('job_id')}", "source": "get_job_status (job service)", "lines": lines}]
    if name in ("run_experiments", "run_tuning"):
        return [{"tool": name, "title": f"Job {result.get('job_id')} submitted", "source": f"{name} (job service)",
                 "lines": ["state: queued (waiting for a SPADE worker)", f"experiment: {result.get('experiment_id')}"]}]
    if name == "suggest_value":
        # AI-N30 3(b): a suggestion and its source are shown as a card (the model does not restate them)
        if result.get("value") is None:
            return [{"tool": name, "title": f"No suggestion for {result.get('field')}", "source": "suggest_value (internal sources only, P5)",
                     "lines": [str(result.get("note") or "SPADE has no internal source for this value; it must come from you.")]}]
        rule = result.get("rule") or result.get("formula")
        return [{"tool": name, "title": f"Suggestion for {result.get('field')}", "source": str(result.get("source", "suggest_value")),
                 "lines": [f"{result.get('value')}{' ' + result['unit'] if result.get('unit') else ''}" + (f" ({rule})" if rule else "")]}]
    if name == "describe_tunable":
        lines = [f"{p['name']}: range {p.get('range')}, baseline {p.get('baseline')}" for p in result.get("parameters", [])]
        lines += [f"objective {k}: {v}" for k, v in (result.get("objectives") or {}).items()]
        lines += [f"algorithms: {_fmt(result.get('algorithms', []))}"]
        return [{"tool": name, "title": f"Tunable parameters ({result.get('system', '')})", "source": "driveflow.assistant.objective.describe_tunable", "lines": lines}]
    if name == "propose_experiments":
        lines = [", ".join(f"{k} {v}" for k, v in e.items() if k != "scenario_id") or "no field set: every value is the Scenario default (see the draft panel)"
                 for e in result.get("experiments", [])]
        cost = result.get("cost") or {}
        if cost:
            lines.append(f"estimated CPU: {cost.get('est_cpu_s')} s ({cost.get('basis', '')})")
        lines += [f"assumption: {a}" for a in result.get("assumptions", [])]
        lines += warning_lines(result)
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
        lines += warning_lines(result)
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


def warning_lines(result: dict) -> list:
    """S21-3: the proposal's non-blocking warnings, one line each, with what they apply to."""
    out = []
    for w in result.get("warnings") or []:
        if not isinstance(w, dict):
            continue
        ex = w.get("experiments")
        where = (f"experiments {', '.join(str(i) for i in ex)}" if isinstance(ex, list) else str(ex)) if ex is not None else ""
        out.append(f"WARNING {w.get('id', '')}: {w.get('message', '')} - {w.get('applies_to', '')}" + (f"; {where}" if where else "")
                   + (f" (not blocking; see {w['evidence']})" if w.get("evidence") else ""))
    return out


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
