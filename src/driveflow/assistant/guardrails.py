"""Harness fixes after F1.11 (AI-N30; Luciano, 2026-11-08, item 3(b)). Designed by reading the F1.11
conversations, which is declared: F1.12 (new requests, the same thresholds) is the independent test.

Deterministic, model-independent rules applied to what the user reads (core/loop.py ``_postprocess``):

- ``missing_card``: when the draft has missing REQUIRED fields and nothing is proposed, the interface asks
  for them with a question card -- one line per field, with a suggestion only when ``suggest_value`` has an
  internal source (P5); it does not depend on the model asking.
- ``bound_capabilities``: sentences of the model's own text about what SPADE can or cannot do (simulate,
  tune, train, support, document) are not shown; the tools' own words about the limit (a refusal, a
  "not simulable" reason, a "not tunable" note) are shown instead, as a card. F1.11: four of the five false
  statements were such sentences.
- ``bound_launch_claims``: a sentence saying something was launched / is running / is queued is removed when
  no launch tool succeeded in the turn and no job exists (F1.11 l03: "la simulación se está ejecutando" after
  an injected note; nothing had been launched).
"""

from __future__ import annotations

import re

from driveflow.assistant import answer

#: a sentence about SPADE's capabilities: a capability verb or noun ...
_CAPABILITY = re.compile(
    r"\b(spade|simul\w*|sintoniz\w*|tunab\w*|tune[ds]?|tuning|entrena\w*|train\w*|soport\w*|support\w*|disponible\w*|available|"
    r"document\w*|implementad\w*|implemented|ofrece|offers?|permite|allows?|interfa\w*|chat)\b", re.I)
#: ... with a limit or a negation (the kind of claim that was false in F1.11)
_LIMIT = re.compile(r"\b(no|not|never|nunca|cannot|can't|only|solo|sólo|únicamente|unicamente|sin|ningun\w*|ningún|none|doesn't|isn't|aren't|tampoco)\b|❌", re.I)
_LAUNCHED = re.compile(r"(se está ejecutando|se esta ejecutando|está corriendo|esta corriendo|en ejecución|en ejecucion|está en cola|esta en cola|"
                       r"lanzad[oa]|iniciad[oa]|se ha iniciado|se inició|is running|now running|has been launched|was launched|launched|has started|"
                       r"is queued|in the queue|started the)", re.I)
NO_LAUNCH = {"es": "No se lanzó nada: la propuesta se confirma con el botón Run (o un «sí» justo después de la propuesta).",
             "en": "Nothing was launched: confirm the proposal with the Run button (or a bare yes right after the proposal)."}
LIMITS_LEAD = {"es": "Los límites de SPADE para este pedido están en la tarjeta (texto de las herramientas).",
               "en": "SPADE's limits for this request are in the card (the tools' own text)."}
NO_SOURCE = {"es": "No puedo confirmarlo con la documentación de SPADE.", "en": "I cannot confirm that from SPADE's documentation."}
ASK = {"es": "Para seguir necesito los valores de la tarjeta.", "en": "To continue I need the values in the card."}


def missing_card(state) -> dict | None:
    """The question card for the draft's missing REQUIRED fields (None when nothing is missing or a
    proposal is pending)."""
    if state.proposal is not None or state.mode not in ("configuration", "objective"):
        return None
    try:
        st = state.draft.status()
    except Exception:  # noqa: BLE001 -- a broken draft shows its own error elsewhere
        return None
    missing = st.get("missing_required") or []
    if not missing:
        return None
    from driveflow.assistant import tools

    lines = []
    for m in missing:
        field = m["field"]
        line = f"{field}" + (f" [{m['unit']}]" if m.get("unit") else "") + (f": {m['why']}" if m.get("why") else "")
        if m.get("choices"):
            line += f" (choices: {', '.join(map(str, m['choices']))})"
        if state.draft.plant is not None and "choices" not in m:
            try:
                sug = tools.suggest_value(state, field)
            except Exception:  # noqa: BLE001 -- no suggestion is a valid answer
                sug = {}
            if sug.get("value") is not None and sug.get("source"):
                line += f" -- suggestion: {sug['value']}{' ' + sug['unit'] if sug.get('unit') else ''} (source: {sug['source']})"
        lines.append(line)
    return {"tool": "ask", "title": "SPADE needs these values to continue", "source": "draft.status() (required fields)", "lines": lines}


def _evidence_limits(tool_events: list) -> list:
    """The tools' own words about a limit in this turn: refusals, not-simulable reasons, not-tunable notes."""
    out = []
    for e in tool_events:
        r = e.get("result")
        if not isinstance(r, dict):
            continue
        if isinstance(r.get("error"), str):
            out.append(r["error"])
        for err in r.get("errors") or []:
            if isinstance(err, dict) and err.get("error"):
                out.append(str(err["error"]))
        ns = r.get("not_simulable")
        if isinstance(ns, dict) and ns.get("reason"):
            subs = ", ".join(s.get("plant", "") for s in ns.get("closest_substitutes", []))
            out.append(f"{ns['reason']}" + (f"; closest substitutes: {subs}" if subs else ""))
        if r.get("tunable_from_chat") is False and r.get("note"):
            out.append(str(r["note"]))
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def bound_capabilities(text: str, tool_events: list, lang: str) -> tuple[str, list, int]:
    """(text, cards to add, sentences removed). Capability-limit sentences of the model are replaced by the
    tools' own words (a card); whole sentences only (answer.units)."""
    us = answer.units(text)
    keep, removed = [], 0
    for u, sep in us:
        if u.strip() and _CAPABILITY.search(u) and _LIMIT.search(u):
            removed += 1
            continue
        keep.append(u + ("\n" if "\n" in sep else " "))
    if not removed:
        return text, [], 0
    ev = _evidence_limits(tool_events)
    cards = [{"tool": "limits", "title": "What SPADE says about this request", "source": "tool results of this turn", "lines": ev}] if ev else []
    lead = LIMITS_LEAD[lang] if ev else NO_SOURCE[lang]
    out = "".join(keep).strip()
    return (lead + (" " + out if out else "")).strip(), cards, removed


def bound_launch_claims(text: str, launched_in_turn: bool, has_jobs: bool, lang: str) -> tuple[str, int]:
    if launched_in_turn or has_jobs:
        return text, 0
    us = answer.units(text)
    keep, removed = [], 0
    for u, sep in us:
        if u.strip() and _LAUNCHED.search(u):
            removed += 1
            continue
        keep.append(u + ("\n" if "\n" in sep else " "))
    if not removed:
        return text, 0
    out = "".join(keep).strip()
    return ((out + " ") if out else "") + NO_LAUNCH[lang], removed


def plant_cards_for_question(user_text: str) -> list:
    """Question mode: the catalog card of every plant the user names, by id or by a common name (the
    recommend_plant keywords), so an answer about a plant cites the catalog."""
    from driveflow.assistant import cards, tools

    low = (user_text or "").lower()
    plants = list(cards.plants_named(user_text))
    for kws, plant, _note in tools._KEYWORDS:
        if plant and plant not in plants and any(k in low for k in kws):
            plants.append(plant)
    return [c for c in (cards.plant_card(p) for p in plants) if c]
