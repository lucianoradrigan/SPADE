"""Deterministic post-processing of the model's final text (F1.9, decided by Luciano 2026-11-04 after
the F1.1 round 3 found false statements in the model's own sentences).

- Questions about SPADE: what the user reads is the card (the verbatim snippet and its source) plus AT
  MOST ONE context sentence of the model. ``spade_answer`` keeps the first sentence only if every number,
  finding ID, parameter name and negation in it also occurs in the cards; otherwise only the cards stay.
- Draft values: the model does not restate values or their origins; the draft panel shows them, each
  with its origin chip (``draft_panel``). ``strip_draft_numbers`` removes every sentence of the final
  text that contains a number of the draft.

Demo fixes (Luciano, 2026-11-08, after the milestone demo left "Requisitos: ¿Confirmas…"): the filter
removes WHOLE sentences and list lines and never cuts one. Sentences are split only where a new one
starts (after . ! ? and before an upper-case letter, ¿, ¡ or a list marker; or at a line break), so
"e.g. 5" or "aprox. 0,2" never splits; the line breaks of the kept text are kept; and a header line
ending in ":" whose content was removed (or that has none) is removed with it. The requirements and
their values are shown by the proposal card (cards.py), not by the model's text.
"""

from __future__ import annotations

import re

from driveflow.assistant.glossary import ID_RE

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?![\w])")
PARAM_RE = re.compile(r"`([^`]+)`|\b([a-z][a-z0-9]*_[a-z0-9_]+)\b")
NEGATION_RE = re.compile(r"\b(no|not|never|nunca|sin|ningun[oa]?|ningún|none|cannot|can't|isn't|aren't|doesn't|no es|tampoco|jam[aá]s)\b", re.I)
CARD_ONLY = {"es": "Según la documentación de SPADE (ver la tarjeta):", "en": "From SPADE's documentation (see the card):"}
SEE_PANEL = {"es": "Revisa el borrador en el panel y confirma la propuesta si está bien.", "en": "Check the draft in the panel and confirm the proposal if it is right."}
_ES = re.compile(r"[áéíóúñ¿¡]|\b(el|la|los|las|que|qué|con|para|del|una|un|simula|quiero|cómo|como)\b", re.I)


def language(text: str) -> str:
    return "es" if _ES.search(text or "") else "en"


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def sentences(text: str) -> list:
    return [s.strip() for s in SENTENCE_RE.split(text or "") if s.strip()]


def card_text(cards: list) -> str:
    return "\n".join([c.get("title", "") for c in cards] + [ln for c in cards for ln in c.get("lines", [])] + [c.get("source", "") for c in cards])


def claims_supported(sentence: str, cards: list) -> bool:
    """Every number, finding ID, parameter name and negation of the sentence occurs in the cards."""
    text = card_text(cards)
    low = text.lower()
    card_nums = {n for n in (_num(x) for x in NUMBER_RE.findall(text)) if n is not None}
    for x in NUMBER_RE.findall(sentence):
        n = _num(x)
        if n is not None and n not in card_nums:
            return False
    if any(i not in text for i in ID_RE.findall(sentence)):
        return False
    for a, b in PARAM_RE.findall(sentence):
        name = (a or b).strip()
        if name and name.lower() not in low:
            return False
    if NEGATION_RE.search(sentence) and not NEGATION_RE.search(text):
        return False
    return True


def spade_answer(text: str, cards: list, lang: str) -> tuple[str, bool]:
    """(text shown, kept): the first sentence if its claims are in the cards, else the fixed lead-in."""
    s = sentences(text)
    first = s[0] if s else ""
    if first and claims_supported(first, cards):
        return first, True
    return CARD_ONLY[lang], False


def draft_numbers(draft) -> set:
    nums = set()

    def add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            nums.add(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                add(x)

    for val in list(draft.preset.values()) + list(draft.base.values()) + [x for vs in draft.vary.values() for x in vs]:
        add(val.value)
    obj = draft.objective or {}
    for r in obj.get("requirements", []) or []:
        add(r.get("value") if isinstance(r, dict) else None)
    add((obj.get("evaluations") or {}).get("value") if isinstance(obj.get("evaluations"), dict) else obj.get("evaluations"))
    return nums


#: where a new sentence starts: after . ! ? followed by spaces and an upper-case letter, ¿, ¡, a list
#: marker or emphasis; or at a line break. Never inside "e.g. 5" or "aprox. 0,2 s".
_UNIT_BREAK = re.compile(r"(?<=[.!?])[ \t]+(?=[A-ZÁÉÍÓÚÑ¿¡*_\-•])|\n+")
_LIST_ITEM = re.compile(r"^(\s*)(?:[-*•+]|\d+[.)])\s+")
_HEADER = re.compile(r":[\s*_]*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
#: a period after these is not the end of a sentence (the unit goes on; a longer unit is never a cut)
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|p\.\s?ej|ej|aprox|approx|etc|vs|cf|fig|n[oº]|núm|num|máx|max|mín|min|dr|sr|sra|ca))\.$", re.I)


def units(text: str) -> list:
    """The sentences and lines of ``text`` as (unit, separator that follows it); joining them gives the text back."""
    out, pos = [], 0
    for m in _UNIT_BREAK.finditer(text or ""):
        if "\n" not in m.group(0) and _ABBREV.search(text[pos:m.start()]):
            continue
        out.append([text[pos:m.start()], m.group(0)])
        pos = m.end()
    out.append([(text or "")[pos:], ""])
    return [u for u in out if u[0].strip() or u[1]]


def _indent(unit: str) -> int | None:
    """Indentation of a list item (None = not a list item)."""
    m = _LIST_ITEM.match(unit)
    return len(m.group(1).expandtabs(4)) if m else None


def _has_content(header: str, rest: list) -> bool:
    """Is the next kept unit after ``header`` (a line ending in ':') its content?"""
    nxt = next((u for u, keep in rest if keep and u.strip()), None)
    if nxt is None:
        return False
    hi, ni = _indent(header), _indent(nxt)
    if hi is None and _TABLE_ROW.match(nxt):
        return True
    if ni is None:
        return False
    return hi is None or ni > hi


def strip_draft_numbers(text: str, draft) -> tuple[str, int]:
    """Drop the whole sentences / lines that contain a number of the draft, and the headers left
    without content; returns (text, units removed). A kept unit is never shortened."""
    nums = draft_numbers(draft)
    if not nums:
        return text, 0
    us = units(text)
    keep = []
    for u, _sep in us:
        found = {n for n in (_num(x) for x in NUMBER_RE.findall(u)) if n is not None}
        keep.append(not (found & nums))
    if all(keep):
        return text, 0
    # a markdown table whose body rows were all removed goes whole (its header row and |---| line too)
    i = 0
    while i < len(us):
        if _TABLE_ROW.match(us[i][0]):
            j = i
            while j < len(us) and _TABLE_ROW.match(us[j][0]):
                j += 1
            body = [k for k in range(i, j) if not _TABLE_SEP.match(us[k][0]) and not (k + 1 < j and _TABLE_SEP.match(us[k + 1][0]))]
            if not any(keep[k] for k in body):
                for k in range(i, j):
                    keep[k] = False
            i = j
        else:
            i += 1
    for i in range(len(us) - 1, -1, -1):  # backwards: a removed sub-header empties its parent too
        u = us[i][0]
        if keep[i] and _HEADER.search(u) and not _has_content(u, list(zip([x[0] for x in us[i + 1:]], keep[i + 1:]))):
            keep[i] = False
    removed = sum(1 for k, (u, _) in zip(keep, us) if not k and u.strip())
    out = []
    for k, (u, sep) in zip(keep, us):
        if k and u.strip():
            out.append(u + ("\n" if "\n" in sep else " "))
    return "".join(out).strip(), removed


ORIGIN_CHIP = {"user": "USER", "suggested": "SUGGESTED", "default": "DEFAULT", "derived": "DERIVED", "assumed": "ASSUMED", "datasheet": "DATASHEET"}


def draft_panel(draft) -> list:
    """Rows for the draft panel (CLI and page): field, value, unit, origin chip, source."""
    rows = []
    if draft.plant:
        rows.append({"field": "plant", "value": draft.plant, "unit": None, "origin": "—", "source": "draft"})
    if draft.controller_type:
        rows.append({"field": "controller_type", "value": draft.controller_type, "unit": None, "origin": "—", "source": "draft"})
    for part in (draft.preset, draft.base):
        for path, v in part.items():
            rows.append({"field": path, "value": v.value, "unit": v.unit, "origin": ORIGIN_CHIP.get(v.origin, v.origin.upper()), "source": v.source})
    for path, vs in draft.vary.items():
        rows.append({"field": path, "value": [v.value for v in vs], "unit": vs[0].unit if vs else None,
                     "origin": "/".join(sorted({ORIGIN_CHIP.get(v.origin, v.origin.upper()) for v in vs})), "source": "; ".join(sorted({v.source for v in vs}))})
    obj = draft.objective or {}
    for r in obj.get("requirements", []) or []:
        if isinstance(r, dict):
            rows.append({"field": f"requirement {r.get('metric')}", "value": f"{r.get('op', '')} {r.get('value')}", "unit": r.get("unit"),
                         "origin": ORIGIN_CHIP.get(r.get("origin", "user"), str(r.get("origin", "user")).upper()), "source": r.get("source", "")})
    return rows


def render_panel(rows: list) -> str:
    if not rows:
        return "(draft empty)"
    w = max(len(r["field"]) for r in rows)
    return "\n".join(f"  {r['field']:<{w}}  {r['value']}{' ' + r['unit'] if r.get('unit') else ''}  [{r['origin']}]" for r in rows)
