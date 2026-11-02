"""Quantities in the user's text, units, and the comparison rule of the no-invention check (C3).

Phase 1 rule (instructions F1.4): a number may enter the draft with origin ``user`` only if the user
wrote it. This module extracts every quantity the user wrote as (value, unit), converts it to SI
with ``UNITS`` (the one unit table of the assistant) and compares with a relative tolerance of
``REL_TOL`` after conversion:

- decimal comma and point both work ("1,5" = "1.5"); thousands separators too when unambiguous
  ("1.500,5", "1,500.5"); a lone separator followed by exactly three digits ("1.500", "1,500") is
  AMBIGUOUS in a bilingual chat and is never usable as a user value (the assistant asks);
- "20 %" and "20%" are kept as the quantity (20, "%"); turning it into a fraction (0.2) is allowed
  only for fields that are fractions, and turning it into a physical value (a load torque) is a
  derivation done by Python with its provenance (``draft``), never a user value;
- number words 0-20 and the tens up to 90 (plus hundred / cien) in Spanish and English. The
  Spanish articles "un" / "una" are NOT read as 1 ("un motor" is not a quantity);
- ranges and lists ("del 20 al 100 %", "20, 40 y 60 %", "from 20 to 100 %") share the unit written
  after the last number; a range yields only its endpoints (and a step if the user wrote one with
  "cada" / "every" / "in steps of"), never invented intermediate values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

REL_TOL = 1e-9

#: unit spelling -> (dimension, factor to the SI unit of that dimension). The SI units are the ones
#: of configs/plant_catalog.yaml. Hz is NOT a speed: an electrical frequency is not a mechanical
#: shaft speed, so it has its own dimension and never converts to rad/s.
UNITS: dict[str, tuple[str, float]] = {}


def _add(dimension: str, factor: float, *spellings: str) -> None:
    for s in spellings:
        UNITS[s] = (dimension, factor)


_add("voltage", 1.0, "V", "v", "volt", "volts", "voltios")
_add("voltage", 1e3, "kV", "kv")
_add("voltage", 1e-3, "mV")
_add("current", 1.0, "A", "amp", "amps", "ampere", "amperes", "amperios")
_add("current", 1e-3, "mA")
_add("current", 1e3, "kA")
_add("resistance", 1.0, "ohm", "ohms", "Ω", "ohmios", "ohmio")
_add("resistance", 1e-3, "mohm", "mΩ", "mOhm", "miliohm", "milliohm")
_add("resistance", 1e3, "kohm", "kΩ", "kOhm")
_add("inductance", 1.0, "H")
_add("inductance", 1e-3, "mH")
_add("inductance", 1e-6, "uH", "µH", "μH")
_add("capacitance", 1.0, "F")
_add("capacitance", 1e-3, "mF")
_add("capacitance", 1e-6, "uF", "µF", "μF")
_add("capacitance", 1e-9, "nF")
_add("power", 1.0, "W")
_add("power", 1e3, "kW", "kw")
_add("power", 745.69987158227, "hp", "HP")
_add("torque", 1.0, "N*m", "Nm", "N·m", "N.m", "N-m")
_add("speed", 1.0, "rad/s")
_add("speed", 2 * math.pi / 60, "rpm", "RPM", "min^-1", "min-1", "1/min", "r/min", "min⁻¹")
_add("frequency", 1.0, "Hz")
_add("frequency", 1e3, "kHz")
_add("time", 1.0, "s", "seg", "sec", "segundos", "seconds", "segundo", "second")
_add("time", 1e-3, "ms")
_add("time", 1e-6, "us", "µs", "μs")
_add("time", 60.0, "min")
_add("inertia", 1.0, "kg*m^2", "kg·m²", "kg*m2", "kgm2", "kg m^2", "kg·m^2", "kgm²")
_add("flux_constant", 1.0, "V*s/rad", "Vs/rad", "V·s/rad", "N*m/A", "Nm/A", "N·m/A")
_add("flux_linkage", 1.0, "V*s", "Vs", "V·s", "Wb")
_add("percent", 1.0, "%")

#: Spelled-out numbers. "un"/"una" are articles, deliberately absent.
NUMBER_WORDS = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


@dataclass(frozen=True)
class Quantity:
    """One number written by the user. ``unit`` is the spelling found (a key of UNITS) or None."""

    value: float
    unit: str | None
    text: str
    ambiguous: bool = False
    role: str = "value"  # value | range_start | range_end | step

    @property
    def dimension(self) -> str | None:
        return UNITS[self.unit][0] if self.unit in UNITS else None

    def si(self) -> float:
        return self.value * UNITS[self.unit][1] if self.unit in UNITS else self.value


class UnitError(ValueError):
    pass


def to_si(value: float, unit: str | None, target_unit: str | None) -> float:
    """Convert value[unit] to the SI unit ``target_unit`` (a catalog unit). Raises UnitError when the
    dimensions differ or a unit is unknown. A unit outside the table must match the target exactly."""
    if unit in (None, "") and target_unit in (None, ""):
        return float(value)
    if unit == target_unit:
        return float(value) * (UNITS[unit][1] if unit in UNITS else 1.0) / (UNITS[target_unit][1] if target_unit in UNITS else 1.0)
    if unit not in UNITS:
        raise UnitError(f"unknown unit {unit!r}")
    if target_unit not in UNITS:
        raise UnitError(f"field unit {target_unit!r} does not convert from {unit!r}")
    (d_from, f_from), (d_to, f_to) = UNITS[unit], UNITS[target_unit]
    if d_from != d_to:
        raise UnitError(f"{unit!r} ({d_from}) is not {target_unit!r} ({d_to})")
    return float(value) * f_from / f_to


def same(a: float, b: float, rel_tol: float = REL_TOL) -> bool:
    """The comparison of the no-invention check: equal within REL_TOL after conversion to SI."""
    if a == b:
        return True
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0)


# ---------------------------------------------------------------------------------------------
# Extraction

# Units are matched longest first so "mH" is not read as "m" + "H" and "rad/s" wins over "s".
_UNIT_ALTS = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
# A sign counts only at the start or after a space or bracket: in "20-30" the dash is a range.
_NUM = r"(?:(?<![\w)\].,])[-+])?(?:\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|(?<!\d)[.,]\d+)"
_TOKEN = re.compile(rf"(?P<num>{_NUM})(?:\s?(?P<unit>{_UNIT_ALTS})(?![A-Za-zÀ-ÿ0-9]))?")
_WORD = re.compile(r"(?<![A-Za-zÀ-ÿ])(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")(?![A-Za-zÀ-ÿ])", re.IGNORECASE)
_WORD_UNIT = re.compile(rf"\s?(?P<unit>{_UNIT_ALTS})(?![A-Za-zÀ-ÿ0-9])")
_STEP_BEFORE = re.compile(r"(cada|every|in steps of|en pasos de|con paso de|paso de|step of|steps of)\s*$", re.IGNORECASE)
_RANGE_START = re.compile(r"(del?|desde|from|between|entre)\s*$", re.IGNORECASE)
_RANGE_MID = re.compile(r"^\s*(al|a|hasta|to|and|y|-|–)\s*$", re.IGNORECASE)
_LIST_MID = re.compile(r"^\s*(,|y|and|o|or|;)\s*$", re.IGNORECASE)


def parse_number(token: str) -> tuple[float, bool]:
    """Parse one numeric token. Returns (value, ambiguous)."""
    t = token.strip()
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    exp = ""
    m = re.search(r"[eE][-+]?\d+$", t)
    if m:
        exp, t = m.group(0), t[: m.start()]
    if t.startswith((".", ",")):
        return sign * float("0." + t[1:] + exp), False
    seps = [c for c in t if c in ".,"]
    if not seps:
        return sign * float(t + exp), False
    if len(set(seps)) == 2:  # both kinds: the last one is the decimal separator
        dec = t[max(t.rfind("."), t.rfind(","))]
        thou = "," if dec == "." else "."
        return sign * float(t.replace(thou, "").replace(dec, ".") + exp), False
    sep = seps[0]
    groups = t.split(sep)
    if len(seps) > 1:  # "1.500.000": thousands only
        if all(len(g) == 3 for g in groups[1:]) and 1 <= len(groups[0]) <= 3:
            return sign * float("".join(groups) + exp), False
        raise ValueError(f"not a number: {token!r}")
    head, tail = groups
    if len(tail) == 3 and 1 <= len(head) <= 3 and not head.startswith("0") and not exp:
        # "1.500" / "1,500": thousands in one language, decimal in the other.
        return sign * float(head + "." + tail), True
    return sign * float(head + "." + tail + exp), False


def _in_identifier(text: str, start: int) -> bool:
    """Digits glued to a word are part of a name ("pmsm_v1", "AI-N26", "S-20", "F0.6"), not a quantity.
    A dash between two digits ("20-30") is a range, so the second number still counts."""
    if start == 0:
        return False
    prev = text[start - 1]
    if prev.isalpha() or prev == "_" or (prev.isdigit() and text[start] not in "+-"):
        return True
    if prev in "-." and start >= 2 and (text[start - 2].isalpha() or text[start - 2] == "_"):
        return True
    if prev == "." and start >= 2 and text[start - 2].isdigit():  # "F0.6": the ".6" after an identifier digit
        return _in_identifier(text, start - 2) or text[start - 2].isdigit()
    return False


def extract_quantities(text: str) -> list[Quantity]:
    """Every quantity the user wrote, in order. Units after the last number of a list or range are
    shared by the numbers before it; range endpoints and steps are marked by ``role``."""
    found: list[tuple[int, int, Quantity]] = []
    for m in _TOKEN.finditer(text):
        if _in_identifier(text, m.start()):
            continue
        try:
            value, amb = parse_number(m.group("num"))
        except ValueError:
            continue
        found.append((m.start(), m.end(), Quantity(value, m.group("unit"), m.group(0).strip(), amb)))
    for m in _WORD.finditer(text):
        unit_m = _WORD_UNIT.match(text, m.end())
        unit = unit_m.group("unit") if unit_m else None
        end = unit_m.end() if unit_m else m.end()
        found.append((m.start(), end, Quantity(float(NUMBER_WORDS[m.group(1).lower()]), unit, text[m.start():end].strip())))
    found.sort(key=lambda x: x[0])

    # Share a trailing unit backwards across a list or range: "20, 40 y 60 %", "del 20 al 100 %".
    out = [q for _, _, q in found]
    for i in range(len(found) - 1, 0, -1):
        _, _, q_next = found[i]
        s_prev, e_prev, q_prev = found[i - 1]
        between = text[e_prev:found[i][0]]
        if q_prev.unit is None and out[i].unit is not None and (_RANGE_MID.match(between) or _LIST_MID.match(between)):
            out[i - 1] = Quantity(q_prev.value, out[i].unit, q_prev.text, q_prev.ambiguous, q_prev.role)
    # Roles: range endpoints and steps.
    for i, (s, e, _q) in enumerate(found):
        before = text[max(0, s - 20):s]
        if _STEP_BEFORE.search(before):
            out[i] = Quantity(out[i].value, out[i].unit, out[i].text, out[i].ambiguous, "step")
        elif i + 1 < len(found) and _RANGE_START.search(before) and _RANGE_MID.match(text[e:found[i + 1][0]]):
            out[i] = Quantity(out[i].value, out[i].unit, out[i].text, out[i].ambiguous, "range_start")
            nxt = out[i + 1]
            out[i + 1] = Quantity(nxt.value, nxt.unit, nxt.text, nxt.ambiguous, "range_end")
    return out


def numbers_in(obj) -> list[float]:
    """Every number inside a tool result (nested dicts and lists); booleans are not numbers."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(numbers_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(numbers_in(v))
    return out
