"""The assistant's draft: what will be simulated, with the origin of every value (task F1.4, C9).

The draft lives in Python, not in the language model's memory (plan §4.1). The model changes it
with patches (``Draft.patch``) and reads ``Draft.status()`` after every change. A patch is applied
atomically: if one operation is refused, nothing changes and the reasons come back.

No-invention rule (instructions §2 and F1.4). Every number in a patch must be backed by evidence
recorded in a ``Ledger``:

- origin ``user``: the user wrote that quantity (``numbers.extract_quantities``, compared in SI with
  ``numbers.same``); an ambiguous number ("1.500") never counts;
- origin ``suggested``: a ``suggest_value`` result for THAT field returned this value;
- origin ``default``: it equals the field's own default on this plant (read from the code by
  ``describe_plant``); ``use_defaults`` reads it without the model sending any number;
- origin ``assumed``: it is in some tool result, and the patch says why it is assumed;
- origin ``derived``: only SPADE derives (a percentage of the rated torque, the values of a range
  with the user's step); a patch cannot send a derived number;
- origin ``datasheet``: reserved for phase 2, refused now.

Fault fields (fault type, severities, extra faults) are phase 4 and are refused; PLECS fields do not
apply to native plants; ``scenario_id`` is generated.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from driveflow.assistant import numbers as nb
from driveflow.datagen.plant_catalog import NATIVE_PLANTS, describe_plant
from driveflow.datagen.scenario import Scenario

ORIGINS = ("user", "derived", "suggested", "default", "assumed", "datasheet")
PATCH_ORIGINS = ("user", "suggested", "default", "assumed")  # what a patch may declare in phase 1
PHASE4_FIELDS = ("fault_type", "electrical_severity", "mechanical_severity", "fault_order_override", "extra_faults")
GENERATED_FIELDS = ("scenario_id", "plant_config_id", "controller_type")
#: preset = the machine (dict-field keys and supply); base = the operating point of every run.
MACHINE_FIELDS = ("dc_supply_v", "buck_vin_v")
#: fields that are fractions in [0, 1]: "50 %" may be given as 0.5.
FRACTION_FIELDS = ("pi_setpoint_weight", "vsc_capacitor_scale")
LOAD_FIELD = "load_torque_nm"
#: plan §4.4; configs/assistant.yaml (task F1.2) overrides them.
DEFAULT_LIMITS = {"max_experiments_per_launch": 50, "max_total_sim_seconds": 600.0}
SIM_COST_PATH = Path(__file__).resolve().parents[3] / "configs" / "sim_cost.yaml"
#: plants or controllers without their own measurement in sim_cost.yaml use one of the same structure.
COST_PROXY = {"dc_pmg132_v1/PI": "dc_perm_ex_v1/PI", "dc_pmg132_v1/MPC": "dc_perm_ex_v1/MPC", "pmsm_v1/FOC_naive": "pmsm_v1/FOC_MTPA",
              # v2 (E3): dcm_exact measured 1.08-1.12x the GEM plant's CPU (dc_perm_ex_v2_convergence.md); the proxy is declared in the basis
              "dc_perm_ex_v2/PI": "dc_perm_ex_v1/PI", "dc_perm_ex_v2/MPC": "dc_perm_ex_v1/MPC", "dc_pmg132_v2/PI": "dc_perm_ex_v1/PI", "dc_pmg132_v2/MPC": "dc_perm_ex_v1/MPC"}
#: Decision E3 (Luciano, 2026-11-08): the assistant offers only the v2 DC plants; the v1 plants are the frozen
#: "GEM-faithful mode" (S-21), kept for reproducing results, not offered.
FROZEN_V1 = ("dc_perm_ex_v1", "dc_pmg132_v1")
ASSISTANT_PLANTS = tuple(p for p in NATIVE_PLANTS if p not in FROZEN_V1)


class PatchError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# Evidence


@dataclass
class Evidence:
    kind: str  # "user" or "tool"
    source: str  # "user message 3", "tool describe_plant (call c1)"
    tool: str | None = None
    quantities: list = field(default_factory=list)  # user: list[nb.Quantity]
    numbers: list = field(default_factory=list)  # tool: list[float] (SI, as the tool returned them)
    result: object = None  # tool: the result itself (suggest_value results are matched by field)


class Ledger:
    """Everything the user wrote and every tool result of the session, in order."""

    def __init__(self) -> None:
        self.items: list[Evidence] = []

    def add_user_message(self, seq: int, text: str) -> None:
        self.items.append(Evidence("user", f"user message {seq}", quantities=nb.extract_quantities(text)))

    def add_tool_result(self, tool: str, call_id: str, result) -> None:
        self.items.append(Evidence("tool", f"tool {tool} (call {call_id})", tool=tool, numbers=nb.numbers_in(result), result=result))

    def user_quantity(self, value: float, unit: str | None, field_unit: str | None, *, fraction_field: bool = False, field_range: dict | None = None) -> str | None:
        """Source of a user quantity equal to value[unit] in the field's unit, or None. An ambiguous
        number counts only when exactly one of its two readings is inside ``field_range`` (SI) and
        that reading is the patch's value; the source then says how it was read."""
        try:
            target = nb.to_si(value, unit, field_unit)
        except nb.UnitError:
            target = None
        for ev in reversed(self.items):
            if ev.kind != "user":
                continue
            for q in ev.quantities:
                if q.ambiguous:
                    src = self._disambiguated(ev, q, value, unit, field_unit, target, field_range)
                    if src:
                        return src
                    continue
                if unit == "%" and q.unit == "%" and nb.same(q.value, value):
                    return f"{ev.source}: '{q.text}'"
                if fraction_field and q.unit == "%" and unit in (None, "") and nb.same(q.value / 100.0, value):
                    return f"{ev.source}: '{q.text}'"
                if target is None:
                    continue
                try:
                    if nb.same(nb.to_si(q.value, q.unit if q.unit is not None else unit, field_unit), target):
                        return f"{ev.source}: '{q.text}'"
                except nb.UnitError:
                    continue
        return None

    @staticmethod
    def _disambiguated(ev, q, value, unit, field_unit, target, field_range) -> str | None:
        if field_range is None or target is None or q.thousands_reading is None:
            return None
        readings = []
        for r in (q.value, q.thousands_reading):
            try:
                si = nb.to_si(r, q.unit if q.unit is not None else unit, field_unit)
            except nb.UnitError:
                return None
            readings.append((r, si, nb.in_range(si, field_range)))
        valid = [x for x in readings if x[2]]
        if len(valid) != 1 or not nb.same(valid[0][1], target):
            return None
        other = next(x for x in readings if not x[2])
        return f"{ev.source}: '{q.text}' read as {valid[0][0]:g} ({other[0]:g} is outside the valid range of the field)"

    def suggestion(self, path: str, si_value: float) -> str | None:
        """Source of a suggest_value result for this field with this value (in the field's SI unit)."""
        for ev in reversed(self.items):
            r = ev.result
            if ev.tool == "suggest_value" and isinstance(r, dict) and r.get("field") == path and _is_number(r.get("value")) and nb.same(float(r["value"]), si_value):
                return ev.source
        return None

    def tool_number(self, si_value: float, tools: tuple | None) -> str | None:
        for ev in reversed(self.items):
            if ev.kind == "tool" and (tools is None or ev.tool in tools) and any(nb.same(x, si_value) for x in ev.numbers):
                return ev.source
        return None


# ---------------------------------------------------------------------------------------------
# Values


@dataclass
class Value:
    value: object  # number (in ``unit``) or string
    unit: str | None
    origin: str
    source: str  # where it came from: the user's words, the tool result, the default's source
    note: str | None = None  # formula, rule or reason
    #: False for a default accepted with ``use_defaults``: shown in the draft, but not passed to
    #: Scenario, so the run is exactly the one with the field untouched (passing the stock motor's
    #: values as overrides would take another code path, e.g. the 0b-D1 load rule).
    passthrough: bool = True

    def to_dict(self) -> dict:
        d = {"value": self.value, "unit": self.unit, "origin": self.origin, "source": self.source}
        if self.note:
            d["note"] = self.note
        if not self.passthrough:
            d["passthrough"] = False
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Value":
        return cls(d["value"], d.get("unit"), d["origin"], d["source"], d.get("note"), d.get("passthrough", True))


@lru_cache(maxsize=None)
def _catalog(plant: str) -> dict:
    return describe_plant(plant)


def field_meta(plant: str, path: str) -> dict:
    """Catalog entry of a field, or of one key of a dict field ("motor_parameter_overrides.psi_e")."""
    name, _, key = path.partition(".")
    entries = {e["name"]: e for e in _catalog(plant)["fields"]}
    if name in PHASE4_FIELDS:
        raise PatchError(f"{name}: faults are configured from the chat in phase 4, not now")
    if name in GENERATED_FIELDS:
        raise PatchError(f"{name} is set by the draft itself (use set_plant / set_controller)")
    if name not in entries:
        raise PatchError(f"{name!r} does not apply to {plant} (describe_plant lists the fields that do)")
    entry = entries[name]
    if "keys" in entry:
        if not key:
            raise PatchError(f"{name} is a group: set one key, e.g. '{name}.{next(iter(entry['keys']))}'")
        if key not in entry["keys"]:
            raise PatchError(f"{name} has no key {key!r}; keys: {list(entry['keys'])}")
        k = entry["keys"][key]
        return {"name": path, "unit": k.get("unit"), "integer": k.get("integer", False), "default": k.get("default"), "default_source": entry.get("default_source"), "range": k.get("range"),
                "importance": entry["importance"], "type": "float"}
    if key:
        raise PatchError(f"{name} is not a group")
    return {"name": path, "unit": entry.get("unit"), "integer": entry.get("type") == "int", "default": entry.get("default"), "range": entry.get("range"),
            "default_source": entry.get("default_source"), "importance": entry["importance"], "type": entry.get("type"), "allowed": entry.get("allowed")}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------------------------
# Draft


@dataclass
class Draft:
    plant: str | None = None
    controller_type: str | None = None
    preset: dict = field(default_factory=dict)  # path -> Value
    base: dict = field(default_factory=dict)  # path -> Value
    vary: dict = field(default_factory=dict)  # path -> list[Value]
    combine: str = "product"
    assumptions: list = field(default_factory=list)
    objective: dict | None = None
    name: str = "assistant"

    # --- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "plant": self.plant, "controller_type": self.controller_type,
            "preset": {k: v.to_dict() for k, v in self.preset.items()},
            "experiments": {"base": {k: v.to_dict() for k, v in self.base.items()},
                            "vary": [{"field": k, "values": [v.to_dict() for v in vs]} for k, vs in self.vary.items()], "combine": self.combine},
            "assumptions": list(self.assumptions), "objective": self.objective, "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Draft":
        ex = d.get("experiments", {})
        return cls(plant=d.get("plant"), controller_type=d.get("controller_type"),
                   preset={k: Value.from_dict(v) for k, v in d.get("preset", {}).items()},
                   base={k: Value.from_dict(v) for k, v in ex.get("base", {}).items()},
                   vary={e["field"]: [Value.from_dict(v) for v in e["values"]] for e in ex.get("vary", [])},
                   combine=ex.get("combine", "product"), assumptions=list(d.get("assumptions", [])), objective=d.get("objective"), name=d.get("name", "assistant"))

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    # --- patches -------------------------------------------------------------------------
    def patch(self, ops: list, ledger: Ledger) -> dict:
        """Apply a list of operations atomically. Returns {"ok", "applied", "errors", "status"}."""
        if not isinstance(ops, list) or not ops:
            raise PatchError("a patch is a non-empty list of operations")
        work = copy.deepcopy(self)
        errors, applied = [], []
        for i, op in enumerate(ops):
            try:
                applied.append(work._apply(op, ledger))
            except (PatchError, nb.UnitError, KeyError, TypeError) as exc:
                errors.append({"op": i, "error": str(exc).strip("'\"")})
        if errors:
            return {"ok": False, "applied": [], "errors": errors, "status": self.status()}
        self.__dict__.update(work.__dict__)
        return {"ok": True, "applied": applied, "errors": [], "status": self.status()}

    def _slot(self, path: str) -> dict:
        is_group = self.plant is not None and any(e["name"] == path and "keys" in e for e in _catalog(self.plant)["fields"])
        return self.preset if ("." in path or path in MACHINE_FIELDS or is_group) else self.base

    def _apply(self, op: dict, ledger: Ledger) -> str:
        if not isinstance(op, dict) or "op" not in op:
            raise PatchError("every operation is an object with an 'op'")
        kind = op["op"]
        if kind == "set_plant":
            plant = op["plant"]
            if plant in FROZEN_V1:
                v2 = plant.replace("_v1", "_v2")
                raise PatchError(f"{plant} is the frozen GEM-faithful plant (one-quadrant converter artifact S-21), kept only to reproduce results; "
                                 f"the assistant simulates {v2} (same motor, converter in discontinuous conduction)")
            if plant not in ASSISTANT_PLANTS:
                raise PatchError(f"unknown plant {plant!r}; plants: {list(ASSISTANT_PLANTS)}")
            dropped = []
            if plant != self.plant:
                self.plant = plant
                for slot in (self.preset, self.base, self.vary):
                    for path in list(slot):
                        try:
                            field_meta(plant, path)
                        except PatchError:
                            dropped.append(path)
                            del slot[path]
                if self.controller_type and self.controller_type not in _catalog(plant)["controllers"]:
                    dropped.append("controller_type")
                    self.controller_type = None
            return f"plant = {plant}" + (f" (dropped, not applicable: {dropped})" if dropped else "")
        if self.plant is None:
            raise PatchError("set the plant first (set_plant)")
        if kind == "set_controller":
            c = op["controller_type"]
            allowed = _catalog(self.plant)["controllers"]
            if c not in allowed:
                raise PatchError(f"{self.plant} accepts controllers {allowed}, not {c!r}")
            self.controller_type = c
            return f"controller_type = {c}"
        if kind == "combine":
            if op["mode"] not in ("product", "zip"):
                raise PatchError("combine mode is 'product' or 'zip'")
            self.combine = op["mode"]
            return f"combine = {self.combine}"
        if kind == "unset":
            path = op["field"]
            found = any(slot.pop(path, None) is not None for slot in (self.preset, self.base, self.vary))
            if not found:
                raise PatchError(f"{path} is not set")
            return f"unset {path}"
        if kind == "use_defaults":
            out = []
            for path in op["fields"]:
                meta = self._default_meta(path)
                if meta["importance"] == "required" and meta["default"] is None:
                    raise PatchError(f"{path} has no default on {self.plant}: ask the user")
                self.preset.pop(path, None)
                self.base.pop(path, None)
                self.vary.pop(path, None)
                self._slot(path)[path] = Value(meta["default"], meta["unit"], "default", f"describe_plant({self.plant}): {meta['default_source']}",
                                               "SPADE default; the plant applies it by itself", passthrough=False)
                out.append(path)
            return f"defaults for {out}"
        if kind in ("set", "assume"):
            path = op["field"]
            origin = "assumed" if kind == "assume" else op.get("origin", "user")
            if kind == "assume" and not op.get("reason"):
                raise PatchError("an assumed value needs a 'reason'")
            v = self._checked_value(path, op["value"], op.get("unit"), origin, ledger, op.get("reason"))
            self.vary.pop(path, None)
            self._slot(path)[path] = v
            if origin == "assumed":
                self.assumptions = [a for a in self.assumptions if a["field"] != path] + [{"field": path, "value": op["value"], "unit": op.get("unit"), "reason": op["reason"]}]
            return f"{path} = {op['value']} {op.get('unit') or ''} ({origin})".replace("  ", " ")
        if kind == "vary":
            path = op["field"]
            origin = op.get("origin", "user")
            if "range" in op:
                values = self._range_values(path, op["range"], op.get("unit"), origin, ledger)
            else:
                if not isinstance(op.get("values"), list) or len(op["values"]) < 2:
                    raise PatchError("vary needs 'values' (two or more) or a 'range' {start, stop, step}")
                values = [self._checked_value(path, x, op.get("unit"), origin, ledger, op.get("reason")) for x in op["values"]]
            self.preset.pop(path, None)
            self.base.pop(path, None)
            self.vary[path] = values
            return f"vary {path} over {len(values)} values"
        raise PatchError(f"unknown operation {kind!r}; operations: set_plant, set_controller, set, assume, unset, vary, combine, use_defaults")

    def _default_meta(self, path: str) -> dict:
        """field_meta, also for a whole group (its default is the dict of its keys' defaults)."""
        entries = {e["name"]: e for e in _catalog(self.plant)["fields"]}
        e = entries.get(path)
        if e is not None and "keys" in e and path not in PHASE4_FIELDS:
            return {"name": path, "unit": None, "default": {k: v.get("default") for k, v in e["keys"].items()}, "default_source": e.get("default_source"),
                    "importance": e["importance"]}
        return field_meta(self.plant, path)

    def _checked_value(self, path: str, value, unit: str | None, origin: str, ledger: Ledger, reason: str | None) -> Value:
        meta = field_meta(self.plant, path)
        if origin == "derived":
            raise PatchError("derived values are computed by SPADE; send the user's quantity (e.g. '20 %') with origin 'user'")
        if origin == "datasheet":
            raise PatchError("datasheets are phase 2; origin 'datasheet' is not available yet")
        if origin not in PATCH_ORIGINS:
            raise PatchError(f"origin must be one of {PATCH_ORIGINS}")
        if isinstance(value, str):
            if meta.get("allowed") and value not in meta["allowed"]:
                raise PatchError(f"{path} must be one of {meta['allowed']}")
            if meta["unit"] is not None or meta.get("type") not in ("str",):
                raise PatchError(f"{path} takes a number, not text")
            return Value(value, None, origin, "user" if origin == "user" else origin)
        if not _is_number(value) or not math.isfinite(value):
            raise PatchError(f"{path}: value must be a finite number")
        field_unit = meta["unit"]
        unit = unit if unit not in ("",) else None
        percent_of_rated = path == LOAD_FIELD and unit == "%"
        if not percent_of_rated:
            if unit == "%" and path not in FRACTION_FIELDS and field_unit != "%":
                raise PatchError(f"{path} is in {field_unit}; a percentage has no meaning for it")
            if unit not in (None, "%"):
                nb.to_si(value, unit, field_unit)  # raises on a wrong dimension
            elif unit is None and field_unit not in (None, "%") and path not in FRACTION_FIELDS:
                raise PatchError(f"{path}: give the unit (the field is in {field_unit})")
        if origin == "user":
            src = ledger.user_quantity(value, unit, field_unit if not percent_of_rated else "%", fraction_field=path in FRACTION_FIELDS,
                                       field_range=None if percent_of_rated else meta.get("range"))
            if src is None:
                raise PatchError(f"{path} = {value} {unit or ''}: the user did not write this quantity (an ambiguous number like '1.500' does not count): ask the user".replace("  ", " "))
            if percent_of_rated:
                return Value(value, "%", "user", src, "percent of the rated torque; SPADE converts it with the motor's rated torque")
            return Value(value, unit, "user", src)
        if percent_of_rated:
            raise PatchError("a load percentage must come from the user")
        si = nb.to_si(value, unit, field_unit) if unit not in (None, "%") else (value / 100.0 if unit == "%" else value)
        if origin == "default":
            d = meta["default"]
            if not _is_number(d) or not nb.same(float(d), si):
                raise PatchError(f"{path} = {value} {unit or ''}: the default of {path} on {self.plant} is {d!r} ({meta['default_source']}); "
                                 "to keep a default use the operation use_defaults".replace("  ", " "))
            return Value(value, unit, "default", f"describe_plant({self.plant}): {meta['default_source']}")
        if origin == "suggested":
            src = ledger.suggestion(path, si)
            if src is None:
                raise PatchError(f"{path} = {value} {unit or ''} (suggested): no suggest_value result for {path} gave this value".replace("  ", " "))
            return Value(value, unit, "suggested", src, reason)
        src = ledger.tool_number(si, None) or ledger.tool_number(float(value), None)
        if src is None:
            raise PatchError(f"{path} = {value} {unit or ''} (assumed): no tool result contains this number".replace("  ", " "))
        return Value(value, unit, origin, src, reason)

    def _range_values(self, path: str, rng: dict, unit: str | None, origin: str, ledger: Ledger) -> list:
        try:
            start, stop, step = float(rng["start"]), float(rng["stop"]), float(rng["step"])
        except KeyError:
            raise PatchError("a range needs start, stop and step; if the user gave no step, ask or suggest one") from None
        checked = [self._checked_value(path, x, unit, origin, ledger, None) for x in (start, stop, step)]
        if step <= 0 or stop <= start:
            raise PatchError("a range needs start < stop and step > 0")
        n = (stop - start) / step
        if abs(n - round(n)) > 1e-9 * max(1.0, abs(n)):
            raise PatchError(f"step {step} does not divide {start}..{stop}")
        out = []
        for i in range(int(round(n)) + 1):
            x = start + i * step
            if i in (0, int(round(n))):
                out.append(checked[0 if i == 0 else 1])
            else:
                out.append(Value(x, checked[0].unit, "derived", f"range {start}..{stop} step {step}",
                                 f"step from {checked[2].source}; endpoints from {checked[0].source} and {checked[1].source}"))
        return out

    # --- rated torque and expansion ---------------------------------------------------------
    def rated_torque(self) -> tuple[float, str] | None:
        """Rated torque of the motor of this draft, with its provenance; None when unknown."""
        v = self.preset.get("dc_nominal_overrides.torque")
        if v is not None:
            return nb.to_si(v.value, v.unit, "N*m"), f"dc_nominal_overrides.torque ({v.origin}: {v.source})"
        if any(p.startswith("motor_parameter_overrides.") for p in self.preset):
            return None  # another motor: the plant's stock rating does not apply (0b-D1 rule)
        stock = _catalog(self.plant).get("stock_motor")
        if stock is None:
            return None
        return float(stock["nominal_values"]["torque"]), f"rated torque of the {self.plant} stock motor (default: describe_plant)"

    def _si(self, path: str, v: Value):
        if isinstance(v.value, str):
            return v.value
        meta = field_meta(self.plant, path)
        if path == LOAD_FIELD and v.unit == "%":
            rt = self.rated_torque()
            if rt is None:
                raise PatchError("a load given as % needs the motor's rated torque (dc_nominal_overrides.torque): ask the user")
            return v.value / 100.0 * rt[0]
        if v.unit == "%" and path in FRACTION_FIELDS:
            x = v.value / 100.0
        elif v.unit in (None, "%"):
            x = float(v.value)
        else:
            x = nb.to_si(v.value, v.unit, meta["unit"])
        if meta.get("integer"):
            if abs(x - round(x)) > 1e-9:
                raise PatchError(f"{path} must be an integer")
            return int(round(x))
        return x

    def _rows(self) -> list[dict]:
        paths = list(self.vary)
        if not paths:
            return [{}]
        lists = [self.vary[p] for p in paths]
        if self.combine == "zip":
            if len({len(x) for x in lists}) != 1:
                raise PatchError(f"combine 'zip' needs lists of equal length: {[len(x) for x in lists]}")
            combos = list(zip(*lists))
        else:
            combos = list(itertools.product(*lists))
        return [dict(zip(paths, c)) for c in combos]

    def experiment_count(self) -> int:
        try:
            return len(self._rows())
        except PatchError:
            return 0

    def conditions(self) -> list[dict]:
        """Scenario keyword dicts, one per experiment (what scenario_sweep receives)."""
        if self.plant is None:
            raise PatchError("no plant")
        out = []
        for i, row in enumerate(self._rows()):
            kw: dict = {"scenario_id": f"{self.name}_{i:03d}", "plant_config_id": self.plant}
            if self.controller_type:
                kw["controller_type"] = self.controller_type
            for path, v in [*self.preset.items(), *self.base.items(), *row.items()]:
                if not v.passthrough:
                    continue
                x = self._si(path, v)
                name, _, key = path.partition(".")
                if key:
                    kw.setdefault(name, {})[key] = x
                else:
                    kw[name] = x
            out.append(kw)
        return out

    def expand(self) -> list[Scenario]:
        return [Scenario(**kw) for kw in self.conditions()]

    # --- status --------------------------------------------------------------------------
    def status(self, limits: dict | None = None) -> dict:
        limits = {**DEFAULT_LIMITS, **(limits or {})}
        st: dict = {"plant": self.plant, "controller_type": self.controller_type, "missing_required": [], "missing_important": [], "defaults_used": [],
                    "invalid": [], "warnings": [], "experiments": 0, "derived": [], "cost": None, "limits": {}, "ready": False, "draft_hash": self.hash()}
        if self.plant is None:
            st["missing_required"].append({"field": "plant", "why": "which plant to simulate", "choices": list(ASSISTANT_PLANTS)})
            return st
        cat = _catalog(self.plant)
        if self.controller_type is None:
            entry = next(e for e in cat["fields"] if e["name"] == "controller_type")
            if entry["default"] is None:
                st["missing_required"].append({"field": "controller_type", "why": "no default on this plant", "choices": cat["controllers"]})
            else:
                st["defaults_used"].append({"field": "controller_type", "value": entry["default"], "source": entry["default_source"]})
        set_paths = set(self.preset) | set(self.base) | set(self.vary)
        for e in cat["fields"]:
            name = e["name"]
            if name in GENERATED_FIELDS or name in PHASE4_FIELDS:
                continue
            is_set = name in set_paths or any(p.startswith(name + ".") for p in set_paths)
            if is_set:
                continue
            if e["importance"] == "required" and e["default"] is None and "keys" not in e:
                st["missing_required"].append({"field": name, "unit": e["unit"], "why": e["description"]})
            elif e["importance"] == "important":
                default = {k: v.get("default") for k, v in e["keys"].items()} if "keys" in e else e["default"]
                st["missing_important"].append({"field": name, "unit": e["unit"], "default": default, "default_source": e["default_source"]})
            elif e.get("default") is not None and e["importance"] != "optional":
                st["defaults_used"].append({"field": name, "value": e["default"], "source": e["default_source"]})
        # load given as % of rated torque: show the torque SPADE will use, or why it cannot.
        loads = ([self.base[LOAD_FIELD]] if LOAD_FIELD in self.base else []) + list(self.vary.get(LOAD_FIELD, []))
        if any(v.unit == "%" for v in loads):
            rt = self.rated_torque()
            if rt is None:
                st["missing_required"].append({"field": "dc_nominal_overrides.torque", "unit": "N*m", "why": "the load is given as % of the rated torque"})
            else:
                st["derived"] += [{"field": LOAD_FIELD, "value": v.value / 100.0 * rt[0], "unit": "N*m", "origin": "derived",
                                   "formula": f"{v.value} % x {rt[0]:g} N*m", "rated_torque_from": rt[1]} for v in loads if v.unit == "%"]
        try:
            rows = self._rows()
            st["experiments"] = len(rows)
        except PatchError as exc:
            st["invalid"].append({"experiment": None, "error": str(exc)})
            rows = []
        if not st["missing_required"] and rows:
            try:
                conds = self.conditions()
            except PatchError as exc:
                st["invalid"].append({"experiment": None, "error": str(exc)})
                conds = []
            from driveflow.datagen.s21_warning import condition_warning

            for i, kw in enumerate(conds):
                try:
                    Scenario(**kw)
                except (TypeError, ValueError, NotImplementedError) as exc:
                    st["invalid"].append({"experiment": i, "error": str(exc)})
                    continue
                w = condition_warning(kw)  # S21-3: a warning, never blocking (``ready`` ignores it)
                if w:
                    st["warnings"].append({"experiment": i, **w})
            if conds:
                st["cost"] = estimate_cost(self.plant, self.controller_type or kw.get("controller_type") or "PI", [c.get("duration_s", Scenario.duration_s) for c in conds])
        sim_s = st["cost"]["sim_seconds"] if st["cost"] else 0.0
        st["limits"] = {"max_experiments_per_launch": limits["max_experiments_per_launch"], "max_total_sim_seconds": limits["max_total_sim_seconds"],
                        "experiments_exceeded": st["experiments"] > limits["max_experiments_per_launch"], "sim_seconds_exceeded": sim_s > limits["max_total_sim_seconds"]}
        st["ready"] = (not st["missing_required"] and not st["invalid"] and st["experiments"] > 0
                       and not st["limits"]["experiments_exceeded"] and not st["limits"]["sim_seconds_exceeded"])
        return st


@lru_cache(maxsize=1)
def _sim_cost() -> dict:
    with SIM_COST_PATH.open() as f:
        return yaml.safe_load(f)


def estimate_cost(plant: str, controller: str, durations_s: list) -> dict:
    """CPU seconds from configs/sim_cost.yaml (measured on one machine; not a guarantee)."""
    table = _sim_cost()["cpu_s_per_simulated_s"]
    key = f"{plant}/{controller}"
    used = key if key in table else COST_PROXY.get(key)
    sim_s = float(sum(durations_s))
    if used is None or used not in table:
        return {"sim_seconds": sim_s, "est_cpu_s": None, "basis": f"no measurement for {key}"}
    rate = float(table[used]["median_cpu_s_per_sim_s"])
    return {"sim_seconds": sim_s, "est_cpu_s": sim_s * rate, "cpu_s_per_sim_s": rate,
            "basis": f"configs/sim_cost.yaml {used}" + (f" (proxy for {key}: same structure)" if used != key else "") + f", measured {_sim_cost()['measured']} on one machine"}


def audit_trace(events: list) -> list:
    """Replay a conversation's draft changes against its own evidence (the no-invention test runs
    this on every recorded cassette). ``events`` in order: {"role": "user", "seq", "text"},
    {"role": "tool", "name", "call_id", "result"} and {"role": "patch", "ops", "accepted"}, where
    ``accepted`` is what the session reported. Returns the problems found (empty = clean): a patch
    the session accepted that the rule refuses, or the reverse."""
    led, draft, problems = Ledger(), Draft(), []
    for i, ev in enumerate(events):
        if ev["role"] == "user":
            led.add_user_message(ev["seq"], ev["text"])
        elif ev["role"] == "tool":
            led.add_tool_result(ev["name"], ev["call_id"], ev["result"])
        elif ev["role"] == "patch":
            try:
                r = draft.patch(ev["ops"], led)
            except PatchError as exc:  # a malformed patch (e.g. ops = []) is refused, not a crash of the replay
                r = {"ok": False, "errors": [{"op": None, "error": str(exc)}]}
            if r["ok"] != ev.get("accepted", True):
                problems.append({"event": i, "accepted_in_session": ev.get("accepted", True), "accepted_now": r["ok"], "errors": r["errors"]})
    return problems
