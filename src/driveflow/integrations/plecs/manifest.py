"""Interface manifest of a PLECS circuit for batch sweeps -- expansion task E2 (spec Sec. 3.4 stage 3).

SPADE does not read the blocks of a .plecs file: the manifest is DECLARED BY HAND and validated
here. It lists
- ``model_vars``: the ModelVars the sweep may set (``plecs.simulate(..., {"ModelVars": ...})``),
  each with a unit, a default and an allowed range;
- ``outputs``: the top-level outports in PORT ORDER (PLECS returns one signal per outport, in that
  order), each with a unit.

A manifest that does not match the circuit is caught at run time, not here: the RPC client rejects a
result whose number of signals differs from ``outputs``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: Columns every PLECS dataset row carries besides the outputs; outputs may not reuse them.
RESERVED = ("timestamp_s", "condition_index", "profile_id", "run_id", "status")


class ManifestError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ModelVar:
    name: str
    unit: str
    default: float
    minimum: float
    maximum: float
    description: str = ""


@dataclasses.dataclass(frozen=True)
class Output:
    name: str
    unit: str
    description: str = ""


@dataclasses.dataclass(frozen=True)
class InterfaceManifest:
    model_vars: tuple
    outputs: tuple
    declared_by: str = "user (hand-written; not read from the circuit)"

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self.outputs]

    def var(self, name: str) -> ModelVar:
        for v in self.model_vars:
            if v.name == name:
                return v
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {"model_vars": [dataclasses.asdict(v) for v in self.model_vars], "outputs": [dataclasses.asdict(o) for o in self.outputs], "declared_by": self.declared_by}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def check_values(self, values: dict) -> dict:
        """Floats for a ModelVars dict, or ManifestError: unknown names, non-finite values and values
        outside the declared range are rejected."""
        out = {}
        for k, v in values.items():
            try:
                mv = self.var(k)
            except KeyError:
                raise ManifestError(f"{k!r} is not a ModelVar of this manifest ({', '.join(x.name for x in self.model_vars)})") from None
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ManifestError(f"ModelVar {k!r} must be a finite number, got {v!r}")
            if not (mv.minimum <= v <= mv.maximum):
                raise ManifestError(f"ModelVar {k!r} = {v:g} {mv.unit} is outside its declared range [{mv.minimum:g}, {mv.maximum:g}]")
            out[k] = float(v)
        return out


def _num(where: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ManifestError(f"{where} must be a finite number, got {value!r}")
    return float(value)


def from_dict(data: dict) -> InterfaceManifest:
    if not isinstance(data, dict):
        raise ManifestError("the manifest must be an object with 'model_vars' and 'outputs'")
    raw_vars, raw_outs = data.get("model_vars") or [], data.get("outputs") or []
    if not raw_outs:
        raise ManifestError("declare at least one output (a top-level outport of the circuit)")
    names = set()
    model_vars = []
    for i, v in enumerate(raw_vars):
        name = str(v.get("name", "")).strip()
        if not _NAME.match(name):
            raise ManifestError(f"model_vars[{i}]: {name!r} is not a valid variable name")
        if name in names:
            raise ManifestError(f"model_vars[{i}]: duplicate name {name!r}")
        names.add(name)
        unit = str(v.get("unit", "")).strip()
        if not unit:
            raise ManifestError(f"model_vars[{i}] ({name}): a unit is required (use '-' for dimensionless)")
        lo, hi, d = _num(f"{name}.minimum", v.get("minimum")), _num(f"{name}.maximum", v.get("maximum")), _num(f"{name}.default", v.get("default"))
        if not lo < hi:
            raise ManifestError(f"{name}: minimum must be below maximum, got [{lo:g}, {hi:g}]")
        if not lo <= d <= hi:
            raise ManifestError(f"{name}: default {d:g} is outside [{lo:g}, {hi:g}]")
        model_vars.append(ModelVar(name, unit, d, lo, hi, str(v.get("description", ""))))
    outputs, out_names = [], set()
    for i, o in enumerate(raw_outs):
        name = str(o.get("name", "")).strip()
        if not _NAME.match(name):
            raise ManifestError(f"outputs[{i}]: {name!r} is not a valid signal name")
        if name in out_names or name in RESERVED:
            raise ManifestError(f"outputs[{i}]: {name!r} is duplicated or reserved")
        out_names.add(name)
        unit = str(o.get("unit", "")).strip()
        if not unit:
            raise ManifestError(f"outputs[{i}] ({name}): a unit is required (use '-' for dimensionless)")
        outputs.append(Output(name, unit, str(o.get("description", ""))))
    clash = names & out_names
    if clash:
        raise ManifestError(f"names used both as ModelVar and output: {sorted(clash)}")
    return InterfaceManifest(tuple(model_vars), tuple(outputs), str(data.get("declared_by") or InterfaceManifest.declared_by))


def from_json(text: str) -> InterfaceManifest:
    try:
        return from_dict(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"not valid JSON: {exc}") from None


#: The spec's didactic buck (Sec. 3.6) as a TEMPLATE for the form -- not read from any file.
EXAMPLE_BUCK = {
    "model_vars": [
        {"name": "Vin", "unit": "V", "default": 48.0, "minimum": 12.0, "maximum": 100.0, "description": "input voltage"},
        {"name": "Rload", "unit": "Ohm", "default": 6.0, "minimum": 1.0, "maximum": 50.0, "description": "load resistance"},
        {"name": "Vref", "unit": "V", "default": 24.0, "minimum": 1.0, "maximum": 95.0, "description": "output voltage reference of the circuit's own controller"},
    ],
    "outputs": [
        {"name": "v_out", "unit": "V", "description": "outport 1"},
        {"name": "i_L", "unit": "A", "description": "outport 2"},
    ],
}
