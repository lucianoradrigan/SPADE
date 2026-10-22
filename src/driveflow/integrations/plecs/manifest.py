"""Interface manifest of a PLECS circuit for batch sweeps -- expansion task E2 (spec Sec. 3.4 stage 3).

SPADE does not read the blocks of a .plecs file: the manifest is DECLARED BY HAND and validated
here. It lists
- ``model_vars``: the ModelVars the sweep may set (``plecs.simulate(..., {"ModelVars": ...})``),
  each with a unit, a default and an allowed range;
- ``outputs``: the top-level outports in PORT ORDER (PLECS returns one signal per outport, in that
  order), each with a unit.

Surrogate plants (task E5, decision D3 of 2026-10-21): the surrogate's ACTION is the duty cycle.
For a PLECS circuit to feed a surrogate, the duty is expected to be exposed either as a ModelVar
named ``D`` (a sweep variable, e.g. a duty sequence parameter) or as an input of the atomic subsystem
(FMU route, tasks E3/E4), and declared here like any other ModelVar or input.

Surrogate source (task A1, round PLECS/OM): two optional blocks make a buck circuit's sweeps usable
to fit a surrogate Gym plant:
- ``excitation``: the action profile the circuit applies during each run, parametrised by ModelVars
  (integrations/plecs/excitation.py), and the outport that records the action actually applied;
- ``surrogate``: which outputs are the plant states and which ModelVars are the operating
  conditions (template ``buck``: states i_l, v_out; conditions v_in_v, load_resistance_ohm).

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
    excitation: dict | None = None
    surrogate: dict | None = None

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self.outputs]

    def var(self, name: str) -> ModelVar:
        for v in self.model_vars:
            if v.name == name:
                return v
        raise KeyError(name)

    def to_dict(self) -> dict:
        d = {"model_vars": [dataclasses.asdict(v) for v in self.model_vars], "outputs": [dataclasses.asdict(o) for o in self.outputs], "declared_by": self.declared_by}
        if self.excitation is not None:
            d["excitation"] = self.excitation
        if self.surrogate is not None:
            d["surrogate"] = self.surrogate
        return d

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
    excitation = surrogate = None
    if data.get("excitation") is not None:
        from driveflow.integrations.plecs import excitation as ex

        try:
            excitation = ex.check_spec(data["excitation"], names, out_names)
        except ex.ExcitationError as exc:
            raise ManifestError(str(exc)) from None
    if data.get("surrogate") is not None:
        surrogate = _check_surrogate(data["surrogate"], names, out_names, excitation)
    return InterfaceManifest(tuple(model_vars), tuple(outputs), str(data.get("declared_by") or InterfaceManifest.declared_by), excitation, surrogate)


#: surrogate template -> (state roles, condition roles)
SURROGATE_TEMPLATES = {"buck": (("i_l", "v_out"), ("v_in_v", "load_resistance_ohm"))}


def _check_surrogate(spec, var_names: set, out_names: set, excitation) -> dict:
    if not isinstance(spec, dict):
        raise ManifestError("surrogate must be an object with 'template', 'states' and 'conditions'")
    tpl = spec.get("template")
    if tpl not in SURROGATE_TEMPLATES:
        raise ManifestError(f"surrogate.template must be one of {sorted(SURROGATE_TEMPLATES)}, got {tpl!r}")
    if excitation is None:
        raise ManifestError("a surrogate source needs an 'excitation' block: constant ModelVars only give step responses")
    states, conds = dict(spec.get("states") or {}), dict(spec.get("conditions") or {})
    s_roles, c_roles = SURROGATE_TEMPLATES[tpl]
    if sorted(states) != sorted(s_roles) or any(v not in out_names for v in states.values()):
        raise ManifestError(f"surrogate.states must map {list(s_roles)} to declared outputs, got {states}")
    if sorted(conds) != sorted(c_roles) or any(v not in var_names for v in conds.values()):
        raise ManifestError(f"surrogate.conditions must map {list(c_roles)} to declared ModelVars, got {conds}")
    return {"template": tpl, "states": {r: states[r] for r in s_roles}, "conditions": {r: conds[r] for r in c_roles}}


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


#: DAB template (task E9 preparation, round 2026-10-21, D8), declared by hand from the
#: initialization commands of Fund.'s DAB_sample1.plecs (8_PE_Simulation_Automation/PlecsAutomation,
#: downloaded to data/upstream/fund/dab/, not committed; licence unclear -> non-commercial use):
#: the notebook passes D1, D2, P and Vref from Python and sets Ro = Vref^2/P; Vin = 200 V, fs = 50 kHz,
#: Lr = 50 µH, Cf = 160 µF are fixed in the file. The ModelVar ranges below are a template, not
#: values read from the circuit. IMPORTANT: DAB_sample1.plecs has NO top-level outports (it records
#: with scopes), so plecs.simulate returns no Time/Values for it over RPC until outports are added in
#: PLECS; the outputs listed here are the ones such outports would need to provide, in port order.
EXAMPLE_DAB = {
    "model_vars": [
        {"name": "D1", "unit": "-", "default": 1.0, "minimum": 0.0, "maximum": 1.0, "description": "inner phase-shift ratio, primary bridge (TPS)"},
        {"name": "D2", "unit": "-", "default": 1.0, "minimum": 0.0, "maximum": 1.0, "description": "inner phase-shift ratio, secondary bridge (TPS)"},
        {"name": "P", "unit": "W", "default": 600.0, "minimum": 100.0, "maximum": 1200.0, "description": "power reference of the circuit's own PI loop"},
        {"name": "Vref", "unit": "V", "default": 200.0, "minimum": 100.0, "maximum": 300.0, "description": "output voltage reference"},
        {"name": "Ro", "unit": "Ohm", "default": 66.67, "minimum": 5.0, "maximum": 1000.0, "description": "load resistance; the notebook sets Ro = Vref^2 / P"},
    ],
    "outputs": [
        {"name": "v_out", "unit": "V", "description": "outport 1 (to be added in PLECS)"},
        {"name": "i_Lr", "unit": "A", "description": "outport 2: leakage-inductor current (to be added in PLECS)"},
    ],
}
#: Buck with an excited duty for a SURROGATE source (task A1): the circuit runs open loop, its duty
#: follows the PRBS profile of integrations/plecs/excitation.py, and outport 3 records the duty applied.
EXAMPLE_BUCK_SURROGATE = {
    "model_vars": [
        {"name": "Vin", "unit": "V", "default": 48.0, "minimum": 12.0, "maximum": 100.0, "description": "input voltage"},
        {"name": "Rload", "unit": "Ohm", "default": 6.0, "minimum": 1.0, "maximum": 50.0, "description": "load resistance"},
        {"name": "prbs_seed", "unit": "-", "default": 1.0, "minimum": 1.0, "maximum": 127.0, "description": "PRBS7 seed of the duty excitation"},
        {"name": "D_min", "unit": "-", "default": 0.1, "minimum": 0.0, "maximum": 1.0, "description": "duty level for a 0 bit"},
        {"name": "D_max", "unit": "-", "default": 0.8, "minimum": 0.0, "maximum": 1.0, "description": "duty level for a 1 bit"},
        {"name": "T_hold", "unit": "s", "default": 0.002, "minimum": 1e-4, "maximum": 0.1, "description": "hold time of each PRBS bit"},
    ],
    "outputs": [
        {"name": "v_out", "unit": "V", "description": "outport 1"},
        {"name": "i_L", "unit": "A", "description": "outport 2"},
        {"name": "duty", "unit": "-", "description": "outport 3: the duty actually applied"},
    ],
    "excitation": {"kind": "prbs", "action_output": "duty", "params": {"prbs_seed": "prbs_seed", "D_min": "D_min", "D_max": "D_max", "T_hold": "T_hold"}},
    "surrogate": {"template": "buck", "states": {"i_l": "i_L", "v_out": "v_out"}, "conditions": {"v_in_v": "Vin", "load_resistance_ohm": "Rload"}},
}
TEMPLATES = {"Buck (spec Sec. 3.6)": EXAMPLE_BUCK, "Buck, excited duty (surrogate source)": EXAMPLE_BUCK_SURROGATE, "DAB (Fund. DAB_sample1, needs outports)": EXAMPLE_DAB}
