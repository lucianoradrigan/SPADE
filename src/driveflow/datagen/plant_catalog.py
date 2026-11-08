"""What each native plant lets a caller configure (round Assistant, task F0.4).

``describe_plant(plant_config_id)`` walks ``dataclasses.fields(Scenario)`` and returns, for every
field that applies to that plant: name, type, SI unit, convention, valid range, default (read from
the code) and where it comes from, importance and a one-line description. The metadata live in
``configs/plant_catalog.yaml``; tests/test_assistant_f0_plant_catalog.py checks that the YAML covers
every field, names no field that does not exist, and that each range matches
``Scenario.__post_init__`` (a value just inside and just outside every bound).

For the DC motor the result also carries the stock motor's parameters, limits, nominal values,
supply and load, and ``controller_design_check`` / ``mpc_design_check`` on that motor.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path

import yaml

from driveflow.datagen.scenario import Scenario

CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "plant_catalog.yaml"
NATIVE_PLANTS = ("dc_perm_ex_v1", "dc_pmg132_v1", "dc_perm_ex_v2", "dc_pmg132_v2", "pmsm_v1", "vsc_dpc_v1", "buck_v1")


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with CATALOG_PATH.open() as f:
        return yaml.safe_load(f)


def _type_name(field: dataclasses.Field) -> str:
    return field.type if isinstance(field.type, str) else getattr(field.type, "__name__", str(field.type))


def _key_defaults(field_name: str, plant_config_id: str = "dc_perm_ex_v1") -> dict:
    """Defaults of the keys of a dict field, read from the code (never from the YAML)."""
    if plant_config_id == "dc_pmg132_v1" and field_name in ("motor_parameter_overrides", "dc_limit_overrides", "dc_nominal_overrides"):
        from driveflow.datagen import runner

        return dict({"motor_parameter_overrides": runner.PMG132_MOTOR_PARAMETER, "dc_limit_overrides": runner.PMG132_LIMITS,
                     "dc_nominal_overrides": runner.PMG132_NOMINAL_VALUES}[field_name])
    if field_name == "motor_parameter_overrides":
        from driveflow.sim import DcPermanentlyExcitedMotor

        return dict(DcPermanentlyExcitedMotor._default_motor_parameter)
    if field_name == "dc_limit_overrides":
        from driveflow.sim import DcPermanentlyExcitedMotor

        return dict(DcPermanentlyExcitedMotor._default_limits)
    if field_name == "dc_nominal_overrides":
        from driveflow.sim import DcPermanentlyExcitedMotor

        return dict(DcPermanentlyExcitedMotor._default_nominal_values)
    if field_name == "pmsm_motor_parameters":
        from driveflow.control.classical import pmsm_foc
        from driveflow.sim import PermanentMagnetSynchronousMotor

        return {**pmsm_foc.MOTOR_PARAMS, "j_rotor": PermanentMagnetSynchronousMotor._default_motor_parameter["j_rotor"], "u_supply": pmsm_foc.U_SUPPLY}
    if field_name == "buck_parameter_overrides":
        from driveflow.sim.buck_system import BuckParams

        return {f.name: f.default for f in dataclasses.fields(BuckParams) if f.name in Scenario._BUCK_PARAMETERS}
    return {}


def _dc_stock(plant_config_id: str = "dc_perm_ex_v1") -> dict:
    from driveflow.control.design_check import controller_design_check, mpc_design_check
    from driveflow.datagen.runner import build_plant
    from driveflow.datagen.s21_warning import motor_reason

    plant = build_plant(Scenario("describe_plant", plant_config_id=plant_config_id))
    motor, load = plant.electrical_motor, plant.mechanical_load
    mp, j_total = dict(motor.motor_parameter), float(load.j_total)
    return {
        "motor_parameter": mp, "limits": dict(motor.limits), "nominal_values": dict(motor.nominal_values), "supply_v": float(plant.supply.u_nominal),
        "load": dict(load.base_load.load_parameter), "j_total_kg_m2": j_total, "control_period_s": float(plant.tau),
        "controller_design_check": {"PI": controller_design_check(mp, j_total, plant.tau, s21_reason=motor_reason(plant_config_id)),
                                    "MPC": mpc_design_check(mp, j_total, plant.tau)},
    }


def describe_plant(plant_config_id: str) -> dict:
    """Catalog of the Scenario fields that apply to one native plant."""
    catalog = load_catalog()
    if plant_config_id not in catalog["plants"]:
        raise ValueError(f"unknown plant {plant_config_id!r}; native plants are {list(catalog['plants'])}")
    fields_out, not_applicable = [], []
    for f in dataclasses.fields(Scenario):
        meta = catalog["fields"][f.name]
        if plant_config_id not in meta.get("applies_to", []):
            not_applicable.append(f.name)
            continue
        entry = {
            "name": f.name, "type": _type_name(f), "unit": meta.get("unit"), "convention": meta.get("convention"),
            "range": meta.get("range"), "validated_by_scenario": meta.get("validated", True), "importance": meta["importance"],
            "default": f.default if f.default is not dataclasses.MISSING else None, "default_source": meta.get("default_source", "Scenario field default"),
            "description": meta["description"],
        }
        if "allowed_per_plant" in meta:
            entry["allowed"] = meta["allowed_per_plant"][plant_config_id]
        # A Scenario default that this plant does not accept (controller_type "PI" or plant_config_id
        # "dc_perm_ex_v1" on the PMSM, say) is not a default here: the caller must set the field.
        invalid_here = (f.name == "plant_config_id" and entry["default"] != plant_config_id) or (
            "allowed" in entry and entry["default"] is not None and entry["default"] not in entry["allowed"])
        if invalid_here:
            entry["default"], entry["default_source"] = None, f"no default for {plant_config_id}: must be set"
        if "range_note" in meta:
            entry["range_note"] = meta["range_note"]
        if "keys" in meta:
            defaults = _key_defaults(f.name, plant_config_id)
            entry["keys"] = {k: {**v, "default": defaults.get(k)} for k, v in meta["keys"].items()}
        fields_out.append(entry)
    out = {"plant_config_id": plant_config_id, **catalog["plants"][plant_config_id], "fields": fields_out, "not_applicable": not_applicable}
    if plant_config_id in Scenario.DC_PLANTS:
        out["stock_motor"] = _dc_stock(plant_config_id)  # the plant's default motor
        # S21-3: the plant's non-blocking warnings (the default motor's), at the top for the assistant
        out["warnings"] = list(out["stock_motor"]["controller_design_check"]["PI"]["warnings"])
    return out
