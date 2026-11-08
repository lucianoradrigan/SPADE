"""Round Assistant, task F0.4: configs/plant_catalog.yaml is the single source of the plant
metadata, and it must agree with Scenario. If a field is added to Scenario without describing it,
or a range here drifts from Scenario.__post_init__, these tests fail."""

import dataclasses
import math
import warnings

import pytest

from driveflow.datagen.plant_catalog import NATIVE_PLANTS, describe_plant, load_catalog
from driveflow.datagen.scenario import Scenario

CATALOG = load_catalog()
FIELDS = {f.name for f in dataclasses.fields(Scenario)}
BASE = {
    "dc_perm_ex_v1": {},
    "dc_pmg132_v1": dict(plant_config_id="dc_pmg132_v1"),
    # v2 plants (decision E3, 2026-11-08): added entries only; no assertion changed
    "dc_perm_ex_v2": dict(plant_config_id="dc_perm_ex_v2"),
    "dc_pmg132_v2": dict(plant_config_id="dc_pmg132_v2"),
    "pmsm_v1": dict(controller_type="FOC_MTPA", plant_config_id="pmsm_v1", pmsm_current_ref_a=50.0),
    "vsc_dpc_v1": dict(controller_type="DPC", plant_config_id="vsc_dpc_v1"),
    "buck_v1": dict(controller_type="PI", plant_config_id="buck_v1"),
}
REJECTED = (ValueError, NotImplementedError)


def _accepts(plant: str, **kwargs) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # e.g. l_d > l_q while probing l_q
            Scenario("probe", **{**BASE[plant], **kwargs})
    except REJECTED:
        return False
    return True


def _probes(rng: dict, integer: bool) -> list[tuple[float, bool]]:
    """(value, should_be_accepted) just inside and just outside every bound of rng."""
    out = []
    for side, toward in (("min", -math.inf), ("max", math.inf)):
        if rng.get(side) is None:
            continue
        bound, inclusive = rng[side], rng.get(f"{side}_inclusive", True)
        step = (lambda b, d: b + (1 if d > 0 else -1)) if integer else math.nextafter
        inward = -toward
        if inclusive:
            out += [(bound, True), (step(bound, toward), False)]
        else:
            out += [(step(bound, inward), True), (bound, False)]
    return out


def test_every_scenario_field_is_covered_for_every_native_plant():
    assert set(CATALOG["fields"]) == FIELDS, f"missing: {sorted(FIELDS - set(CATALOG['fields']))}; unknown: {sorted(set(CATALOG['fields']) - FIELDS)}"
    for name, meta in CATALOG["fields"].items():
        applies, na = set(meta.get("applies_to", [])), set(meta.get("not_applicable", []))
        assert not applies & na, name
        assert applies | na == set(NATIVE_PLANTS), f"{name}: every native plant must be in applies_to or not_applicable"
        if applies:
            assert meta["importance"] in ("required", "important", "optional"), name
            assert meta.get("description"), name


CASES = [(plant, name) for name, meta in CATALOG["fields"].items() for plant in meta.get("applies_to", [])]


@pytest.mark.parametrize("plant, name", CASES)
def test_ranges_match_scenario_validation(plant, name):
    meta = CATALOG["fields"][name]
    rng = meta.get("range")
    if rng is not None and meta.get("validated", True):
        for value, ok in _probes(rng, meta.get("integer", False)):
            assert _accepts(plant, **{name: value}) is ok, f"{name}={value!r} on {plant}: expected {'accepted' if ok else 'rejected'}"
        if not meta.get("integer", False):
            assert not _accepts(plant, **{name: math.nan}), f"{name}: NaN must be rejected"
    for key, kmeta in (meta.get("keys") or {}).items():
        for value, ok in _probes(kmeta["range"], kmeta.get("integer", False)):
            assert _accepts(plant, **{name: {key: value}}) is ok, f"{name}[{key!r}]={value!r} on {plant}"
    for value in (meta.get("allowed_per_plant") or {}).get(plant, []):
        assert _accepts(plant, **{name: value}), f"{name}={value!r} must be accepted on {plant}"
    if meta.get("allowed_per_plant"):
        assert not _accepts(plant, **{name: "not_a_valid_choice"})


@pytest.mark.parametrize("plant", NATIVE_PLANTS)
def test_describe_plant(plant):
    d = describe_plant(plant)
    names = {f["name"] for f in d["fields"]}
    assert names | set(d["not_applicable"]) == FIELDS and not names & set(d["not_applicable"])
    for f in d["fields"]:
        assert {"name", "type", "unit", "convention", "range", "default", "default_source", "importance", "description"} <= set(f)
    if plant == "dc_perm_ex_v1":
        stock = d["stock_motor"]
        assert stock["motor_parameter"]["psi_e"] == 0.165 and stock["supply_v"] == 65.0
        assert stock["controller_design_check"]["PI"]["status"] == "ok"
        mp = next(f for f in d["fields"] if f["name"] == "motor_parameter_overrides")
        assert mp["keys"]["psi_e"]["default"] == 0.165 and "V*s/rad" in mp["keys"]["psi_e"]["unit"]
    by_name = {f["name"]: f for f in d["fields"]}
    expected_controller = "PI" if plant in ("dc_perm_ex_v1", "dc_pmg132_v1", "dc_perm_ex_v2", "dc_pmg132_v2", "buck_v1") else None  # v2 added (E3)
    assert by_name["controller_type"]["default"] == expected_controller
    assert by_name["plant_config_id"]["default"] == ("dc_perm_ex_v1" if plant == "dc_perm_ex_v1" else None)
    if plant == "pmsm_v1":
        pm = next(f for f in d["fields"] if f["name"] == "pmsm_motor_parameters")
        assert pm["keys"]["p"]["default"] == 3 and "PAIRS" in pm["keys"]["p"]["convention"]


def test_unknown_plant_is_refused():
    with pytest.raises(ValueError, match="unknown plant"):
        describe_plant("induction_v1")
