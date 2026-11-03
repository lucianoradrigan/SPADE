"""Phase 1, F1.4: the assistant's draft (driveflow.assistant.draft): provenance of every value,
atomic patches checked against the evidence, status(), expansion to Scenario, hash()."""

import math

import pytest

from driveflow.assistant.draft import Draft, Ledger, PatchError, estimate_cost
from driveflow.datagen import Scenario

RPM = 2 * math.pi / 60


def ledger(*texts, tools=()):
    led = Ledger()
    for i, t in enumerate(texts, 1):
        led.add_user_message(i, t)
    for name, result in tools:
        led.add_tool_result(name, f"c_{name}", result)
    return led


def dc_draft(led, *extra):
    d = Draft()
    r = d.patch([{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"}, *extra], led)
    assert r["ok"], r["errors"]
    return d


def test_configuration_request_expands_to_the_scenarios_a_caller_would_write():
    led = ledger("Motor DC de stock con PI a 1500 rpm, 2 s por corrida, carga del 20 al 100 % cada 20 %")
    d = dc_draft(led, {"op": "set", "field": "omega_ref_rad_s", "value": 1500, "unit": "rpm"}, {"op": "set", "field": "duration_s", "value": 2, "unit": "s"},
                 {"op": "vary", "field": "load_torque_nm", "range": {"start": 20, "stop": 100, "step": 20}, "unit": "%"})
    scenarios = d.expand()
    assert len(scenarios) == 5
    for i, (s, frac) in enumerate(zip(scenarios, (0.2, 0.4, 0.6, 0.8, 1.0))):
        expected = Scenario(f"assistant_{i:03d}", controller_type="PI", plant_config_id="dc_perm_ex_v1", omega_ref_rad_s=1500 * RPM, duration_s=2.0, load_torque_nm=frac * 16.0)
        assert s.omega_ref_rad_s == pytest.approx(expected.omega_ref_rad_s, rel=1e-12)
        assert s.load_torque_nm == pytest.approx(expected.load_torque_nm, rel=1e-12)
        assert (s.duration_s, s.plant_config_id, s.controller_type, s.scenario_id) == (2.0, "dc_perm_ex_v1", "PI", expected.scenario_id)
    st = d.status()
    assert st["ready"] and st["experiments"] == 5 and st["missing_required"] == [] and st["invalid"] == []
    assert [x["value"] for x in st["derived"]] == pytest.approx([3.2, 6.4, 9.6, 12.8, 16.0])
    assert "stock motor" in st["derived"][0]["rated_torque_from"]
    origins = [v.origin for v in d.vary["load_torque_nm"]]
    assert origins == ["user", "derived", "derived", "derived", "user"]  # intermediate values come from the user's step
    assert st["cost"]["sim_seconds"] == 10.0 and st["cost"]["est_cpu_s"] > 0


def test_untouched_fields_give_the_scenario_defaults_bit_for_bit():
    led = ledger("motor DC con PI")
    d = dc_draft(led)
    [kw] = d.conditions()
    assert kw == {"scenario_id": "assistant_000", "plant_config_id": "dc_perm_ex_v1", "controller_type": "PI"}
    # Accepting the defaults shows them in the draft but passes nothing to Scenario.
    r = d.patch([{"op": "use_defaults", "fields": ["motor_parameter_overrides", "dc_supply_v", "duration_s"]}], led)
    assert r["ok"], r["errors"]
    assert d.conditions() == [kw]
    assert d.preset["motor_parameter_overrides"].value == {"r_a": 0.016, "l_a": 1.9e-05, "psi_e": 0.165, "j_rotor": 0.025}
    assert d.preset["motor_parameter_overrides"].origin == "default"
    assert {m["field"] for m in d.status()["missing_important"]} == {"omega_ref_rad_s", "load_torque_nm", "dc_limit_overrides"}


def test_invented_numbers_are_refused_and_the_patch_is_atomic():
    led = ledger("Simula el motor a 3000 rpm")
    d = dc_draft(led)
    before = d.hash()
    r = d.patch([{"op": "set", "field": "duration_s", "value": 2, "unit": "s"}, {"op": "set", "field": "omega_ref_rad_s", "value": 3000, "unit": "rpm"}], led)
    assert not r["ok"] and r["errors"][0]["op"] == 0 and "did not write" in r["errors"][0]["error"]
    assert d.hash() == before  # nothing applied
    for value, unit in ((314, "rad/s"), (314.16, "rad/s"), (3000, "rad/s")):
        r = d.patch([{"op": "set", "field": "omega_ref_rad_s", "value": value, "unit": unit}], led)
        assert not r["ok"], (value, unit)
    assert d.patch([{"op": "set", "field": "omega_ref_rad_s", "value": 3000, "unit": "rpm"}], led)["ok"]


@pytest.mark.parametrize("op, message", [
    ({"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s", "origin": "derived"}, "computed by SPADE"),
    ({"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s", "origin": "datasheet"}, "phase 2"),
    ({"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s", "origin": "suggested"}, "suggest_value"),
    ({"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s", "origin": "default"}, "the default of"),
    ({"op": "assume", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}, "reason"),
    ({"op": "set", "field": "fault_type", "value": "outer_race"}, "phase 4"),
    ({"op": "set", "field": "pmsm_current_ref_a", "value": 5, "unit": "A"}, "does not apply"),
    ({"op": "set", "field": "omega_ref_rad_s", "value": 100}, "give the unit"),
    ({"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "V"}, "voltage"),
    ({"op": "set", "field": "motor_parameter_overrides", "value": 1}, "group"),
    ({"op": "set", "field": "scenario_id", "value": "x"}, "draft itself"),
    ({"op": "vary", "field": "load_torque_nm", "range": {"start": 20, "stop": 100}, "unit": "%"}, "step"),
    ({"op": "explode"}, "unknown operation"),
])
def test_refusals(op, message):
    led = ledger("100 rad/s, del 20 al 100 %")
    d = dc_draft(led)
    r = d.patch([op], led)
    assert not r["ok"] and message in r["errors"][0]["error"], r["errors"]


def test_suggested_default_and_assumed_values_need_their_backing():
    led = ledger("motor DC", tools=[("suggest_value", {"field": "duration_s", "value": 2.0, "unit": "s", "rule": "time to settle"}),
                                    ("describe_plant", {"range": {"max": 400.0}, "supply_v": 65.0})])
    d = dc_draft(led)
    assert d.patch([{"op": "set", "field": "duration_s", "value": 2.0, "unit": "s", "origin": "suggested"}], led)["ok"]
    assert d.base["duration_s"].source == "tool suggest_value (call c_suggest_value)"
    # A suggestion for another field does not back this one.
    r = d.patch([{"op": "set", "field": "omega_ref_rad_s", "value": 2.0, "unit": "rad/s", "origin": "suggested"}], led)
    assert not r["ok"] and "no suggest_value result for omega_ref_rad_s" in r["errors"][0]["error"]
    # "default" must be the field's own default, not any number of describe_plant (a range bound, say).
    assert d.patch([{"op": "set", "field": "duration_s", "value": 1.0, "unit": "s", "origin": "default"}], led)["ok"]
    r = d.patch([{"op": "set", "field": "omega_ref_rad_s", "value": 400.0, "unit": "rad/s", "origin": "default"}], led)
    assert not r["ok"] and "the default of omega_ref_rad_s" in r["errors"][0]["error"]
    r = d.patch([{"op": "set", "field": "dc_supply_v", "value": 65.0, "unit": "V", "origin": "default"}], led)
    assert not r["ok"] and "use_defaults" in r["errors"][0]["error"]  # its Scenario default is None (65 V is applied by the plant)
    r = d.patch([{"op": "assume", "field": "omega_ref_rad_s", "value": 65.0, "unit": "rad/s", "reason": "x"}], led)
    assert r["ok"] and d.assumptions == [{"field": "omega_ref_rad_s", "value": 65.0, "unit": "rad/s", "reason": "x"}]


def test_load_percent_needs_the_rated_torque_of_the_motor():
    # "1,146" would be ambiguous (thousands or decimal) and is refused; see test_assistant_numbers.
    led = ledger("motor de 220 V, psi_e 1,15 V*s/rad, 50 % de carga, par nominal 28,75 Nm")
    d = dc_draft(led, {"op": "set", "field": "motor_parameter_overrides.psi_e", "value": 1.15, "unit": "V*s/rad"},
                 {"op": "set", "field": "load_torque_nm", "value": 50, "unit": "%"})
    st = d.status()
    assert not st["ready"] and st["missing_required"][0]["field"] == "dc_nominal_overrides.torque"  # stock rating does not apply to another motor
    assert d.patch([{"op": "set", "field": "dc_nominal_overrides.torque", "value": 28.75, "unit": "Nm"}], led)["ok"]
    st = d.status()
    assert st["derived"][0]["value"] == pytest.approx(14.375) and "dc_nominal_overrides.torque" in st["derived"][0]["rated_torque_from"]
    [kw] = d.conditions()
    assert kw["load_torque_nm"] == pytest.approx(14.375) and kw["motor_parameter_overrides"] == {"psi_e": 1.15}


def test_pmg132_rated_torque_is_its_own():
    led = ledger("PMG 132 al 100 % de carga")
    d = Draft()
    assert d.patch([{"op": "set_plant", "plant": "dc_pmg132_v1"}, {"op": "set", "field": "load_torque_nm", "value": 100, "unit": "%"}], led)["ok"]
    assert d.conditions()[0]["load_torque_nm"] == pytest.approx(20.0)


def test_pmsm_requires_controller_and_current():
    led = ledger("PMSM con 10 A")
    d = Draft()
    assert d.patch([{"op": "set_plant", "plant": "pmsm_v1"}], led)["ok"]
    missing = {m["field"] for m in d.status()["missing_required"]}
    assert missing == {"controller_type", "pmsm_current_ref_a"}
    assert d.patch([{"op": "set_controller", "controller_type": "FOC_MTPA"}, {"op": "set", "field": "pmsm_current_ref_a", "value": 10, "unit": "A"}], led)["ok"]
    assert d.status()["ready"]
    assert d.expand()[0].pmsm_current_ref_a == 10.0


def test_changing_plant_drops_fields_that_do_not_apply():
    led = ledger("100 rad/s y 10 A")
    d = dc_draft(led, {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"})
    r = d.patch([{"op": "set_plant", "plant": "pmsm_v1"}], led)
    assert r["ok"] and "omega_ref_rad_s" in r["applied"][0] and "controller_type" in r["applied"][0]
    assert d.base == {} and d.controller_type is None


def test_scenario_validation_and_limits_show_in_status():
    led = ledger("PMSM con 500 A")
    d = Draft()
    assert d.patch([{"op": "set_plant", "plant": "pmsm_v1"}, {"op": "set_controller", "controller_type": "FOC_MTPA"},
                    {"op": "set", "field": "pmsm_current_ref_a", "value": 500, "unit": "A"}], led)["ok"]
    st = d.status()
    assert not st["ready"] and st["invalid"] and "400" in st["invalid"][0]["error"]
    led = ledger("buck con 60 s por corrida, 10, 20 y 30 V de salida")
    d = Draft()
    assert d.patch([{"op": "set_plant", "plant": "buck_v1"}, {"op": "set", "field": "duration_s", "value": 60, "unit": "s"},
                    {"op": "vary", "field": "buck_vout_ref_v", "values": [10, 20, 30], "unit": "V"}], led)["ok"]
    st = d.status(limits={"max_total_sim_seconds": 100})
    assert st["experiments"] == 3 and st["limits"]["sim_seconds_exceeded"] and not st["ready"]


def test_zip_and_product():
    led = ledger("100 y 200 rad/s, 1 y 2 s")
    d = dc_draft(led, {"op": "vary", "field": "omega_ref_rad_s", "values": [100, 200], "unit": "rad/s"},
                 {"op": "vary", "field": "duration_s", "values": [1, 2], "unit": "s"})
    assert d.experiment_count() == 4
    assert d.patch([{"op": "combine", "mode": "zip"}], led)["ok"]
    assert [(c["omega_ref_rad_s"], c["duration_s"]) for c in d.conditions()] == [(100.0, 1.0), (200.0, 2.0)]


def test_round_trip_and_hash():
    led = ledger("1500 rpm")
    d = dc_draft(led, {"op": "set", "field": "omega_ref_rad_s", "value": 1500, "unit": "rpm"}, {"op": "use_defaults", "fields": ["duration_s"]})
    d2 = Draft.from_dict(d.to_dict())
    assert d2.to_dict() == d.to_dict() and d2.hash() == d.hash()
    assert d2.conditions() == d.conditions()
    d2.patch([{"op": "unset", "field": "omega_ref_rad_s"}], led)
    assert d2.hash() != d.hash()


def test_cost_proxy_is_declared():
    c = estimate_cost("dc_pmg132_v1", "MPC", [1.0, 1.0])
    assert c["sim_seconds"] == 2.0 and "proxy for dc_pmg132_v1/MPC" in c["basis"]
    with pytest.raises(PatchError):
        Draft().conditions()


CASSETTES = __import__("pathlib").Path(__file__).parent / "cassettes"


def test_no_invention_on_a_trace_and_on_every_cassette():
    from driveflow.assistant.draft import audit_trace
    import json

    trace = [{"role": "user", "seq": 1, "text": "motor DC a 1500 rpm"},
             {"role": "patch", "ops": [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set", "field": "omega_ref_rad_s", "value": 1500, "unit": "rpm"}], "accepted": True},
             {"role": "patch", "ops": [{"op": "set", "field": "duration_s", "value": 3, "unit": "s"}], "accepted": True}]
    problems = audit_trace(trace)
    assert len(problems) == 1 and problems[0]["event"] == 2  # 3 s was never given: an invented value accepted in a session is caught
    # Every recorded cassette (task F1.2 onwards) must replay clean.
    for path in sorted(CASSETTES.glob("*.json")) if CASSETTES.is_dir() else []:
        events = json.loads(path.read_text()).get("draft_events", [])
        assert audit_trace(events) == [], path.name


def test_ambiguous_number_resolved_by_the_field_range_is_reported():
    led = ledger("PMSM con 1.500 A y peso de consigna 1,000")
    d = Draft()
    assert d.patch([{"op": "set_plant", "plant": "pmsm_v1"}, {"op": "set", "field": "pmsm_current_ref_a", "value": 1.5, "unit": "A"}], led)["ok"]
    assert "read as 1.5" in d.base["pmsm_current_ref_a"].source
    # psi_e: both readings are valid (only > 0 is required), so the assistant must ask.
    led = ledger("motor DC con psi_e 1,146 V*s/rad")
    d = dc_draft(led)
    assert not d.patch([{"op": "set", "field": "motor_parameter_overrides.psi_e", "value": 1.146, "unit": "V*s/rad"}], led)["ok"]
