"""Phase 1, F1.4 (C3): extraction of the user's quantities and the comparison rule of the
no-invention check (driveflow.assistant.numbers)."""

import math

import pytest

from driveflow.assistant import numbers as nb
from driveflow.assistant.draft import Ledger


def q(text):
    return [(x.value, x.unit, x.role, x.ambiguous) for x in nb.extract_quantities(text)]


@pytest.mark.parametrize("token, value, ambiguous", [
    ("1,5", 1.5, False), ("1.5", 1.5, False), ("0,181", 0.181, False), ("1.500,5", 1500.5, False), ("1,500.5", 1500.5, False),
    ("1.500", 1.5, True), ("1,500", 1.5, True), ("1.500.000", 1_500_000.0, False), ("16e-3", 0.016, False), ("-5", -5.0, False), (",5", 0.5, False),
    ("1500", 1500.0, False), ("12,25", 12.25, False),
])
def test_parse_number(token, value, ambiguous):
    v, amb = nb.parse_number(token)
    assert v == pytest.approx(value) and amb is ambiguous


def test_units_and_si():
    assert q("220 V y 1,5 A") == [(220.0, "V", "value", False), (1.5, "A", "value", False)]
    assert nb.to_si(5, "kW", "W") == 5000.0
    assert nb.to_si(300, "rpm", "rad/s") == pytest.approx(300 * 2 * math.pi / 60)
    assert nb.to_si(16, "mH", "H") == pytest.approx(0.016)
    with pytest.raises(nb.UnitError):
        nb.to_si(50, "Hz", "rad/s")  # an electrical frequency is not a shaft speed
    with pytest.raises(nb.UnitError):
        nb.to_si(5, "V", "A")


def test_words_and_articles():
    assert q("tres motores, five runs") == [(3.0, None, "value", False), (5.0, None, "value", False)]
    assert q("un motor de una tonelada") == []  # articles are not quantities
    assert q("veinte segundos") == [(20.0, "segundos", "value", False)]


def test_lists_and_ranges_share_the_unit():
    assert q("20, 40 y 60 %") == [(20.0, "%", "value", False), (40.0, "%", "value", False), (60.0, "%", "value", False)]
    assert q("del 20 al 100 % cada 20 %") == [(20.0, "%", "range_start", False), (100.0, "%", "range_end", False), (20.0, "%", "step", False)]
    assert q("from 20 to 100 %") == [(20.0, "%", "range_start", False), (100.0, "%", "range_end", False)]


def test_identifiers_are_not_quantities():
    assert q("pmsm_v1, AI-N26, S-20 y F0.6") == []
    assert q("20-30 %") == [(20.0, "%", "value", False), (30.0, "%", "value", False)]


def _ledger(text):
    led = Ledger()
    led.add_user_message(1, text)
    return led


# The comparison rule, with explicit false-positive (a user value refused) and false-negative
# (an invented value accepted) cases.
@pytest.mark.parametrize("text, value, unit, field_unit", [
    ("a 300 rpm", 300, "rpm", "rad/s"),
    ("a 300 rpm", 300 * 2 * math.pi / 60, "rad/s", "rad/s"),  # same quantity, converted exactly
    ("un motor de 5 kW", 5000, "W", "W"),
    ("resistencia 1,6 ohm", 1.6, "ohm", "ohm"),
    ("resistance 1.6 ohm", 1.6, "ohm", "ohm"),
    ("l_a de 16 mH", 0.016, "H", "H"),
    ("tensión 1.500,5 V", 1500.5, "V", "V"),
    ("veinte segundos", 20, "s", "s"),
])
def test_user_values_are_accepted(text, value, unit, field_unit):
    assert _ledger(text).user_quantity(value, unit, field_unit) is not None


@pytest.mark.parametrize("text, value, unit, field_unit", [
    ("a 3000 rpm", 314, "rad/s", "rad/s"),  # the model "rounds" 314.159... to 314
    ("a 3000 rpm", 314.16, "rad/s", "rad/s"),
    ("a 300 rpm", 300, "rad/s", "rad/s"),  # the number without its unit is another quantity
    ("a 1.500 rpm", 1500, "rpm", "rad/s"),  # ambiguous: never a user value
    ("a 1.500 rpm", 1.5, "rpm", "rad/s"),
    ("un motor de 5 kW", 5, "W", "W"),
    ("un motor", 1, None, None),  # article, not a number
    ("220 V", 220, "A", "A"),  # wrong dimension
    ("del 20 al 100 %", 60, "%", "%"),  # an intermediate value of a range is not the user's
])
def test_invented_values_are_refused(text, value, unit, field_unit):
    assert _ledger(text).user_quantity(value, unit, field_unit) is None


def test_fraction_fields_accept_a_percentage():
    led = _ledger("peso de consigna 50 %")
    assert led.user_quantity(0.5, None, None, fraction_field=True) is not None
    assert led.user_quantity(0.5, None, None, fraction_field=False) is None


def test_tool_numbers():
    led = Ledger()
    led.add_tool_result("describe_plant", "c1", {"nominal": {"torque": 16.0, "ok": True}, "list": [1e-3, None]})
    assert led.tool_number(16.0, ("describe_plant",)) is not None
    assert led.tool_number(16.0, ("suggest_value",)) is None
    assert led.tool_number(1.0, None) is None  # True is not a number


# Range disambiguation (DECISIONES_S21_F1 §5 point 2): an ambiguous number counts only when exactly
# one of its readings is inside the field's valid range; Python decides and reports the reading.
@pytest.mark.parametrize("text, value, unit, field_unit, rng, accepted", [
    ("peso 1,000", 1.0, None, None, {"min": 0.0, "max": 1.0}, True),          # 1000 outside [0, 1]
    ("peso 1,000", 1000.0, None, None, {"min": 0.0, "max": 1.0}, False),
    ("1.500 A", 1.5, "A", "A", {"min": -400.0, "max": 400.0}, True),           # 1500 A outside +-400 A
    ("psi_e 1,146 V*s/rad", 1.146, "V*s/rad", "V*s/rad", {"min": 0.0, "min_inclusive": False}, False),  # both readings valid: ask
    ("1.000 substeps", 1000.0, None, None, {"min": 1, "max": 10000}, False),  # both valid: ask
    ("1.500 A", 1.5, "A", "A", None, False),                                     # no range known: ask
])
def test_range_disambiguation(text, value, unit, field_unit, rng, accepted):
    led = _ledger(text)
    src = led.user_quantity(value, unit, field_unit, field_range=rng)
    assert (src is not None) is accepted
    if accepted:
        assert "read as" in src and "outside the valid range" in src


def test_in_range_bounds():
    assert nb.in_range(1.0, {"min": 0.0, "max": 1.0, "max_inclusive": True})
    assert not nb.in_range(1.0, {"min": 0.0, "max": 1.0, "max_inclusive": False})
    assert not nb.in_range(0.0, {"min": 0.0, "min_inclusive": False})
    assert nb.in_range(5.0, None) and nb.in_range(5.0, {})
