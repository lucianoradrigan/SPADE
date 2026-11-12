"""V1 completion (M6.2, 2026-11-12): the rotor inertia j_rotor on the DC Motor and PMSM / FOC pages.

The control opens at the stock value and then sends nothing (the stock configuration and its hash are the ones from
before the control existed); a changed value reaches the simulator; invalid inputs show Scenario's message on the page.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.datagen import Scenario
from driveflow.viz.adapters import simulation_adapter as sim

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")


def _page(name: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.switch_page(f"pages/{name}.py").run()
    return at


@pytest.fixture
def scenarios(monkeypatch):
    seen = []

    def recording(**kwargs):
        sc = Scenario(**kwargs)
        seen.append(sc)
        return sc

    monkeypatch.setattr(sim, "Scenario", recording)
    return seen


def _dc_run(at: AppTest) -> AppTest:
    at.slider(key="dc_duration_slider").set_value(0.05).run()
    return at.button(key="dc_run_button").click().run()


def test_dc_rotor_inertia_opens_at_the_stock_value_and_sends_nothing(scenarios):
    at = _page("dc_motor")
    assert at.number_input(key="dc_j_rotor").value == pytest.approx(sim.dc_defaults()["params"]["j_rotor"]) == pytest.approx(0.025)
    at = _dc_run(at)
    assert not at.exception and not at.error
    assert "j_rotor" not in (scenarios[-1].motor_parameter_overrides or {})


def test_dc_rotor_inertia_reaches_the_scenario(scenarios):
    at = _page("dc_motor")
    at.number_input(key="dc_j_rotor").set_value(0.05).run()
    at = _dc_run(at)
    assert not at.exception and not at.error
    assert scenarios[-1].motor_parameter_overrides["j_rotor"] == pytest.approx(0.05)


def test_dc_invalid_value_shows_the_scenario_message(scenarios):
    at = _page("dc_motor")
    at.number_input(key="dc_j_rotor").set_value(0.05).run()
    at.number_input(key="dc_supply_v").set_value(50.0).run()
    at.number_input(key="dc_dc_limit_overrides_u").set_value(62.0).run()  # S-24: a voltage limit above the supply
    at = _dc_run(at)
    assert not at.exception
    assert at.error and "supply" in at.error[0].value.lower()


def test_pmsm_rotor_inertia_opens_at_the_stock_value_and_sends_nothing(monkeypatch):
    import driveflow.viz.views.pmsm as view

    calls, real = [], view.generate_mtpa_vs_naive_cloud
    monkeypatch.setattr(view, "generate_mtpa_vs_naive_cloud", lambda **kw: calls.append(kw) or real(**kw))
    at = _page("pmsm")
    assert at.number_input(key="pmsm_j_rotor").value == pytest.approx(view.PMSM_J_ROTOR) == pytest.approx(0.03883)
    at.button(key="pmsm_run_button").click().run()
    assert not at.exception and not at.error
    assert "j_rotor" not in calls[-1]["mp"] and "j_rotor" not in at.session_state["pmsm_result"]["params"]


def test_pmsm_rotor_inertia_reaches_the_generator(monkeypatch):
    import driveflow.viz.views.pmsm as view

    calls, real = [], view.generate_mtpa_vs_naive_cloud
    monkeypatch.setattr(view, "generate_mtpa_vs_naive_cloud", lambda **kw: calls.append(kw) or real(**kw))
    at = _page("pmsm")
    at.number_input(key="pmsm_j_rotor").set_value(0.1).run()
    at.button(key="pmsm_run_button").click().run()
    assert not at.exception and not at.error
    assert calls[-1]["mp"]["j_rotor"] == pytest.approx(0.1) and at.session_state["pmsm_result"]["params"]["j_rotor"] == pytest.approx(0.1)


def test_pmsm_rotor_inertia_is_validated_by_scenario():
    from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS, U_SUPPLY
    from driveflow.viz.views.pmsm import PMSM_J_ROTOR, pmsm_motor_parameters

    assert pmsm_motor_parameters({**MOTOR_PARAMS, "u_supply": U_SUPPLY, "j_rotor": PMSM_J_ROTOR}) == ({}, [])
    assert pmsm_motor_parameters({**MOTOR_PARAMS, "u_supply": U_SUPPLY, "j_rotor": 0.1})[0] == {"j_rotor": 0.1}
    with pytest.raises(ValueError) as expected:
        Scenario("x", plant_config_id="pmsm_v1", controller_type="FOC_MTPA", pmsm_current_ref_a=100.0, pmsm_motor_parameters={"j_rotor": -1.0})
    with pytest.raises(ValueError, match=str(expected.value)[:30].replace("(", r"\(").replace(")", r"\)").replace("[", r"\[").replace("*", r"\*")):
        pmsm_motor_parameters({**MOTOR_PARAMS, "u_supply": U_SUPPLY, "j_rotor": -1.0})
