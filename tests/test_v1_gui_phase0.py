"""V1 (round B, 2026-11-09): the Phase 0 simulator options on the DC Motor and PMSM / FOC pages.

Each new control reaches the Scenario; with every control at its stock value the run is exactly the run
of the page before V1 (no default changes); invalid values show Scenario's own message.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from driveflow.datagen import Scenario, run_scenario
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
    """Every Scenario the DC Motor page builds, in order."""
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


# ----------------------------------------------------------------------------------- DC Motor
def test_the_stock_options_send_nothing_new(scenarios):
    at = _dc_run(_page("dc_motor"))
    assert not at.exception
    sc = scenarios[-1]
    assert sc.plant_config_id == "dc_perm_ex_v2"
    assert (sc.load_torque_nm, sc.dc_supply_v, sc.dc_limit_overrides, sc.dc_nominal_overrides) == (None, None, None, None)


def test_with_stock_options_the_run_equals_the_run_without_them():
    base = {"controller": "PI", "fault_label": "healthy", "control_mode": "speed", "setpoint": 300.0, "duration_s": 0.05, "seed": 0,
            "electrical_severity": 8.0, "mechanical_severity": None, "electrical_noise_pct": 0.0, "mechanical_noise_pct": 0.0,
            "motor": {k: sim.dc_defaults()["params"][k] for k in ("r_a", "l_a", "psi_e")}, "pi_setpoint_weight": None}
    stock = {"plant": "dc_perm_ex_v2", "load_torque_nm": 0.0, "dc_supply_v": 65.0,
             "dc_limit_overrides": dict(sim.dc_defaults()["limits"]), "dc_nominal_overrides": dict(sim.dc_defaults()["nominal"])}
    with_options = sim.run_dc({**base, **stock}, {})
    # The page's Scenario before V1 (simulation_adapter.run_dc on 2026-11-08):
    before = pd.DataFrame.from_records(run_scenario(Scenario(
        scenario_id="dashboard_run", plant_config_id="dc_perm_ex_v2", controller_type="PI", fault_type=None, duration_s=0.05, seed=0,
        electrical_severity=8.0, mechanical_severity=None, motor_parameter_overrides=base["motor"], electrical_noise_pct=0.0, mechanical_noise_pct=0.0,
        fault_order_override=None, extra_faults=[], omega_ref_rad_s=300.0)))
    pd.testing.assert_frame_equal(with_options, before)


def test_each_option_reaches_the_scenario(scenarios):
    at = _page("dc_motor")
    at.number_input(key="dc_load_torque_nm").set_value(2.5).run()
    at.number_input(key="dc_supply_v").set_value(70.0).run()
    at.number_input(key="dc_dc_limit_overrides_i").set_value(150.0).run()
    at.number_input(key="dc_dc_nominal_overrides_torque").set_value(15.0).run()
    _dc_run(at)
    assert not at.exception
    sc = scenarios[-1]
    assert sc.plant_config_id == "dc_perm_ex_v2"
    assert sc.load_torque_nm == 2.5 and sc.dc_supply_v == 70.0
    assert sc.dc_limit_overrides == {"i": 150.0} and sc.dc_nominal_overrides == {"torque": 15.0}
    assert at.session_state["dc_result"]["envelope"]["i_lim"] == 150.0


def test_v1_is_the_gem_faithful_mode_with_the_s21_warning(scenarios):
    at = _page("dc_motor")
    assert not any("S-21" in w.value for w in at.warning)
    at.selectbox(key="dc_plant_mode").set_value("dc_perm_ex_v1").run()
    assert any(w.value == sim.S21_V1_WARNING for w in at.warning)
    _dc_run(at)
    assert not at.exception and scenarios[-1].plant_config_id == "dc_perm_ex_v1"


def test_an_invalid_option_shows_the_scenario_message_and_runs_nothing(scenarios):
    at = _page("dc_motor")
    at.number_input(key="dc_supply_v").set_value(50.0).run()
    at.number_input(key="dc_dc_limit_overrides_u").set_value(62.0).run()  # above the 50 V supply (stock limit: 60 V)
    with pytest.raises(ValueError) as expected:
        Scenario("x", plant_config_id="dc_perm_ex_v2", dc_supply_v=50.0, dc_limit_overrides={"u": 62.0})
    _dc_run(at)
    assert not at.exception
    assert [e.value for e in at.error] == [str(expected.value)]
    assert "dc_result" not in at.session_state


# ----------------------------------------------------------------------------------- PMSM / FOC
def test_pmsm_stock_controls_keep_the_configuration():
    from driveflow.viz.views.pmsm import pmsm_motor_parameters
    from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS, U_SUPPLY

    assert pmsm_motor_parameters({**MOTOR_PARAMS, "u_supply": U_SUPPLY}) == ({}, [])
    at = _page("pmsm")
    at.button(key="pmsm_run_button").click().run()
    assert not at.exception and "u_supply" not in at.session_state["pmsm_result"]["params"]


def test_pmsm_supply_reaches_the_generator(monkeypatch):
    import driveflow.viz.views.pmsm as view

    calls = []
    real = view.generate_mtpa_vs_naive_cloud
    monkeypatch.setattr(view, "generate_mtpa_vs_naive_cloud", lambda **kw: calls.append(kw) or real(**kw))
    at = _page("pmsm")
    at.number_input(key="pmsm_u_supply").set_value(300.0).run()
    at.button(key="pmsm_run_button").click().run()
    assert not at.exception
    assert calls[-1]["u_supply"] == 300.0 and at.session_state["pmsm_result"]["params"]["u_supply"] == 300.0


def test_pmsm_messages_are_the_scenario_messages():
    from driveflow.viz.views.pmsm import pmsm_motor_parameters
    from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS, U_SUPPLY

    with pytest.raises(ValueError) as gui:
        pmsm_motor_parameters({**MOTOR_PARAMS, "u_supply": U_SUPPLY, "p": 0})
    with pytest.raises(ValueError) as scenario:
        Scenario("x", plant_config_id="pmsm_v1", controller_type="FOC_MTPA", pmsm_current_ref_a=100.0, pmsm_motor_parameters={"p": 0})
    assert str(gui.value) == str(scenario.value)
    _, warnings = pmsm_motor_parameters({**MOTOR_PARAMS, "u_supply": U_SUPPLY, "l_d": 2e-3})  # l_d > l_q
    assert len(warnings) == 1 and "l_d" in warnings[0]


def test_pmsm_page_shows_the_saliency_warning():
    at = _page("pmsm")
    at.number_input(key="pmsm_l_d").set_value(2e-3).run()
    assert not at.exception
    assert any("l_d" in w.value and "l_q" in w.value for w in at.warning)
