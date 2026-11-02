"""Round Assistant, task F0.3: Scenario.pmsm_motor_parameters (pmsm_v1). Default None must stay
bit for bit (tests/test_datagen_pmsm.py is not modified)."""

import math

import numpy as np
import pytest

from driveflow.control.classical.pmsm_foc import MOTOR_PARAMS, mtpa_id_iq, torque_of
from driveflow.datagen import Scenario, run_scenario

KEYS = ("i_d", "i_q", "torque_nm", "rpm", "u_d", "u_q")


def _run(controller="FOC_MTPA", i_s=150.0, duration_s=0.02, **kwargs):
    return run_scenario(Scenario("p", controller_type=controller, plant_config_id="pmsm_v1", pmsm_current_ref_a=i_s, duration_s=duration_s, **kwargs))


def _col(recs, key):
    return np.array([r[key] for r in recs], dtype=float)


@pytest.mark.parametrize("controller", ["FOC_MTPA", "FOC_naive"])
def test_explicit_stock_parameters_are_bit_identical_to_the_default(controller):
    default = _run(controller)
    explicit = _run(controller, pmsm_motor_parameters=dict(MOTOR_PARAMS))
    for key in KEYS:
        assert np.array_equal(_col(default, key), _col(explicit, key)), key


def test_the_parameters_reach_plant_controller_mtpa_and_torque():
    """Doubling psi_p doubles the magnet torque the run reports (torque_of with the merged dict)
    and moves the MTPA split (mtpa_id_iq with the merged dict)."""
    mp = {**MOTOR_PARAMS, "psi_p": 2 * MOTOR_PARAMS["psi_p"]}
    recs = _run(pmsm_motor_parameters={"psi_p": mp["psi_p"]})
    i_d_ref, i_q_ref = recs[0]["i_d_ref"], recs[0]["i_q_ref"]
    assert (i_d_ref, i_q_ref) == pytest.approx(mtpa_id_iq(150.0, mp=mp))
    assert (i_d_ref, i_q_ref) != pytest.approx(mtpa_id_iq(150.0))
    last = recs[-1]
    assert last["torque_nm"] == pytest.approx(torque_of(last["i_d"], last["i_q"], mp))


def test_u_supply_and_j_rotor_are_accepted():
    recs = _run(pmsm_motor_parameters={"u_supply": 400.0, "j_rotor": 0.05})
    assert np.all(np.isfinite(_col(recs, "i_q")))


def test_non_salient_machine_mtpa_equals_the_naive_policy():
    """l_d = l_q: no reluctance torque, MTPA is i_d = 0. Declared tolerance: 0.5 % of the torque
    over the last 5 ms (both runs track the same references, so they agree to rounding)."""
    nonsalient = {"l_d": 0.8e-3, "l_q": 0.8e-3}
    assert mtpa_id_iq(200.0, mp={**MOTOR_PARAMS, **nonsalient}) == (0.0, 200.0)
    mtpa = _col(_run("FOC_MTPA", 200.0, 0.03, pmsm_motor_parameters=nonsalient), "torque_nm")[-50:].mean()
    naive = _col(_run("FOC_naive", 200.0, 0.03, pmsm_motor_parameters=nonsalient), "torque_nm")[-50:].mean()
    assert mtpa == pytest.approx(naive, rel=5e-3)


def test_l_d_above_l_q_warns():
    with pytest.warns(UserWarning, match="MTPA policy assumes l_q >= l_d"):
        Scenario("p", controller_type="FOC_MTPA", plant_config_id="pmsm_v1", pmsm_current_ref_a=50.0, pmsm_motor_parameters={"l_d": 2e-3})


@pytest.mark.parametrize("params, match", [
    ({"p": 0}, r"\['p'\]"), ({"p": 2.5}, r"\['p'\]"), ({"p": True}, r"\['p'\]"),
    ({"l_d": 0.0}, "l_d"), ({"r_s": -1.0}, "r_s"), ({"psi_p": math.nan}, "psi_p"), ({"l_q": math.inf}, "l_q"),
    ({"ke": 1.0}, "unknown key"),
])
def test_invalid_parameters_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        Scenario("p", controller_type="FOC_MTPA", plant_config_id="pmsm_v1", pmsm_current_ref_a=50.0, pmsm_motor_parameters=params)


def test_other_plants_reject_the_field():
    with pytest.raises(ValueError, match="applies only to plant_config_id='pmsm_v1'"):
        Scenario("s", pmsm_motor_parameters={"p": 4})
