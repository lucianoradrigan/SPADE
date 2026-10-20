"""PMSM in the data pipeline (plant_config_id "pmsm_v1") -- TODO_CLAUDE_CODE.md P1."""

import numpy as np
import pytest

from driveflow.control.classical.pmsm_foc import mtpa_id_iq
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.export_parquet import PMSM_COLUMNS, SCHEMA_COLUMNS, records_to_dataframe


def _run(policy, i_s=200.0, seed=0, duration_s=0.03):
    return records_to_dataframe(run_scenario(Scenario(scenario_id="t", controller_type=policy, plant_config_id="pmsm_v1", pmsm_current_ref_a=i_s, seed=seed, duration_s=duration_s)))


def test_schema_is_the_shared_schema_plus_the_pmsm_columns():
    df = _run("FOC_MTPA")
    assert list(df.columns) == SCHEMA_COLUMNS + PMSM_COLUMNS
    assert len(df) == 300 and (df["label"] == "normal").all() and (df["plant_config_id"] == "pmsm_v1").all()
    assert df[["vc_real", "acc_x", "current_r"]].isna().all().all()


@pytest.mark.parametrize("policy", ["FOC_MTPA", "FOC_naive"])
def test_currents_reach_the_policy_reference(policy):
    df = _run(policy)
    i_d_ref, i_q_ref = (mtpa_id_iq(200.0) if policy == "FOC_MTPA" else (0.0, 200.0))
    assert df["i_d_ref"].iloc[0] == pytest.approx(i_d_ref) and df["i_q_ref"].iloc[0] == pytest.approx(i_q_ref)
    tail = df.iloc[-50:]
    assert np.hypot(tail["i_d"] - i_d_ref, tail["i_q"] - i_q_ref).max() < 0.02 * 200.0


def test_mtpa_gives_more_torque_than_id0_for_the_same_current():
    # Measured 2026-10-20 at |i_s| = 200 A: 119.9 N m (MTPA) vs. 59.7 N m (i_d = 0).
    assert _run("FOC_MTPA")["torque_nm"].iloc[-1] > 1.8 * _run("FOC_naive")["torque_nm"].iloc[-1]


def test_negative_reference_brakes():
    assert _run("FOC_MTPA", i_s=-150.0)["torque_nm"].iloc[-1] < 0


def test_reproducible():
    a, b = _run("FOC_MTPA", seed=3), _run("FOC_MTPA", seed=3)
    np.testing.assert_array_equal(a[PMSM_COLUMNS].to_numpy(), b[PMSM_COLUMNS].to_numpy())


@pytest.mark.parametrize("kwargs, error", [
    ({"controller_type": "FOC_MTPA", "plant_config_id": "pmsm_v1"}, ValueError),  # no current reference
    ({"controller_type": "FOC_MTPA", "plant_config_id": "pmsm_v1", "pmsm_current_ref_a": 450.0}, ValueError),
    ({"controller_type": "FOC_MTPA", "plant_config_id": "pmsm_v1", "pmsm_current_ref_a": float("nan")}, ValueError),
    ({"controller_type": "PI", "plant_config_id": "pmsm_v1", "pmsm_current_ref_a": 100.0}, NotImplementedError),
    ({"controller_type": "FOC_MTPA", "plant_config_id": "dc_perm_ex_v1"}, NotImplementedError),
])
def test_invalid_pmsm_scenarios_are_rejected(kwargs, error):
    with pytest.raises(error):
        Scenario(scenario_id="bad", **kwargs)
