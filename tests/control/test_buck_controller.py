"""E6: closed-loop performance of BuckVoltageController on the default averaged buck."""

import numpy as np
import pytest

from driveflow.control.classical.buck_pi import BuckVoltageController
from driveflow.datagen.export_parquet import BUCK_COLUMNS, records_to_dataframe
from driveflow.datagen.runner import run_scenario
from driveflow.datagen.scenario import Scenario

pytestmark = pytest.mark.control


def _run(**kw):
    df = records_to_dataframe(run_scenario(Scenario(scenario_id="b", plant_config_id="buck_v1", **kw)))
    return df["timestamp_s"].to_numpy(), df["buck_v_out"].to_numpy(), df


def test_startup_settles_and_tracks():
    t, v, df = _run(duration_s=0.01)
    outside = np.flatnonzero(np.abs(v - 24.0) > 0.02 * 24.0)
    settle = t[outside[-1] + 1] if len(outside) else 0.0
    assert settle < 2e-3, settle  # measured 1.32 ms
    assert v.max() < 24.0 * 1.05  # measured peak 24.94 V
    assert abs(v[-1] - 24.0) < 1e-3
    assert df["buck_duty"].between(0, 1).all() and (df["buck_i_l"].abs() <= 15.0 * 1.05).all()


def test_load_step_recovers():
    t, v, df = _run(duration_s=0.02, buck_load_step_ohm=3.0, buck_load_step_time_s=0.01)
    after = v[t > 0.01]
    assert after.min() > 24.0 * 0.85  # measured dip to 22.0 V for 6 -> 3 ohm
    assert abs(v[-1] - 24.0) < 1e-3
    assert (df["buck_load_ohm"][t > 0.01] == 3.0).all() and (df["buck_load_ohm"][t <= 0.01] == 6.0).all()


def test_buck_scenario_validation_and_columns():
    with pytest.raises(ValueError):
        Scenario(scenario_id="x", plant_config_id="buck_v1", buck_vout_ref_v=50.0)  # above V_in
    with pytest.raises(ValueError):
        Scenario(scenario_id="x", plant_config_id="buck_v1", buck_parameter_overrides={"bogus": 1.0})
    with pytest.raises(NotImplementedError):
        Scenario(scenario_id="x", plant_config_id="buck_v1", controller_type="DPC")
    _, _, df = _run(duration_s=0.001)
    assert list(df.columns[-len(BUCK_COLUMNS):]) == BUCK_COLUMNS and len(df) == 50
    assert df["rpm"].isna().all() and (df["label"] == "normal").all()


def test_existing_plants_export_no_buck_columns():
    for kw in ({"plant_config_id": "dc_perm_ex_v1"}, {"plant_config_id": "vsc_dpc_v1", "controller_type": "DPC"}):
        df = records_to_dataframe(run_scenario(Scenario(scenario_id="x", duration_s=0.001, **kw)))
        assert not set(BUCK_COLUMNS) & set(df.columns)


def test_controller_is_derived_from_plant_parameters():
    from driveflow.sim.buck_system import BuckParams, BuckSystem

    a = BuckVoltageController(BuckSystem(BuckParams()))
    b = BuckVoltageController(BuckSystem(BuckParams(l_h=440e-6, c_f=440e-6)))
    assert b.kp_i == pytest.approx(2 * a.kp_i) and b.kp_v == pytest.approx(2 * a.kp_v)
