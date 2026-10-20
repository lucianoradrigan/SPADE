"""S-09 (TODO list follow-up): optional per-axis converter voltage limit on the VSC Scenario, and
the applied command recorded in every VSC record."""

import numpy as np
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.export_parquet import SCHEMA_COLUMNS, VSC_COLUMNS, records_to_dataframe
from driveflow.sim.vsc_params import UPSTREAM_CONVERTER_LIMIT_V

pytestmark = pytest.mark.control


def _run(magnitude_v, limit=None, duration_s=0.1):
    sc = Scenario(scenario_id="s09", controller_type="DPC", plant_config_id="vsc_dpc_v1", reference_magnitude_v=magnitude_v,
                  duration_s=duration_s, vsc_voltage_limit_v=limit)
    return records_to_dataframe(run_scenario(sc))


def _rmse(df):
    e = np.hypot(df["vc_real"] - df["v_ref_real"], df["vc_imag"] - df["v_ref_imag"]).to_numpy()[50:]
    return float(np.sqrt(np.mean(e**2)))


def test_vsc_records_carry_the_applied_command():
    df = _run(50.0)
    assert list(df.columns) == SCHEMA_COLUMNS + VSC_COLUMNS
    assert not df["v_o_saturated"].any()


def test_default_has_no_limit_and_needs_more_than_the_real_converter_at_150_v():
    df = _run(150.0)
    # 150 V at 8 ohm needs ~189 V of converter voltage in steady state (S-09).
    assert np.abs(df[["v_o_real", "v_o_imag"]].to_numpy()[50:]).max() > UPSTREAM_CONVERTER_LIMIT_V


def test_limit_clips_every_axis_and_costs_tracking_only_where_it_is_active():
    limited = _run(150.0, limit=UPSTREAM_CONVERTER_LIMIT_V)
    assert np.abs(limited[["v_o_real", "v_o_imag"]].to_numpy()).max() <= UPSTREAM_CONVERTER_LIMIT_V
    assert limited["v_o_saturated"].iloc[50:].mean() > 0.1
    assert _rmse(limited) > 2 * _rmse(_run(150.0))
    nominal_limited, nominal = _run(50.0, limit=UPSTREAM_CONVERTER_LIMIT_V), _run(50.0)
    assert _rmse(nominal_limited) == pytest.approx(_rmse(nominal), rel=0.05)


def test_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        Scenario(scenario_id="bad", controller_type="DPC", plant_config_id="vsc_dpc_v1", vsc_voltage_limit_v=0.0)


def test_vsc_page_warns_and_limits():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")
    at = AppTest.from_file(app, default_timeout=300)
    at.run()
    at.switch_page("pages/vsc_dpc.py").run()
    at.slider(key="dpc_magnitude_slider").set_value(150.0).run()
    at.slider(key="dpc_duration_slider").set_value(0.05).run()
    at.button(key="dpc_run_button").click().run()
    assert not at.exception
    assert any("could not track this reference" in w.value for w in at.warning)
    at.toggle(key="dpc_voltage_limit").set_value(True).run()
    at.button(key="dpc_run_button").click().run()
    assert not at.exception
    assert any("clipped the command" in w.value for w in at.warning)
