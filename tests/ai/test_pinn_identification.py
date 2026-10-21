"""ai/pinn/identification.py (expansion task E10b): windowing, both estimators on simulated runs with
known parameters, and the bootstrap interval."""

import numpy as np
import pandas as pd
import pytest

from driveflow.ai.pinn import identification as idn
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import _VSC_R_OHM


@pytest.fixture(scope="module")
def degraded_run():
    return pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="pid", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.1, vsc_capacitor_scale=0.6)))


def test_window_alignment(degraded_run):
    d = idn.from_dataframe(degraded_run, _VSC_R_OHM, 0.06, 0.02)
    k0 = int(np.searchsorted(degraded_run["timestamp_s"].to_numpy(), 0.06 - 1e-12))
    assert d.n == 200 and d.t[0] == degraded_run["timestamp_s"].iloc[k0]
    assert d.x0_v[0] == degraded_run["vc_real"].iloc[k0 - 1] and d.v_o[0, 1] == degraded_run["v_o_imag"].iloc[k0]
    np.testing.assert_array_equal(d.i_load()[1], d.v_c[0] / _VSC_R_OHM)


@pytest.mark.parametrize("kwargs, msg", [
    (dict(start_s=0.0, length_s=0.02), "after the first sample"),
    (dict(start_s=0.09, length_s=0.02), "window needs"),
    (dict(start_s=0.06, length_s=0.001), "window needs"),
])
def test_bad_windows_are_rejected(degraded_run, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        idn.from_dataframe(degraded_run, _VSC_R_OHM, **kwargs)


def test_missing_columns_and_bad_load(degraded_run):
    with pytest.raises(ValueError, match="missing VSC column"):
        idn.from_dataframe(degraded_run.drop(columns=["v_o_real"]), _VSC_R_OHM, 0.06, 0.02)
    with pytest.raises(ValueError, match="load resistance"):
        idn.from_dataframe(degraded_run, 0.0, 0.06, 0.02)
    bad = degraded_run.copy()
    bad.loc[650, "vc_real"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        idn.from_dataframe(bad, _VSC_R_OHM, 0.06, 0.02)


def test_least_squares_is_exact_on_noise_free_data(degraded_run):
    r = idn.identify(idn.from_dataframe(degraded_run, _VSC_R_OHM, 0.06, 0.02), "least_squares", n_bootstrap=20)
    truth = {"L": idn.NOMINAL["L"], "R_f": idn.NOMINAL["R_f"], "C": 0.6 * idn.NOMINAL["C"]}
    for p in idn.PARAMS:
        assert r["params"][p]["estimate"] == pytest.approx(truth[p], rel=1e-6)
    assert r["uncertainty_method"] == idn.UNCERTAINTY_METHOD and r["n_bootstrap"] == 20


def test_pinn_recovers_degraded_capacitor(degraded_run):
    """Noise-free C x0.6 run: the PINN, started AT nominal, moves C to within 5 % of the truth;
    L and R_f stay within 5 % (measured: -2 %, -2 %, +0.1 %)."""
    r = idn.identify(idn.from_dataframe(degraded_run, _VSC_R_OHM, 0.06, 0.02), "pinn", n_bootstrap=2, steps=3000)
    assert r["params"]["C"]["estimate"] == pytest.approx(0.6 * idn.NOMINAL["C"], rel=0.05)
    assert r["params"]["L"]["estimate"] == pytest.approx(idn.NOMINAL["L"], rel=0.05)
    assert r["params"]["R_f"]["estimate"] == pytest.approx(idn.NOMINAL["R_f"], rel=0.05)
    for p in idn.PARAMS:
        v = r["params"][p]
        assert np.isfinite(v["low"]) and v["low"] <= v["high"]
    assert r["physics_loss"] < 0.05


def test_noise_is_relative_and_seeded(degraded_run):
    d = idn.from_dataframe(degraded_run, _VSC_R_OHM, 0.06, 0.02)
    a = idn.add_noise(d, 5.0, np.random.default_rng(3))
    b = idn.add_noise(d, 5.0, np.random.default_rng(3))
    np.testing.assert_array_equal(a.v_c, b.v_c)
    np.testing.assert_array_equal(a.v_o, d.v_o)  # the command is known exactly
    rel = np.std(a.v_c - d.v_c) / np.sqrt(np.mean(d.v_c ** 2))
    assert 0.04 < rel < 0.06
    assert idn.add_noise(d, 0.0, np.random.default_rng(0)) is d


def test_block_bootstrap_weights_cover_n():
    w = idn._block_bootstrap_weights(200, 20, np.random.default_rng(0))
    assert w.sum() == 200 and w.min() >= 0


def test_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        idn.identify(None, "kalman")
