"""Buck surrogate and SPADE/BuckSurrogate-v0 (expansion task E5, decision D3)."""

import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import driveflow.envs  # noqa: F401 -- registers SPADE/*
from driveflow.envs import registry
from driveflow.envs import surrogate_env as se
from driveflow.sim.buck_system import BuckParams


def test_split_is_by_condition_and_fixed():
    splits = {(v, r): se.condition_split(v, r) for v in se.VIN_GRID for r in se.R_GRID}
    assert sum(s == "train" for s in splits.values()) == 12
    assert {k for k, s in splits.items() if s != "train"} == set(se.SPLIT)
    assert all(se.condition_split(v, r) == "extrapolation" for v, r in se.EXTRAPOLATION)


def test_zoh_family_recovers_the_native_parameters_from_a_wrong_start():
    records = [r for r in se.build_dataset(0) if r["split"] == "train"][:4]
    fit = se.fit_zoh(records)
    nom = BuckParams()
    for k in ("l_h", "c_f", "r_l_ohm"):
        assert fit[k] == pytest.approx(getattr(nom, k), rel=1e-6)


def test_mlp_plant_interface_and_domain_check():
    sur, _ = se.latest_surrogate()
    p = se.make_plant(sur, "mlp", 48.0, 12.0)
    s0 = p.reset()
    s1 = p.simulate(0.5)
    assert (s0.i_l, s0.v_out) == (0.0, 0.0) and np.isfinite([s1.i_l, s1.v_out]).all() and s1.i_l > 0
    p.set_load(6.0)
    assert p.load_resistance_ohm == 6.0
    assert se.domain_violations(sur["domain"], 48.0, 12.0, 0.5) == []
    v = se.domain_violations(sur["domain"], 72.0, 40.0, 0.95)
    assert len(v) == 3 and "v_in_v" in v[0]


def test_registered_and_contract():
    assert registry.is_registered("SPADE/BuckSurrogate-v0")
    env = gym.make("SPADE/BuckSurrogate-v0")
    check_env(env.unwrapped)
    d = registry.describe("SPADE/BuckSurrogate-v0")
    assert d["source"] == "SIMULATED · SURROGATE" and d["surrogate"]["family"] in se.FAMILIES and "fit_error" in d["surrogate"]
    assert d["observation"]["shape"] == (4,) and d["action"]["names"] == ("duty_cycle",)


@pytest.mark.parametrize("family", se.FAMILIES)
def test_env_flags_out_of_domain_and_uses_the_surrogate(family):
    env = gym.make("SPADE/BuckSurrogate-v0", family=family, vin_v=72.0, vout_ref_range=(5.0, 40.0)).unwrapped
    _, info = env.reset(seed=1, options={"load_resistance_ohm": 12.0})
    assert info["surrogate_family"] == family and any("v_in_v" in m for m in info["out_of_domain"])
    _, _, _, _, info = env.step(np.array([0.5]))
    assert info["out_of_domain"]
    assert type(env.plant).__name__ == {"zoh": "ZohSurrogatePlant", "mlp": "MlpSurrogatePlant"}[family]


def test_zoh_surrogate_env_matches_the_native_env_step_for_step():
    """The identified grey box equals the native plant here, so the two environments coincide;
    this is the expected result for noise-free data from the same model, not a fidelity claim."""
    a = gym.make("SPADE/Buck-v0").unwrapped
    b = gym.make("SPADE/BuckSurrogate-v0", family="zoh").unwrapped
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        act = rng.uniform(0, 1, size=1)
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
    np.testing.assert_allclose(oa, ob, rtol=1e-6, atol=1e-6)


def test_compare_condition_reports_errors_and_domain():
    sur, _ = se.latest_surrogate()
    c = se.compare_condition(sur, "mlp", 48.0, 12.0, 24.0, 6.0)
    assert c["status"] == "ok" and c["rmse_v_out_v"] >= 0 and len(c["native"]["t"]) == len(c["surrogate"]["t"]) == 1000
    out = se.compare_condition(sur, "zoh", 48.0, 40.0, 24.0, None)
    assert out["out_of_domain"] and out["rmse_v_out_v"] == pytest.approx(0.0, abs=1e-9)


def test_report_records_the_default_family_rule():
    from pathlib import Path

    rep = json.loads((Path(__file__).resolve().parents[2] / "docs" / "audit" / "plecs" / "e5_surrogate_and_comparison.json").read_text())
    v = rep["validation"]
    assert rep["default_family"] == min(se.FAMILIES, key=lambda f: v[f]["rollout_rel_rmse_v_out"])
    assert {r["split"] for r in rep["rows"]} == {"train", "validation", "test", "extrapolation"}
