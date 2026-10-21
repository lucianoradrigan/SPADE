"""E1: the Gymnasium wrappers of the native plants keep the Gymnasium contract and add no physics.

The equivalence tests drive the environment with the very controller datagen.runner uses (PI
cascade for the DC motor, scheduled DPC for the VSC), computed from the environment's own plant
state, and require the trajectory to equal run_scenario's records exactly (==, not approx).
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import driveflow.envs  # noqa: F401  (registers the IDs)
from driveflow.control.classical import PICascadeController
from driveflow.control.dpc.controller import DEFAULT_SCHEDULE_REGION, ScheduledDpcController
from driveflow.datagen.runner import _DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH, run_scenario
from driveflow.datagen.scenario import Scenario
from driveflow.envs import registry
from driveflow.envs.native import BuckEnv, DcMotorEnv, VscEnv
from driveflow.control.classical.buck_pi import BuckVoltageController
from driveflow.envs.reward import RewardTerm, evaluate, parse_terms

pytestmark = pytest.mark.simulation

ENV_IDS = ["SPADE/DcMotor-v0", "SPADE/Vsc-v0", "SPADE/Buck-v0", "SPADE/Pmsm-v0"]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_check_env_passes(env_id):
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_registered_ids_and_verify(env_id):
    assert env_id in gym.registry and registry.is_registered(env_id)
    cap = registry.verify(env_id)
    assert cap.can_step_interactively and cap.can_reset_episode and cap.can_simulate_batch
    assert not cap.can_train_rl  # no trainer before E8
    assert cap.validated_at is not None
    assert any("check_env passed" in m for m in cap.diagnostic_messages)


def test_planned_ids_are_not_registered():
    for env_id in registry.PLANNED:
        assert not registry.is_registered(env_id)
        with pytest.raises(KeyError):
            registry.snippet(env_id)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_snippet_runs(env_id):
    code = registry.snippet(env_id).replace("range(100)", "range(5)")
    exec(compile(code, "<snippet>", "exec"), {})


def test_dc_motor_env_reproduces_run_scenario_bit_for_bit():
    duration, seed, omega_ref = 0.2, 3, 150.0
    records = run_scenario(Scenario(scenario_id="eq", plant_config_id="dc_perm_ex_v1", controller_type="PI", omega_ref_rad_s=omega_ref,
                                    duration_s=duration, seed=seed))
    env = DcMotorEnv(omega_ref_rad_s=omega_ref, duration_s=duration)
    env.reset(seed=seed)
    controller = PICascadeController(env.plant)
    controller.reset()
    rows, done = [], False
    while not done:
        action = controller.control(env.plant_state, omega_ref)
        _, _, terminated, truncated, info = env.step(action)
        assert not info["action_clipped"]
        rows.append(info)
        done = terminated or truncated
    assert truncated and not terminated and info["truncation"] == "horizon"
    assert len(rows) == len(records)
    for rec, info in zip(records, rows):
        assert info["rpm"] == rec["rpm"]
        assert info["current_a"] == rec["current_r"]
        assert info["voltage_v"] == rec["voltage_v"]
        assert info["torque_nm"] == rec["torque_nm"]
        assert info["t_s"] == pytest.approx(rec["timestamp_s"], abs=1e-15)


@pytest.mark.parametrize("r_ohm,magnitude,seed", [(8.0064, 50.0, 0), (20.0, 30.0, 45)])
def test_vsc_env_reproduces_run_scenario_bit_for_bit(r_ohm, magnitude, seed):
    duration = 0.05
    records = run_scenario(Scenario(scenario_id="eq", plant_config_id="vsc_dpc_v1", controller_type="DPC", load_resistance_ohm=r_ohm,
                                    reference_magnitude_v=magnitude, duration_s=duration, seed=seed))
    env = VscEnv(reference_magnitude_v=magnitude, load_resistance_ohm=r_ohm, duration_s=duration)
    env.reset(seed=seed)
    controller = ScheduledDpcController(_DPC_NOMINAL_WEIGHTS_PATH, _DPC_WEIGHTS_PATH, r_ohm, DEFAULT_SCHEDULE_REGION)
    controller.reset()
    rows, k, done = [], 0, False
    while not done:
        action = controller.control(env.plant_state, env.reference, k)
        _, _, terminated, truncated, info = env.step(action)
        assert not info["action_clipped"]
        rows.append(info)
        k += 1
        done = terminated or truncated
    assert len(rows) == len(records)
    for rec, info in zip(records, rows):
        for key in ("vc_real", "vc_imag", "i_f_real", "i_f_imag", "v_ref_real", "v_ref_imag"):
            assert info[key] == rec[key], key
        assert float(info["action_applied"][0]) == rec["v_o_real"]
        assert float(info["action_applied"][1]) == rec["v_o_imag"]


def test_vsc_observation_carries_next_reference_and_units():
    env = VscEnv(duration_s=0.001)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
    vr0, vi0 = env.reference.at_step(0)
    assert obs[4] == np.float32(vr0 / 100.0) and obs[5] == np.float32(vi0 / 100.0)
    obs, *_ = env.step([0.0, 0.0])
    vr1, _ = env.reference.at_step(1)
    assert obs[4] == np.float32(vr1 / 100.0)
    md = env.metadata
    assert len(md["observation_names"]) == len(md["observation_units"]) == len(md["observation_scales"]) == 6
    assert md["action_units"] == ("V", "V")


def test_same_seed_same_episode_and_random_references():
    env = DcMotorEnv(omega_ref_rad_s=None, duration_s=0.01)
    _, a = env.reset(seed=11)
    _, b = env.reset(seed=11)
    _, c = env.reset(seed=12)
    assert a["omega_ref_rad_s"] == b["omega_ref_rad_s"] != c["omega_ref_rad_s"]
    assert 50.0 <= c["omega_ref_rad_s"] <= 300.0
    _, d = env.reset(seed=11, options={"omega_ref_rad_s": 80.0})
    assert d["omega_ref_rad_s"] == 80.0

    vsc = VscEnv(reference_magnitude_v=None, load_resistance_ohm=None, duration_s=0.01)
    _, x = vsc.reset(seed=5)
    _, y = vsc.reset(seed=5)
    assert (x["reference_magnitude_v"], x["load_resistance_ohm"]) == (y["reference_magnitude_v"], y["load_resistance_ohm"])
    assert 4.0 <= x["load_resistance_ohm"] <= 1000.0 and 25.0 <= x["reference_magnitude_v"] <= 100.0


def test_out_of_bounds_action_is_clipped_and_logged_and_nan_rejected():
    env = DcMotorEnv(duration_s=0.01)
    env.reset(seed=0)
    *_, info = env.step([1.7])
    assert info["action_clipped"] and info["action_applied"][0] == 1.0
    with pytest.raises(ValueError):
        env.step([np.nan])
    vsc = VscEnv(voltage_limit_v=100.0, duration_s=0.01)
    vsc.reset(seed=0)
    *_, info = vsc.step([500.0, -20.0])
    assert info["action_clipped"] and info["action_applied"].tolist() == [100.0, -20.0]


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        DcMotorEnv().step([0.5])


def test_limit_violation_terminates_with_reason():
    # an unstable operating point: VSC load below R* ~= 3.37 ohm with zero input diverges
    env = VscEnv(load_resistance_ohm=0.5, voltage_limit_v=10.0, duration_s=0.5)
    env.reset(seed=0, options={"reference_magnitude_v": 50.0})
    env.plant.state.vc_real = 1.0  # a small disturbance; the open-loop pole is ~12, so it grows
    terminated = truncated = False
    while not (terminated or truncated):
        *_, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated and info["termination"] == "limit_violation:capacitor_voltage"


def test_reward_terms_breakdown_and_validation():
    env = DcMotorEnv(duration_s=0.01, reward_terms=[{"kind": "tracking", "weight": 2.0}])
    env.reset(seed=0)
    _, reward, *_, info = env.step([0.3])
    assert set(info["reward_terms"]) == {"tracking"}
    assert reward == pytest.approx(-sum(info["reward_terms"].values()))
    assert info["reward_terms"]["tracking"] == pytest.approx(2.0 * info["signals"]["tracking_error"] ** 2)
    with pytest.raises(ValueError):
        parse_terms([{"kind": "tracking", "weight": 0.0}])
    with pytest.raises(ValueError):
        parse_terms([{"kind": "tracking"}, {"kind": "tracking"}])
    with pytest.raises(ValueError):
        RewardTerm("efficiency")
    with pytest.raises(ValueError):
        RewardTerm("tracking", weight=float("nan"))
    r, b = evaluate(parse_terms([RewardTerm("overcurrent", 1.0, threshold=0.8)]), {"current_fraction": 0.5})
    assert r == 0.0 and b == {"overcurrent": 0.0}


def test_describe_matches_instance():
    d = registry.describe("SPADE/DcMotor-v0")
    assert d["observation"]["shape"] == (5,) and d["action"]["low"] == [0.0] and d["action"]["high"] == [1.0]
    assert d["horizon_steps"] == 5000 and d["limits"]["current_a"] == 210.0
    assert [t["kind"] for t in d["reward_terms"]] == ["tracking", "overcurrent", "action_rate"]


@pytest.mark.parametrize("step", [None, (3.0, 0.01)])
def test_buck_env_reproduces_run_scenario_bit_for_bit(step):
    duration, vref, r = 0.02, 18.0, 8.0
    kw = {} if step is None else {"buck_load_step_ohm": step[0], "buck_load_step_time_s": step[1]}
    records = run_scenario(Scenario(scenario_id="eq", plant_config_id="buck_v1", buck_vout_ref_v=vref, load_resistance_ohm=r, duration_s=duration, **kw))
    env = BuckEnv(vout_ref_v=vref, load_resistance_ohm=r, duration_s=duration,
                  load_step_ohm=None if step is None else step[0], load_step_time_s=None if step is None else step[1])
    env.reset(seed=0)
    controller = BuckVoltageController(env.plant)
    rows, done = [], False
    while not done:
        _, _, terminated, truncated, info = env.step([controller.control(env.plant_state, vref)])
        rows.append(info)
        done = terminated or truncated
    assert truncated and not terminated and len(rows) == len(records) == 1000
    for rec, info in zip(records, rows):
        assert info["v_out_v"] == rec["buck_v_out"] and info["i_l_a"] == rec["buck_i_l"]
        assert info["action_applied"][0] == rec["buck_duty"] and info["load_resistance_ohm"] == rec["buck_load_ohm"]


def test_buck_env_random_conditions_and_current_termination():
    env = BuckEnv(vout_ref_v=None, load_resistance_ohm=None, duration_s=0.005)
    _, a = env.reset(seed=4)
    _, b = env.reset(seed=4)
    assert (a["v_ref_v"], a["load_resistance_ohm"]) == (b["v_ref_v"], b["load_resistance_ohm"])
    assert 5.0 <= a["v_ref_v"] <= 40.0 and 3.0 <= a["load_resistance_ohm"] <= 30.0
    # full duty into a near short circuit: the inductor current passes 2 x i_max and the episode ends
    env = BuckEnv(load_resistance_ohm=0.05, duration_s=0.05)
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        *_, terminated, truncated, info = env.step([1.0])
    assert terminated and info["termination"] == "limit_violation:inductor_current"


@pytest.mark.parametrize("controller,strategy,i_s,seed", [("FOC_MTPA", "mtpa", 250.0, 3), ("FOC_naive", "naive", -120.0, 0), ("FOC_MTPA", "mtpa", 390.0, 7)])
def test_pmsm_env_reproduces_run_scenario_bit_for_bit(controller, strategy, i_s, seed):
    """Round 2026-10-21, D10: the FOC current controller acting through SPADE/Pmsm-v0 gives exactly
    the i_d/i_q of run_scenario (same plant, same Euler substeps, same seed)."""
    from driveflow.control.classical.pmsm_foc import DqCurrentController

    ref = run_scenario(Scenario(scenario_id="pmsm_eq", plant_config_id="pmsm_v1", controller_type=controller, pmsm_current_ref_a=i_s, duration_s=0.03, seed=seed))
    env = gym.make("SPADE/Pmsm-v0", strategy=strategy).unwrapped
    _, info = env.reset(seed=seed, options={"current_ref_a": i_s})
    assert (env.i_d_ref, env.i_q_ref) == (ref[0]["i_d_ref"], ref[0]["i_q_ref"])
    foc = DqCurrentController(env.plant)
    foc.reset()
    for k in range(env.n_steps):
        a, _ = foc.control(env.plant_state, env.plant, env.i_d_ref, env.i_q_ref)
        _, _, term, trunc, info = env.step(a)
        assert (info["i_d_a"], info["i_q_a"], info["omega_rad_s"]) == (ref[k]["i_d"], ref[k]["i_q"], ref[k]["omega_rad_s"])
        assert info["u_d_v"] == ref[k]["u_d"] and info["u_q_v"] == ref[k]["u_q"]
    assert trunc and not term and env.n_steps == len(ref)


def test_pmsm_env_validation_and_limits():
    from driveflow.envs.native import PmsmEnv

    with pytest.raises(ValueError, match="strategy"):
        PmsmEnv(strategy="fw")
    with pytest.raises(ValueError, match="I_s|i_s"):
        PmsmEnv(current_ref_a=500.0)
    env = PmsmEnv(duration_s=0.002)
    _, info = env.reset(seed=1, options={"current_ref_a": 100.0})
    _, _, _, _, info = env.step(np.array([5.0, -5.0]))
    assert info["action_clipped"] and info["action_applied"].tolist() == [1.0, -1.0]
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0]))
