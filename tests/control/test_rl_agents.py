"""DQN / DDPG in Keras, the policy controller and the benchmark (expansion task E8)."""

import json

import gymnasium as gym
import numpy as np
import pytest

import driveflow.envs  # noqa: F401
from driveflow.control.rl import benchmark as bm
from driveflow.control.rl.common import ActionDiscretizer, ReplayBuffer
from driveflow.control.rl.ddpg import DDPGAgent, DDPGConfig
from driveflow.control.rl.dqn import DQNAgent, DQNConfig
from driveflow.control.rl.policy_controller import PolicyController

pytestmark = pytest.mark.control
#: Declared reproducibility tolerance: same seed, same machine -> episode rewards equal to 1e-9
#: (deterministic TF ops on CPU).
REPRO_TOL = 1e-9


def _buck(**kw):
    return gym.make("SPADE/Buck-v0", duration_s=0.004, **kw)  # 200-step episodes keep the test short


def test_discretizer_grid_and_record():
    d = ActionDiscretizer.grid([0.0], [1.0], 5)
    assert d.n == 5 and d.action(4).tolist() == [1.0] and d.to_dict()["levels"][2] == [0.5]
    d2 = ActionDiscretizer.grid([-1, -1], [1, 1], 3)
    assert d2.n == 9 and d2.action(0).tolist() == [-1.0, -1.0]


def test_replay_buffer_wraps():
    b = ReplayBuffer(3, 2, 1, seed=0)
    for k in range(5):
        b.add([k, k], [k], k, [k, k], False)
    assert b.n == 3 and set(b.obs[:, 0].tolist()) == {3.0, 4.0, 2.0}


@pytest.mark.parametrize("algorithm", ["dqn", "ddpg"])
def test_same_seed_same_curve(algorithm):
    def run():
        env = _buck()
        if algorithm == "dqn":
            ag = DQNAgent(4, ActionDiscretizer.grid([0], [1], 5), DQNConfig(episodes=3, warmup_steps=100, eval_every=3, batch=32), seed=7)
        else:
            ag = DDPGAgent(4, [0.0], [1.0], DDPGConfig(episodes=3, warmup_steps=100, eval_every=3, batch=32), seed=7)
        return ag.train(env), ag
    (h1, a1), (h2, _) = run(), run()
    np.testing.assert_allclose(h1.episode_reward, h2.episode_reward, rtol=0, atol=REPRO_TOL)
    np.testing.assert_allclose(h1.eval_reward, h2.eval_reward, rtol=0, atol=REPRO_TOL)
    assert len(h1.episode_reward) == 3 and len(h1.eval_reward) == 1


@pytest.mark.parametrize("algorithm", ["dqn", "ddpg"])
def test_policy_controller_matches_the_agent(algorithm):
    env = _buck()
    if algorithm == "dqn":
        ag = DQNAgent(4, ActionDiscretizer.grid([0], [1], 5), DQNConfig(episodes=1, warmup_steps=50, eval_every=1, batch=16), seed=0)
    else:
        ag = DDPGAgent(4, [0.0], [1.0], DDPGConfig(episodes=1, warmup_steps=50, eval_every=1, batch=16), seed=0)
    ag.train(env)
    pc = PolicyController(json.loads(json.dumps(ag.policy_dict())))
    rng = np.random.default_rng(0)
    for _ in range(20):
        o = rng.normal(size=4).astype(np.float32)
        np.testing.assert_allclose(pc.control(o), ag.act(o), rtol=1e-5, atol=1e-6)


def test_benchmark_runs_every_controller_on_the_same_grid():
    env = _buck()
    res = bm.run_grid(env, {"PI": bm.buck_pi, "zero": lambda e: (lambda o: np.array([0.0]))}, bm.BUCK_GRID[:2])
    s = bm.summarize(res)
    assert [r["options"] for r in res["PI"]] == [r["options"] for r in res["zero"]]
    assert s["PI"]["median_tracking_rmse"] < 1.0 and s["zero"]["median_tracking_rmse"] > 5.0  # zero duty never reaches v_ref


def test_dqn_needs_a_discretizer_in_the_job_spec():
    from driveflow.jobs.service import JobService
    from driveflow.runs.schemas import ExperimentSpec, SpadeError

    with pytest.raises(SpadeError, match="discrete_levels"):
        JobService().submit(ExperimentSpec.create("rl_training", algorithm="dqn", seed=0, params={"env_id": "SPADE/Buck-v0", "episodes": 2}))
    with pytest.raises(SpadeError, match="supports"):
        JobService().submit(ExperimentSpec.create("rl_training", algorithm="ddpg", seed=0, params={"env_id": "SPADE/DcMotor-v0", "episodes": 2}))


@pytest.mark.slow
def test_training_job_writes_checkpoints_playback_and_benchmark(tmp_path):
    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker
    from driveflow.runs.schemas import ExperimentSpec
    from driveflow.runs.store import workspace_root

    s = JobService()
    job = s.submit(ExperimentSpec.create("rl_training", algorithm="ddpg", seed=0,
                                         params={"env_id": "SPADE/Buck-v0", "episodes": 2, "eval_every": 1, "env_kwargs": {"duration_s": 0.004}}))
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    run = s.run_for(s.get(job.job_id))
    names = {a["path"].split("/", 2)[-1] for a in run.artifacts}
    assert run.status == "completed" and {"policy.json", "history.json", "benchmark.json"} <= names
    assert any("checkpoints/policy_ep0001.json" in a["path"] for a in run.artifacts) and any("playback/eval_ep0002.json" in a["path"] for a in run.artifacts)
    bench = json.loads(s.store.artifact_path(next(a for a in run.artifacts if a["path"].endswith("benchmark.json"))).read_text())
    assert set(bench["summary"]) == {"RL (DDPG)", "PI (BuckVoltageController)"} and bench["status"] == "policy candidate -- not a default"
