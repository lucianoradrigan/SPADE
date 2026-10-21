"""Shared pieces of the RL agents -- expansion task E8: seeding, replay buffer, small MLPs, action
discretisation and the training-history record."""

from __future__ import annotations

import dataclasses
import json

import numpy as np


def seed_everything(seed: int) -> None:
    """Python, NumPy and TensorFlow seeds, and deterministic TF ops, so the same seed gives the same
    training curve on the same machine (the tolerance actually observed is reported, not assumed)."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # older TF: best effort
        pass


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.next = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.n = 0
        self.i = 0
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, done) -> None:
        self.obs[self.i], self.act[self.i], self.rew[self.i], self.next[self.i], self.done[self.i] = o, a, r, o2, float(done)
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next[idx], self.done[idx]


def mlp(inputs: int, outputs: int, hidden=(64, 64), out_activation=None, name="mlp"):
    import keras

    layers = [keras.Input((inputs,))] + [keras.layers.Dense(h, activation="relu") for h in hidden] + [keras.layers.Dense(outputs, activation=out_activation)]
    return keras.Sequential(layers, name=name)


@dataclasses.dataclass(frozen=True)
class ActionDiscretizer:
    """DQN's discrete action i -> the continuous action vector ``levels[i]`` of a Box space. Saved
    with the policy (spec Sec. 5 step 1: the transformation is recorded)."""

    levels: tuple

    @classmethod
    def grid(cls, low, high, n_per_dim: int) -> "ActionDiscretizer":
        low, high = np.atleast_1d(np.asarray(low, float)), np.atleast_1d(np.asarray(high, float))
        axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in zip(low, high)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(low))
        return cls(tuple(tuple(float(v) for v in row) for row in mesh))

    @property
    def n(self) -> int:
        return len(self.levels)

    def action(self, i: int) -> np.ndarray:
        return np.asarray(self.levels[int(i)], dtype=np.float64)

    def to_dict(self) -> dict:
        return {"kind": "grid", "levels": [list(l) for l in self.levels]}


@dataclasses.dataclass
class TrainingHistory:
    episode_reward: list = dataclasses.field(default_factory=list)
    episode_length: list = dataclasses.field(default_factory=list)
    episode_violation: list = dataclasses.field(default_factory=list)  # termination reason or None
    loss: list = dataclasses.field(default_factory=list)               # DQN: Q loss; DDPG: critic loss
    actor_loss: list = dataclasses.field(default_factory=list)          # DDPG only
    eval_episode: list = dataclasses.field(default_factory=list)        # episode index of each evaluation
    eval_reward: list = dataclasses.field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), default=float)


def run_episode(env, act, seed: int | None = None, options: dict | None = None, record: bool = False) -> dict:
    """One episode with act(obs) -> action. Returns the total reward, length, termination reason
    and, if record, the per-step info signals (the 'playback' of an executed episode)."""
    obs, info = env.reset(seed=seed, options=options)
    total, steps, reason, trace = 0.0, 0, None, []
    while True:
        a = act(obs)
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        steps += 1
        if record:
            trace.append({"t_s": info.get("t_s"), "reward": float(r), "action": [float(x) for x in np.atleast_1d(info.get("action_applied", a))],
                          **{k: float(v) for k, v in info.items() if isinstance(v, (int, float)) and k not in ("t_s", "step")}})
        if term or trunc:
            reason = info.get("termination") if term else None
            break
    return {"reward": total, "length": steps, "termination": reason, "trace": trace}
