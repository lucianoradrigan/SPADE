"""Deep deterministic policy gradient for continuous actions -- expansion task E8.

Origin: Fund. ``7_Reinforcement_Learning/DDPG_buck_control.ipynb`` (PyTorch: actor, critic, target
networks, Ornstein-Uhlenbeck exploration noise, soft target updates, a duty-smoothness penalty).
Reimplemented in Keras; no code copied. The smoothness penalty is SPADE's own ``action_rate``
reward term (envs/reward.py), set in the reward editor instead of being hard-coded in the agent.

- The actor ends in tanh, scaled to the Box action bounds.
- The critic takes (obs, action) and outputs Q.
- Random actions for ``warmup_steps``, then the actor plus OU noise (σ as a fraction of the action
  range), clipped to the bounds.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from driveflow.control.rl.common import ReplayBuffer, TrainingHistory, run_episode, seed_everything


@dataclasses.dataclass
class DDPGConfig:
    episodes: int = 60
    gamma: float = 0.98
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    batch: int = 64
    buffer: int = 100_000
    warmup_steps: int = 2_000
    tau: float = 0.005
    noise_theta: float = 0.15
    noise_sigma: float = 0.2
    hidden: tuple = (64, 64)
    eval_every: int = 10
    eval_seeds: tuple = (1001, 1002)


class OUNoise:
    def __init__(self, dim: int, theta: float, sigma: float, rng):
        self.dim, self.theta, self.sigma, self.rng = dim, theta, sigma, rng
        self.x = np.zeros(dim)

    def reset(self):
        self.x = np.zeros(self.dim)

    def __call__(self):
        self.x += -self.theta * self.x + self.sigma * self.rng.standard_normal(self.dim)
        return self.x.copy()


class DDPGAgent:
    def __init__(self, obs_dim: int, low, high, config: DDPGConfig, seed: int = 0):
        import keras
        import tensorflow as tf

        seed_everything(seed)
        self.cfg, self.seed = config, seed
        self.low, self.high = np.asarray(low, np.float64), np.asarray(high, np.float64)
        act_dim = len(self.low)
        self.mid, self.half = (self.high + self.low) / 2, (self.high - self.low) / 2

        def actor():
            inp = keras.Input((obs_dim,))
            x = inp
            for h in config.hidden:
                x = keras.layers.Dense(h, activation="relu")(x)
            return keras.Model(inp, keras.layers.Dense(act_dim, activation="tanh")(x))

        def critic():
            o, a = keras.Input((obs_dim,)), keras.Input((act_dim,))
            x = keras.layers.Concatenate()([o, a])
            for h in config.hidden:
                x = keras.layers.Dense(h, activation="relu")(x)
            return keras.Model([o, a], keras.layers.Dense(1)(x))

        self.actor, self.actor_t, self.critic, self.critic_t = actor(), actor(), critic(), critic()
        self.actor_t.set_weights(self.actor.get_weights())
        self.critic_t.set_weights(self.critic.get_weights())
        self.opt_a, self.opt_c = keras.optimizers.Adam(config.lr_actor), keras.optimizers.Adam(config.lr_critic)
        self.rng = np.random.default_rng(seed)
        self.noise = OUNoise(act_dim, config.noise_theta, config.noise_sigma, self.rng)
        self.buffer = ReplayBuffer(config.buffer, obs_dim, act_dim, seed)
        self.steps = 0
        gamma, tau = config.gamma, config.tau

        @tf.function
        def _train(o, a, r, o2, d):
            a2 = self.actor_t(o2)
            y = r + gamma * (1.0 - d) * self.critic_t([o2, a2])
            with tf.GradientTape() as tape:
                lc = tf.reduce_mean(tf.square(y - self.critic([o, a])))
            self.opt_c.apply_gradients(zip(tape.gradient(lc, self.critic.trainable_variables), self.critic.trainable_variables))
            with tf.GradientTape() as tape:
                la = -tf.reduce_mean(self.critic([o, self.actor(o)]))
            self.opt_a.apply_gradients(zip(tape.gradient(la, self.actor.trainable_variables), self.actor.trainable_variables))
            for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                for w, wt in zip(net.weights, tgt.weights):
                    wt.assign(tau * w + (1 - tau) * wt)
            return lc, la

        self._train = _train

    def _to_env(self, u: np.ndarray) -> np.ndarray:
        return self.mid + self.half * u

    def _to_unit(self, a: np.ndarray) -> np.ndarray:
        return (np.asarray(a) - self.mid) / np.where(self.half > 0, self.half, 1.0)

    def act(self, obs) -> np.ndarray:
        u = np.asarray(self.actor(np.asarray(obs, np.float32)[None], training=False))[0]
        return self._to_env(u)

    def train(self, env, history: TrainingHistory | None = None, callback=None, checkpoint=None) -> TrainingHistory:
        h = history or TrainingHistory()
        c = self.cfg
        for ep in range(c.episodes):
            obs, _ = env.reset(seed=int(self.seed * 100_000 + ep))
            self.noise.reset()
            total, n, lcs, las, reason = 0.0, 0, [], [], None
            while True:
                if self.steps < c.warmup_steps:
                    u = self.rng.uniform(-1, 1, size=len(self.low))
                else:
                    u = np.clip(np.asarray(self.actor(np.asarray(obs, np.float32)[None], training=False))[0] + self.noise(), -1, 1)
                obs2, r, term, trunc, info = env.step(self._to_env(u))
                self.buffer.add(obs, u, r, obs2, term)
                obs, total, n = obs2, total + float(r), n + 1
                self.steps += 1
                if self.buffer.n >= c.warmup_steps:
                    lc, la = self._train(*self.buffer.sample(c.batch))
                    lcs.append(float(lc))
                    las.append(float(la))
                if term or trunc:
                    reason = info.get("termination") if term else None
                    break
            h.episode_reward.append(total)
            h.episode_length.append(n)
            h.episode_violation.append(reason)
            h.loss.append(float(np.mean(lcs)) if lcs else None)
            h.actor_loss.append(float(np.mean(las)) if las else None)
            if (ep + 1) % c.eval_every == 0 or ep == c.episodes - 1:
                h.eval_episode.append(ep + 1)
                h.eval_reward.append(float(np.mean([run_episode(env, self.act, seed=s)["reward"] for s in c.eval_seeds])))
                if checkpoint is not None:
                    checkpoint(ep + 1, self)
            if callback is not None:
                callback(ep + 1, h)
        return h

    def policy_dict(self) -> dict:
        return {"algorithm": "ddpg", "low": self.low.tolist(), "high": self.high.tolist(), "weights": [w.tolist() for w in self.actor.get_weights()],
                "hidden": list(self.cfg.hidden)}
