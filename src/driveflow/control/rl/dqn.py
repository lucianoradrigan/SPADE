"""Deep Q-network for discrete actions -- expansion task E8.

Origin: Fund. ``7_Reinforcement_Learning/RL_buck_control.ipynb`` (PyTorch DQN with replay, a target
network and discrete Δduty actions). Reimplemented in Keras; no code copied. Differences:
- The discrete actions are ABSOLUTE duty levels (``ActionDiscretizer``), not Δduty. SPADE's
  observation does not carry the previous duty, so a Δduty policy would not be Markov in it.
- The target network is updated softly (τ).
- The loss is the Huber loss.

DQN accepts only a Discrete action space, or a Box space with an explicit ``ActionDiscretizer``
(spec Sec. 5 step 1). The discretiser is part of the saved policy.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from driveflow.control.rl.common import ActionDiscretizer, ReplayBuffer, TrainingHistory, mlp, run_episode, seed_everything


@dataclasses.dataclass
class DQNConfig:
    episodes: int = 60
    gamma: float = 0.98
    lr: float = 1e-3
    batch: int = 64
    buffer: int = 50_000
    warmup_steps: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    tau: float = 0.01
    hidden: tuple = (64, 64)
    eval_every: int = 10
    eval_seeds: tuple = (1001, 1002)
    train_every: int = 1


class DQNAgent:
    def __init__(self, obs_dim: int, discretizer: ActionDiscretizer, config: DQNConfig, seed: int = 0):
        import keras
        import tensorflow as tf

        seed_everything(seed)
        self.cfg, self.disc, self.seed = config, discretizer, seed
        self.q = mlp(obs_dim, discretizer.n, config.hidden, name="q")
        self.q_target = mlp(obs_dim, discretizer.n, config.hidden, name="q_target")
        self.q_target.set_weights(self.q.get_weights())
        self.opt = keras.optimizers.Adam(config.lr)
        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(config.buffer, obs_dim, 1, seed)
        self.steps = 0
        huber = keras.losses.Huber()
        gamma, tau = config.gamma, config.tau

        @tf.function
        def _train(o, a, r, o2, d):
            q_next = tf.reduce_max(self.q_target(o2), axis=1, keepdims=True)
            target = r + gamma * (1.0 - d) * q_next
            with tf.GradientTape() as tape:
                q = tf.gather(self.q(o), tf.cast(a[:, 0], tf.int32), axis=1, batch_dims=1)[:, None]
                loss = huber(target, q)
            grads = tape.gradient(loss, self.q.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
            for w, wt in zip(self.q.weights, self.q_target.weights):
                wt.assign(tau * w + (1 - tau) * wt)
            return loss

        self._train = _train

    def greedy(self, obs) -> int:
        return int(np.argmax(np.asarray(self.q(np.asarray(obs, np.float32)[None], training=False))[0]))

    def act(self, obs) -> np.ndarray:
        return self.disc.action(self.greedy(obs))

    def epsilon(self) -> float:
        c = self.cfg
        return c.eps_end + (c.eps_start - c.eps_end) * max(0.0, 1 - self.steps / c.eps_decay_steps)

    def train(self, env, history: TrainingHistory | None = None, callback=None, checkpoint=None) -> TrainingHistory:
        h = history or TrainingHistory()
        c = self.cfg
        for ep in range(c.episodes):
            obs, _ = env.reset(seed=int(self.seed * 100_000 + ep))
            total, n, losses, reason = 0.0, 0, [], None
            while True:
                i = int(self.rng.integers(self.disc.n)) if self.rng.random() < self.epsilon() else self.greedy(obs)
                obs2, r, term, trunc, info = env.step(self.disc.action(i))
                self.buffer.add(obs, [i], r, obs2, term)
                obs, total, n = obs2, total + float(r), n + 1
                self.steps += 1
                if self.buffer.n >= c.warmup_steps and self.steps % c.train_every == 0:
                    losses.append(float(self._train(*self.buffer.sample(c.batch))))
                if term or trunc:
                    reason = info.get("termination") if term else None
                    break
            h.episode_reward.append(total)
            h.episode_length.append(n)
            h.episode_violation.append(reason)
            h.loss.append(float(np.mean(losses)) if losses else None)
            if (ep + 1) % c.eval_every == 0 or ep == c.episodes - 1:
                h.eval_episode.append(ep + 1)
                h.eval_reward.append(float(np.mean([run_episode(env, self.act, seed=s)["reward"] for s in c.eval_seeds])))
                if checkpoint is not None:
                    checkpoint(ep + 1, self)
            if callback is not None:
                callback(ep + 1, h)
        return h

    def policy_dict(self) -> dict:
        return {"algorithm": "dqn", "discretizer": self.disc.to_dict(), "weights": [w.tolist() for w in self.q.get_weights()], "hidden": list(self.cfg.hidden)}
