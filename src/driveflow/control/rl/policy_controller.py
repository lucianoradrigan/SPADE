"""A trained policy as a controller -- expansion task E8.

``PolicyController`` loads a saved policy (``policy_dict()`` of DQNAgent / DDPGAgent: weights plus,
for DQN, the action discretiser) and evaluates it with a numpy forward pass, with no TensorFlow at
run time. It has the same reset()/control() shape as the PI, MPC and DPC controllers. The difference
is its input: SPADE's controllers read the plant STATE and a reference, while a policy reads the
environment's OBSERVATION, which already contains the reference. ``control(obs)`` takes that
observation. ``benchmark.py`` drives every controller through the same environment, so the
comparison uses the same plant, conditions, limits and timestep.

A saved policy is a candidate: nothing in SPADE uses it as a default.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _relu_mlp(weights: list, x: np.ndarray, out: str | None) -> np.ndarray:
    W = [np.asarray(w, dtype=np.float64) for w in weights]
    h = np.asarray(x, dtype=np.float64)
    for k in range(0, len(W) - 2, 2):
        h = np.maximum(h @ W[k] + W[k + 1], 0.0)
    y = h @ W[-2] + W[-1]
    return np.tanh(y) if out == "tanh" else y


class PolicyController:
    def __init__(self, policy: dict):
        if policy.get("algorithm") not in ("dqn", "ddpg"):
            raise ValueError(f"unknown policy algorithm {policy.get('algorithm')!r}")
        self.policy = policy
        self.algorithm = policy["algorithm"]
        if self.algorithm == "dqn":
            self.levels = np.asarray(policy["discretizer"]["levels"], dtype=np.float64)
        else:
            self.low, self.high = np.asarray(policy["low"]), np.asarray(policy["high"])

    @classmethod
    def load(cls, path) -> "PolicyController":
        return cls(json.loads(Path(path).read_text()))

    def reset(self) -> None:
        """Stateless policy; kept for the controller interface."""

    def control(self, obs) -> np.ndarray:
        if self.algorithm == "dqn":
            q = _relu_mlp(self.policy["weights"], np.asarray(obs)[None], None)[0]
            return self.levels[int(np.argmax(q))].copy()
        u = _relu_mlp(self.policy["weights"], np.asarray(obs)[None], "tanh")[0]
        return (self.high + self.low) / 2 + (self.high - self.low) / 2 * u

    __call__ = control
