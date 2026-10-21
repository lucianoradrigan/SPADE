"""Reward as a list of named terms -- expansion task E1 (spec Sec. 3.4 stage 5C, Sec. 8 RL Lab).

Every term is a penalty on a dimensionless signal the environment computes from its own state, so
weights are comparable across plants. The reward is ``-sum(weight * penalty)`` over the enabled
terms; ``evaluate`` also returns the per-term breakdown, which the environments put in
``info["reward_terms"]`` so that a reward can be explained after the fact instead of guessed.

Signals each environment must provide (all dimensionless, see envs/native.py for the scales):

- ``tracking_error``: (reference - output) / output scale.
- ``current_fraction``: |i| / current limit.
- ``action_delta``: (a_t - a_(t-1)) / width of the action space.

Design taken from Fund. 7_Reinforcement_Learning/RL_buck_control.ipynb (BuckAvgEnv: a reward
made of a tracking term plus penalties); no code was copied. The same schema is meant for the
PLECS environment (E4) and the RL Lab reward editor (E8).
"""

from __future__ import annotations

import dataclasses
import math

#: kind -> (signal it reads, formula shown to the user)
TERM_KINDS = {
    "tracking": ("tracking_error", "w · e²,  e = (reference − output) / scale"),
    "overcurrent": ("current_fraction", "w · max(0, |i|/i_limit − threshold)²"),
    "action_rate": ("action_delta", "w · ((a_t − a_(t−1)) / action width)²"),
}


@dataclasses.dataclass(frozen=True)
class RewardTerm:
    kind: str
    weight: float = 1.0
    enabled: bool = True
    #: overcurrent only: fraction of the current limit where the penalty starts
    threshold: float = 1.0

    def __post_init__(self):
        if self.kind not in TERM_KINDS:
            raise ValueError(f"Unknown reward term {self.kind!r}; expected one of {sorted(TERM_KINDS)}")
        if not (math.isfinite(self.weight) and self.weight >= 0):
            raise ValueError(f"Reward weight for {self.kind!r} must be finite and >= 0, got {self.weight}")
        if not (math.isfinite(self.threshold) and self.threshold > 0):
            raise ValueError(f"Overcurrent threshold must be finite and > 0, got {self.threshold}")

    def penalty(self, signals: dict) -> float:
        x = float(signals[TERM_KINDS[self.kind][0]])
        if self.kind == "overcurrent":
            return max(0.0, abs(x) - self.threshold) ** 2
        return x * x

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_TERMS = (RewardTerm("tracking", 1.0), RewardTerm("overcurrent", 10.0, threshold=0.8), RewardTerm("action_rate", 0.01))


def parse_terms(terms) -> tuple[RewardTerm, ...]:
    """Accepts RewardTerm objects or dicts; at least one enabled term with a positive weight."""
    out = tuple(t if isinstance(t, RewardTerm) else RewardTerm(**t) for t in (DEFAULT_TERMS if terms is None else terms))
    if not any(t.enabled and t.weight > 0 for t in out):
        raise ValueError("Add at least one reward term with a positive weight.")
    kinds = [t.kind for t in out if t.enabled]
    if len(kinds) != len(set(kinds)):
        raise ValueError("Each reward term kind may appear only once (avoids counting the same signal twice).")
    return out


def evaluate(terms: tuple[RewardTerm, ...], signals: dict) -> tuple[float, dict]:
    """(reward, {kind: weighted penalty}) -- the breakdown sums to -reward."""
    breakdown = {t.kind: t.weight * t.penalty(signals) for t in terms if t.enabled}
    return -float(sum(breakdown.values())), breakdown
