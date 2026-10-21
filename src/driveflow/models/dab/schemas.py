"""DAB performance-model config (task E9 preparation, round 2026-10-21, D8; domain ``dab``).

A small regressor from operating point and TPS modulation to converter stress, trained on a PLECS
sweep table. The idea comes from Fund. ``9_Case_Studies_PE/DAB_Design/Performance_Modeling_and_Design``
(PyTorch, CC BY-NC data); SPADE's version is written for Keras and no code or data was copied.
Declared in YAML and trained by ``experiments/train_model.py`` (detected by ``task: dab_performance``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class DabConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DabPerformanceConfig:
    domain: str
    tier: str
    inputs: tuple
    outputs: tuple
    group_by: tuple
    hidden: tuple = (64, 64)
    valid_column: str | None = None
    task: str = "dab_performance"

    def __post_init__(self):
        if self.domain != "dab":
            raise DabConfigError(f"a DAB performance model belongs to domain 'dab' (never mixed with other domains), got {self.domain!r}")
        if self.tier not in ("pc", "rpi5", "esp32"):
            raise DabConfigError(f"tier must be pc, rpi5 or esp32, got {self.tier!r}")
        if not self.inputs or not self.outputs:
            raise DabConfigError("inputs and outputs must be non-empty")
        if set(self.inputs) & set(self.outputs):
            raise DabConfigError("a column cannot be both input and output")
        if not self.group_by or not set(self.group_by) <= set(self.inputs):
            raise DabConfigError("group_by (the operating point the split keeps together) must be a non-empty subset of inputs")
        if any(h <= 0 for h in self.hidden):
            raise DabConfigError("hidden sizes must be > 0")


def load_dab_config(path) -> DabPerformanceConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise DabConfigError(f"{path}: expected a mapping")
    raw = dict(raw)
    for k in ("inputs", "outputs", "group_by", "hidden"):
        if k in raw:
            raw[k] = tuple(raw[k])
    try:
        return DabPerformanceConfig(**raw)
    except TypeError as exc:
        raise DabConfigError(f"{path}: malformed config: {exc}") from exc
