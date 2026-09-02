"""Classifier config schema (docs/design_ai_layer_transversal.md Sec. 6.2, Sec. 8 step 4): the
validated definition build_classifier(config) reads to construct a keras.Model. A block LIST, not
a fixed layer count -- adding a layer is adding a YAML list item, not editing Python (Sec. 6.1's
"definición, construcción, entrenamiento" separation).

Unlike regressors/schemas.py, there is no per-tier structural guardrail here: the design doc's
per-tier constraint (Sec. 9's "no LSTM on ESP32") is specific to recurrent architectures, which
this schema has none of -- only Conv1D(+SE) blocks, which are already cheap enough for every tier
(sensor.py/gateway.py, both already ESP32/RPi5-deployed, are exactly that shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_TIERS = frozenset({"pc", "rpi5", "esp32"})
ALLOWED_BLOCK_TYPES = frozenset({"conv1d"})


class ClassifierConfigError(ValueError):
    """A classifier config failed schema validation -- raised at load time, not once
    build_classifier tries to use it."""


@dataclass(frozen=True)
class ConvBlockConfig:
    """One `blocks` list entry (Sec. 6.2): Conv1D + BatchNorm + ReLU, with an optional
    squeeze-excite gate (classifiers.gateway.se_block)."""

    filters: int
    kernel_size: int
    use_se: bool = False

    def __post_init__(self):
        if self.filters <= 0:
            raise ClassifierConfigError(f"block filters must be > 0, got {self.filters}")
        if self.kernel_size <= 0:
            raise ClassifierConfigError(f"block kernel_size must be > 0, got {self.kernel_size}")


@dataclass(frozen=True)
class ClassifierConfig:
    """Sec. 6.2's config: `domain`/`tier` are metadata (not read by build_classifier itself --
    they select which config to use, see the eventual ai/registry.py, Sec. 7), the rest defines
    the architecture. `n_channels` is deliberately NOT a field here -- see builder.py's docstring:
    it's a runtime argument, matching build_gateway/build_sensor's existing convention."""

    domain: str
    tier: str
    input_window: int
    num_classes: int
    blocks: tuple
    dense_units: tuple = ()
    dropout: float = 0.0

    def __post_init__(self):
        if not self.domain:
            raise ClassifierConfigError("domain must be non-empty")
        if self.tier not in ALLOWED_TIERS:
            raise ClassifierConfigError(f"tier {self.tier!r} not in {sorted(ALLOWED_TIERS)}")
        if self.input_window <= 0:
            raise ClassifierConfigError(f"input_window must be > 0, got {self.input_window}")
        if self.num_classes < 2:
            raise ClassifierConfigError(f"num_classes must be >= 2, got {self.num_classes}")
        if not self.blocks:
            raise ClassifierConfigError("blocks must be non-empty -- at least one conv block is required")
        if any(u <= 0 for u in self.dense_units):
            raise ClassifierConfigError(f"all dense_units must be > 0, got {self.dense_units}")
        if not (0.0 <= self.dropout < 1.0):
            raise ClassifierConfigError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


def load_classifier_config(path) -> ClassifierConfig:
    """Loads and validates one classifier config YAML file. Raises ClassifierConfigError on any
    schema violation -- malformed structure, unknown block type, invalid field values."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ClassifierConfigError(f"{path}: expected a mapping, got {raw!r}")
    raw = dict(raw)

    blocks_raw = raw.pop("blocks", None)
    if not isinstance(blocks_raw, list):
        raise ClassifierConfigError(f"{path}: 'blocks' must be a list")
    blocks = []
    for b in blocks_raw:
        if not isinstance(b, dict):
            raise ClassifierConfigError(f"{path}: each block must be a mapping, got {b!r}")
        b = dict(b)
        block_type = b.pop("type", None)
        if block_type not in ALLOWED_BLOCK_TYPES:
            raise ClassifierConfigError(f"{path}: block type {block_type!r} not in {sorted(ALLOWED_BLOCK_TYPES)}")
        try:
            blocks.append(ConvBlockConfig(**b))
        except TypeError as exc:
            raise ClassifierConfigError(f"{path}: malformed block entry {b!r}: {exc}") from exc

    dense_units = tuple(raw.pop("dense_units", ()) or ())
    try:
        return ClassifierConfig(blocks=tuple(blocks), dense_units=dense_units, **raw)
    except TypeError as exc:
        raise ClassifierConfigError(f"{path}: malformed config: {exc}") from exc
