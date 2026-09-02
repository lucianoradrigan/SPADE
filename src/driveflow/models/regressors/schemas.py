"""Forecaster config schema (docs/design_ai_layer_transversal.md Sec. 6.3, Sec. 8 step 4). Carries
the mandatory guardrail from that document's Sec. 9: an ESP32-tier config must not be able to
express a recurrent architecture at all (LSTM has per-step state and inference cost the design doc
explicitly rules out for a bare MCU -- Sec. 4.1 -- and GRU shares the same structural problem, just
lighter, so it is excluded too rather than treated as an ESP32-safe alternative). This is enforced
here, structurally, in RECURRENT_TYPES_BY_TIER -- an ESP32 config with recurrent_type: lstm (or
gru) fails at load time, not once someone tries to flash it. RPi5 keeps both lstm (pruned) and gru
available, per Sec. 4.1's "GRU compacto ... o LSTM podado" -- only the PC tier is unrestricted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_TIERS = frozenset({"pc", "rpi5", "esp32"})
ALLOWED_RECURRENT_TYPES = frozenset({"lstm", "gru", "none"})

#: Sec. 4.1/Sec. 9 guardrail -- which recurrent_type values a given tier's config may declare.
#: "none" always stays available (a tier can always downgrade to a not-yet-designed non-recurrent
#: architecture, Sec. 8 step 8); lstm/gru are only added back in for tiers the design doc actually
#: names them for.
RECURRENT_TYPES_BY_TIER = {
    "pc": frozenset({"lstm", "gru", "none"}),
    "rpi5": frozenset({"lstm", "gru", "none"}),
    "esp32": frozenset({"none"}),
}


class ForecasterConfigError(ValueError):
    """A forecaster config failed schema validation -- raised at load time, not once
    build_forecaster tries to use it."""


@dataclass(frozen=True)
class ForecasterConfig:
    """Sec. 6.3's config. `n_channels` is deliberately NOT a field here, same reasoning as
    classifiers/schemas.py's ClassifierConfig -- see builder.py's docstring."""

    domain: str
    tier: str
    input_window: int
    horizon: int
    recurrent_type: str
    layers: tuple
    use_attention: bool = False

    def __post_init__(self):
        if not self.domain:
            raise ForecasterConfigError("domain must be non-empty")
        if self.tier not in ALLOWED_TIERS:
            raise ForecasterConfigError(f"tier {self.tier!r} not in {sorted(ALLOWED_TIERS)}")
        if self.recurrent_type not in ALLOWED_RECURRENT_TYPES:
            raise ForecasterConfigError(f"recurrent_type {self.recurrent_type!r} not in {sorted(ALLOWED_RECURRENT_TYPES)}")
        if self.recurrent_type not in RECURRENT_TYPES_BY_TIER[self.tier]:
            raise ForecasterConfigError(
                f"tier {self.tier!r} cannot use recurrent_type {self.recurrent_type!r} -- allowed for this tier: "
                f"{sorted(RECURRENT_TYPES_BY_TIER[self.tier])} (see docs/design_ai_layer_transversal.md Sec. 4.1/9)"
            )
        if self.input_window <= 0:
            raise ForecasterConfigError(f"input_window must be > 0, got {self.input_window}")
        if self.horizon <= 0:
            raise ForecasterConfigError(f"horizon must be > 0, got {self.horizon}")
        if self.recurrent_type != "none" and not self.layers:
            raise ForecasterConfigError("layers must be non-empty when recurrent_type is 'lstm' or 'gru'")
        if any(u <= 0 for u in self.layers):
            raise ForecasterConfigError(f"all layers units must be > 0, got {self.layers}")


def load_forecaster_config(path) -> ForecasterConfig:
    """Loads and validates one forecaster config YAML file. Raises ForecasterConfigError on any
    schema violation -- including a tier/recurrent_type combination the design doc forbids."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ForecasterConfigError(f"{path}: expected a mapping, got {raw!r}")
    raw = dict(raw)
    layers = tuple(raw.pop("layers", ()) or ())
    try:
        return ForecasterConfig(layers=layers, **raw)
    except TypeError as exc:
        raise ForecasterConfigError(f"{path}: malformed config: {exc}") from exc
