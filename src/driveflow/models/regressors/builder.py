"""Generic forecaster construction from a ForecasterConfig (Sec. 6.1/6.3, Sec. 8 step 4): stacked
LSTM/GRU with an optional self-attention pooling head, producing a direct (horizon, n_channels)
multi-step forecast of the (normalized) signal itself.

This is a DIFFERENT output convention from regressors/envelope_forecaster.py, which predicts a
(n_bins, n_channels) RMS-per-bin envelope, not raw future values -- that architecture already
exists and is what Fase D.1 adapts directly (INSTRUCTIONS.md Sec. 6: "adaptar
build_envelope_forecaster()"), not this builder. This module is the config-driven LSTM/GRU family
Sec. 4.1 describes for the PC tier; the two are siblings, not a replacement of one by the other.

Only recurrent_type in {"lstm", "gru"} is implemented here -- "none" is reserved for the
ESP32-tier non-recurrent architecture (Sec. 4.1: "Conv1D causal dilatada ... o predictor
estadístico simple"), which Sec. 8 defers to step 8 (distillation) and is not designed yet; the
schema (schemas.py) already accepts it so an ESP32 config can be authored and validated ahead of
that builder existing, without this function silently mishandling it.

n_channels is a runtime argument, not a config field -- same reasoning as classifiers/builder.py.
"""

import keras

from driveflow.models.regressors.schemas import ForecasterConfig

_RECURRENT_LAYER = {"lstm": keras.layers.LSTM, "gru": keras.layers.GRU}


def build_forecaster(config: ForecasterConfig, n_channels: int, name: str = "forecaster") -> keras.Model:
    if config.recurrent_type not in _RECURRENT_LAYER:
        raise NotImplementedError(
            f"recurrent_type={config.recurrent_type!r} has no builder yet -- only 'lstm'/'gru' are "
            "implemented (Sec. 8 step 4 scope is the PC tier; the ESP32 non-recurrent architecture "
            "is deferred to step 8, see module docstring)."
        )
    layer_cls = _RECURRENT_LAYER[config.recurrent_type]

    inp = keras.Input(shape=(config.input_window, n_channels), name="input")
    x = inp
    for i, units in enumerate(config.layers):
        return_sequences = config.use_attention or i < len(config.layers) - 1
        x = layer_cls(units, return_sequences=return_sequences, name=f"{config.recurrent_type}_{i + 1}")(x)

    if config.use_attention:
        x = keras.layers.Attention(name="self_attention")([x, x])
        x = keras.layers.GlobalAveragePooling1D(name="attention_pool")(x)

    head = keras.layers.Dense(config.horizon * n_channels, name="forecast_head")(x)
    out = keras.layers.Reshape((config.horizon, n_channels), name="forecast_output")(head)

    return keras.Model(inputs=inp, outputs=out, name=name)
