"""Generic forecaster construction from a ForecasterConfig (Sec. 6.1/6.3, Sec. 8 steps 4/8):
stacked LSTM/GRU with an optional self-attention pooling head (recurrent_type in {"lstm", "gru"}),
OR a causal dilated Conv1D stack -- a small TCN (recurrent_type == "none", Sec. 8 step 8: the
ESP32-tier non-recurrent architecture, Sec. 4.1's "Conv1D causal dilatada"). Either way, output is
a direct (horizon, n_channels) multi-step forecast of the (normalized) signal itself.

This is a DIFFERENT output convention from regressors/envelope_forecaster.py, which predicts a
(n_bins, n_channels) RMS-per-bin envelope, not raw future values -- that architecture already
exists and is what Fase D.1 adapts directly (INSTRUCTIONS.md Sec. 6: "adaptar
build_envelope_forecaster()"), not this builder. This module is the config-driven LSTM/GRU/TCN
family Sec. 4.1 describes across all three tiers; the two families are siblings, not a replacement
of one by the other.

For recurrent_type == "none", `config.layers` is reinterpreted as filter counts per causal Conv1D
layer (dilation rate doubling per layer: 1, 2, 4, ... -- a standard WaveNet/TCN-style receptive-
field growth) rather than recurrent unit counts -- the schema (schemas.py) doesn't need a separate
field for this since a ForecasterConfig is only ever built for one recurrent_type at a time. The
"predictor estadístico simple (suavizado exponencial / Kalman escalar)" alternative Sec. 4.1 also
names is NOT implemented here (that is a non-learned, non-Keras statistical filter, out of scope
for a keras.Model-returning builder) -- `layers=()` degenerates to a trivial linear-from-last-
observation model instead, which is a legitimate (if minimal) recurrent_type: "none" build, not an
error.

n_channels is a runtime argument, not a config field -- same reasoning as classifiers/builder.py.
"""

import keras

from driveflow.models.regressors.schemas import ForecasterConfig

_RECURRENT_LAYER = {"lstm": keras.layers.LSTM, "gru": keras.layers.GRU}
#: Fixed causal-conv kernel size for recurrent_type == "none" -- not config-authored (same
#: reasoning as the pointwise stage of classifiers/builder.py's "dsconv1d" always using kernel
#: size 1): one architectural knob (layer count/filters) is enough for this tier's config surface.
_TCN_KERNEL_SIZE = 3


def _build_recurrent_tower(x, config: ForecasterConfig):
    layer_cls = _RECURRENT_LAYER[config.recurrent_type]
    for i, units in enumerate(config.layers):
        return_sequences = config.use_attention or i < len(config.layers) - 1
        x = layer_cls(units, return_sequences=return_sequences, name=f"{config.recurrent_type}_{i + 1}")(x)
    if config.use_attention:
        x = keras.layers.Attention(name="self_attention")([x, x])
        x = keras.layers.GlobalAveragePooling1D(name="attention_pool")(x)
    return x


def _build_tcn_tower(x, config: ForecasterConfig):
    for i, filters in enumerate(config.layers):
        x = keras.layers.Conv1D(
            filters, _TCN_KERNEL_SIZE, padding="causal", dilation_rate=2**i, name=f"tcn_{i + 1}_conv"
        )(x)
        x = keras.layers.BatchNormalization(name=f"tcn_{i + 1}_bn")(x)
        x = keras.layers.ReLU(name=f"tcn_{i + 1}_relu")(x)
    # Causal convolution means the LAST timestep's receptive field already covers the whole
    # context window -- it is the natural summary to forecast from, unlike GlobalAveragePooling1D
    # (which would blend in earlier, less-relevant timesteps).
    return keras.layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)


def build_forecaster(config: ForecasterConfig, n_channels: int, name: str = "forecaster") -> keras.Model:
    inp = keras.Input(shape=(config.input_window, n_channels), name="input")

    if config.recurrent_type in _RECURRENT_LAYER:
        x = _build_recurrent_tower(inp, config)
    elif config.recurrent_type == "none":
        x = _build_tcn_tower(inp, config)
    else:
        raise NotImplementedError(f"recurrent_type={config.recurrent_type!r} has no builder -- expected 'lstm', 'gru', or 'none'.")

    head = keras.layers.Dense(config.horizon * n_channels, name="forecast_head")(x)
    out = keras.layers.Reshape((config.horizon, n_channels), name="forecast_output")(head)

    return keras.Model(inputs=inp, outputs=out, name=name)
