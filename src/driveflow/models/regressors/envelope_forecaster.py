"""Envelope forecaster, ported verbatim from paper_federative's `build_envelope_forecaster`
(repo/03_forecast_regressors_etapa2/train_forecast_envelope_combined.py) -- same DS-CNN backbone
as classifiers/sensor.py, different head: predicts a future RMS-per-bin envelope instead of a
class. `window_samples` is a required argument here instead of a module-level constant, same
reasoning as sensor.py/gateway.py.
"""

import keras

BACKBONE_DIM = 32


def build_envelope_forecaster(n_channels: int, n_bins: int, window_samples: int, name: str = "envelope_forecaster") -> keras.Model:
    inp = keras.Input(shape=(window_samples, n_channels), name="input")

    x = keras.layers.DepthwiseConv1D(9, padding="same", name="dw1")(inp)
    x = keras.layers.Conv1D(32, 1, padding="same", name="pw1")(x)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.ReLU(name="relu1")(x)
    x = keras.layers.MaxPooling1D(4, name="pool1")(x)

    x = keras.layers.DepthwiseConv1D(7, padding="same", name="dw2")(x)
    x = keras.layers.Conv1D(32, 1, padding="same", name="pw2")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.ReLU(name="relu2")(x)
    x = keras.layers.MaxPooling1D(4, name="pool2")(x)

    x = keras.layers.DepthwiseConv1D(5, padding="same", name="dw3")(x)
    x = keras.layers.Conv1D(32, 1, padding="same", name="pw3")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.ReLU(name="relu3")(x)

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    backbone = keras.layers.Dense(BACKBONE_DIM, activation="relu", name="backbone_output")(x)
    head = keras.layers.Dense(n_bins * n_channels, activation="softplus", name="envelope_head")(backbone)
    out = keras.layers.Reshape((n_bins, n_channels), name="envelope_output")(head)

    return keras.Model(inputs=inp, outputs=out, name=name)
