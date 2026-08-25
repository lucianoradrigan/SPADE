"""DS-CNN classifier, ported verbatim (architecture, activations, layer sizes) from
paper_federative's `build_sensor` (repo/02_local_baselines_etapa1/train_baseline.py) -- targets
an ESP32-class edge device, ~4-8K params, <5ms INT8 inference. The only change is
`window_samples` becoming a required argument instead of a module-level constant: the original
hardcodes `WINDOW_SAMPLES=4000` (FS_TARGET=8000Hz); driveflow computes it from its own fs_hz (see
models/common/windowing.py).
"""

import keras

BACKBONE_DIM = 32


def build_sensor(n_classes: int, n_channels: int, window_samples: int, name: str = "sensor") -> keras.Model:
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
    out = keras.layers.Dense(n_classes, activation="softmax", name="head")(backbone)

    return keras.Model(inputs=inp, outputs=out, name=name)
