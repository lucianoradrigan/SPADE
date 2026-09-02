"""Generic classifier construction from a ClassifierConfig (Sec. 6.1/6.2, Sec. 8 step 4): reads
the config's block list and builds a plain Conv1D(+SE)-stack -> GAP -> dense-head model. Reuses
classifiers.gateway.se_block rather than redefining squeeze-excite a second time.

n_channels is a runtime argument, not a config field -- matching build_gateway/build_sensor's
existing convention (models/classifiers/gateway.py, sensor.py): the live channel set is discovered
from the actual dataset at training/window-build time (models.common.windowing.discover_channels),
not fixed at config-authoring time.
"""

import keras

from driveflow.models.classifiers.gateway import se_block
from driveflow.models.classifiers.schemas import ClassifierConfig


def build_classifier(config: ClassifierConfig, n_channels: int, name: str = "classifier") -> keras.Model:
    inp = keras.Input(shape=(config.input_window, n_channels), name="input")
    x = inp
    for i, block in enumerate(config.blocks):
        x = keras.layers.Conv1D(block.filters, block.kernel_size, padding="same", name=f"block{i + 1}_conv")(x)
        x = keras.layers.BatchNormalization(name=f"block{i + 1}_bn")(x)
        x = keras.layers.ReLU(name=f"block{i + 1}_relu")(x)
        if block.use_se:
            x = se_block(x, prefix=f"block{i + 1}_se")

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    for i, units in enumerate(config.dense_units):
        x = keras.layers.Dense(units, activation="relu", name=f"dense{i + 1}")(x)
        if config.dropout > 0:
            x = keras.layers.Dropout(config.dropout, name=f"dropout{i + 1}")(x)
    out = keras.layers.Dense(config.num_classes, activation="softmax", name="head")(x)

    return keras.Model(inputs=inp, outputs=out, name=name)
