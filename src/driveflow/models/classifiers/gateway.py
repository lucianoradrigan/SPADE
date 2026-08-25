"""ResNet-1D + SE classifier, ported verbatim from paper_federative's `build_gateway`/
`se_block`/`resnet_block` (repo/02_local_baselines_etapa1/train_baseline.py) -- targets a
Raspberry-Pi-class gateway, ~200K params FP32. `window_samples` is a required argument here
instead of a module-level constant, same reasoning as sensor.py.

The projection head (for feature distillation against the sensor's backbone_output) is kept even
though it's inert in this training regime (loss_weight=0, no distillation happening yet -- that
would be a Stage-2-equivalent extension, not implemented) because the architecture itself is what
Patch 5-era planning already committed to; leaving it out would silently change the model, not
simplify it.
"""

import keras

BACKBONE_DIM = 32
GW_FEATURE_DIM = 64


def se_block(x, ratio: int = 8, prefix: str = "se"):
    ch = x.shape[-1]
    s = keras.layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    s = keras.layers.Dense(max(1, ch // ratio), activation="relu", name=f"{prefix}_fc1")(s)
    s = keras.layers.Dense(ch, activation="sigmoid", name=f"{prefix}_fc2")(s)
    s = keras.layers.Reshape((1, ch), name=f"{prefix}_reshape")(s)
    return keras.layers.Multiply(name=f"{prefix}_mul")([x, s])


def resnet_block(x, filters: int, ks: int = 7, prefix: str = "res"):
    sc = x
    if x.shape[-1] != filters:
        sc = keras.layers.Conv1D(filters, 1, padding="same", name=f"{prefix}_sc")(sc)
    x = keras.layers.Conv1D(filters, ks, padding="same", name=f"{prefix}_c1")(x)
    x = keras.layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = keras.layers.ReLU(name=f"{prefix}_r1")(x)
    x = keras.layers.Conv1D(filters, ks, padding="same", name=f"{prefix}_c2")(x)
    x = keras.layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = se_block(x, prefix=f"{prefix}_se")
    x = keras.layers.Add(name=f"{prefix}_add")([x, sc])
    return keras.layers.ReLU(name=f"{prefix}_r2")(x)


def build_gateway(n_classes: int, n_channels: int, window_samples: int, name: str = "gateway") -> keras.Model:
    inp = keras.Input(shape=(window_samples, n_channels), name="input")

    x = keras.layers.Conv1D(32, 15, padding="same", name="stem")(inp)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)
    x = keras.layers.MaxPooling1D(4, name="stem_pool")(x)

    x = resnet_block(x, 32, prefix="r1")
    x = keras.layers.MaxPooling1D(2, name="p1")(x)
    x = resnet_block(x, 64, prefix="r2")
    x = keras.layers.MaxPooling1D(2, name="p2")(x)
    x = resnet_block(x, 64, prefix="r3")

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    gw_feat = keras.layers.Dense(GW_FEATURE_DIM, activation="relu", name="gateway_features")(x)

    proj = keras.layers.Dense(BACKBONE_DIM, activation="relu", name="projection_head")(gw_feat)
    out = keras.layers.Dense(n_classes, activation="softmax", name="head")(gw_feat)

    return keras.Model(inputs=inp, outputs={"classification": out, "projection": proj}, name=name)
