"""Keras MLP for the DAB performance model (schemas.DabPerformanceConfig)."""

from __future__ import annotations


def build_dab_model(config, n_inputs: int | None = None, n_outputs: int | None = None):
    import keras

    n_in = n_inputs or len(config.inputs)
    n_out = n_outputs or len(config.outputs)
    layers = [keras.Input((n_in,))] + [keras.layers.Dense(h, activation="relu") for h in config.hidden] + [keras.layers.Dense(n_out)]
    return keras.Sequential(layers, name="dab_performance")
