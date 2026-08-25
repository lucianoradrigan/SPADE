"""Keras/TF port of DPC4PowerElectronics/Main.txt's training loop.

Differences from the MATLAB script (deliberate, not fidelity gaps in the *model*):
- Main.txt's ``modelGradients1`` loops over the minibatch one sample at a time (``for i =
  1:numSamples``), computing ``customLoss1`` per single-column input and summing. loss.py's
  ``dpc_loss`` is already vectorized over the batch (same math, see its docstring), so this loop
  is a plain batched ``GradientTape`` step -- there is no per-sample Python loop to port.
- ``numEpochs = 10000`` in Main.txt; this port exposes epochs as a parameter (10000 full epochs
  is a multi-hour CPU run here) rather than hardcoding MATLAB's value.
"""

from pathlib import Path

import numpy as np
import scipy.io
import tensorflow as tf

from .loss import dpc_loss
from .network import N_INPUTS, build_dpc_network

#: Column order in Data4train.mat, matching Custom_Loss.txt's XBatch row order exactly (see
#: network.py's docstring). Public (not `_`-prefixed): this is also the schema the dashboard's
#: "evaluate your own dataset" upload feature documents to users and validates uploads against --
#: a real external contract, not an internal-only detail.
COLUMNS = [
    "if_alpha", "if_beta", "vc_alpha", "vc_beta", "vref_alpha", "vref_beta", "r",
    "vref_alphaph", "vref_betaph", "vref_alphaph3", "vref_betaph3",
    "vref_alphaph4", "vref_betaph4", "vref_alphaph5", "vref_betaph5",
]


def load_training_data(mat_path: Path, num_train: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Loads Data4train.mat and splits it the way Main.txt does: the first ``num_train`` rows
    for training (Main.txt's ``num=6000``), the remainder held out (Main.txt never uses it, but
    the file has 10000 rows -- kept here as a genuine holdout for validating the port, since
    Macro-fase A's own closing criterion required holdout validation too).
    """
    mat = scipy.io.loadmat(mat_path)
    data = mat["Data4train"].astype(np.float32)
    if data.shape[1] != N_INPUTS:
        raise ValueError(f"expected {N_INPUTS} columns in Data4train.mat, got {data.shape[1]}")
    return data[:num_train], data[num_train:]


def train_on_array(
    x_train: np.ndarray,
    model: "tf.keras.Model | None" = None,
    epochs: int = 200,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[tf.keras.Model, list[float]]:
    """Core training loop, decoupled from Data4train.mat -- trains/fine-tunes on whatever
    (N, 15) array of rows is passed in. `train_dpc` (below) is a thin wrapper of this for the
    original MATLAB dataset; `experiments/finetune_dpc_closed_loop.py` reuses this directly to
    fine-tune on a dataset augmented with real closed-loop rollout states (see that script's
    docstring for why the static Data4train.mat rows alone aren't enough for good closed-loop
    tracking).

    Args:
        model: pass an existing model to continue training it (fine-tuning) instead of
            initializing fresh weights.
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    if model is None:
        model = build_dpc_network()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    n = x_train.shape[0]
    num_batches = max(1, n // batch_size)
    history = []

    for epoch in range(epochs):
        perm = rng.permutation(n)
        shuffled = x_train[perm]
        epoch_losses = []
        for b in range(num_batches):
            batch = shuffled[b * batch_size : (b + 1) * batch_size]
            x_batch = tf.constant(batch)
            with tf.GradientTape() as tape:
                v_o = model(x_batch, training=True)
                loss = dpc_loss(x_batch, v_o)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(float(loss))
        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:5d}  loss {mean_loss:12.3f}")

    return model, history


def train_dpc(
    mat_path: Path,
    epochs: int = 200,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    num_train: int = 6000,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[tf.keras.Model, list[float]]:
    """Trains a fresh DPC network on Data4train.mat. Returns (model, per-epoch mean training loss).

    Args:
        epochs: Main.txt uses 10000; default here is much lower since there's no MATLAB run to
            match epoch-for-epoch, and 10000 epochs is impractically slow on CPU for a first
            port. Loss trend (monotonically decreasing, converging) is what's validated, not a
            specific final value -- see experiments/train_dpc.py's docstring.
    """
    x_train, _ = load_training_data(mat_path, num_train=num_train)
    return train_on_array(x_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed, verbose=verbose)
