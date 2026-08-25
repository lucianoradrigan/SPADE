"""Keras/TF port of DPC4PowerElectronics/Custom_Loss.txt (``customLoss1``) -- the model-based
loss that trains the network in network.py end-to-end, without any labeled optimal action.

What the loss does (ported line-for-line from the MATLAB, see Custom_Loss.txt): the network's
10 outputs are read as 5 (real, imag) pairs -- the converter voltage command ``v_o`` the network
proposes for each of the next 5 control steps. Instead of comparing that command against a label
(there isn't one -- this is the whole point of differentiable predictive control), the loss
*simulates* the filter+capacitor's own discrete-time model (``ADF``/``BDF``, the LCL filter state
matrices for state=[i_f, v_c], identified offline in MATLAB/System Identification and pasted here
as constants -- not something this port re-derives) forward for those 5 steps using the proposed
``v_o`` as the input, starting from the batch's measured (i_f, v_c). It then penalizes:

- ``voltage_loss``: predicted capacitor voltage vs. the reference voltage, at each horizon step.
- ``current_loss``: predicted filter current vs. what it *should* be given the load current plus
  the capacitor's own reactive/quadrature current (``C*wref*v_ref`` term -- KCL at the filter
  capacitor node, i.e. i_filter = i_load + i_capacitor, approximated using the reference voltage's
  own frequency/phase rather than a numerically-differentiated measured voltage).

The load current itself (``i_load = v_c/R``) is evaluated once from the *current* measurement and
held constant across the horizon -- the MATLAB does not re-simulate the load, so neither does this
port (a deliberate fidelity choice, not an omission).

Real and imaginary (alpha/beta) axes are recursed independently through the same real 2x2 ADF/BDF
-- the MATLAB has no cross-axis (jω) term in the state matrices themselves, so this port doesn't
add one either.
"""

import numpy as np
import tensorflow as tf

HORIZON = 5

# Identified discrete-time LCL filter model, state=[i_f, v_c], pasted verbatim from
# Custom_Loss.txt -- these are NOT re-derived here, they come from the original MATLAB toolbox.
ADF = np.array(
    [
        [0.896993414465221, -0.0190993472724508],
        [6.39191488718020, 0.935192109010122],
    ],
    dtype=np.float32,
)
BDF = np.array(
    [
        [0.0190993472724508, 0.0648078909898776],
        [0.0648078909898777, -6.52153066915996],
    ],
    dtype=np.float32,
)
CAPACITANCE_F = 15e-6
OMEGA_REF_RAD_S = 2 * np.pi * 50.0


def simulate_horizon(x_batch: tf.Tensor, v_o: tf.Tensor) -> dict:
    """Runs the same ADF/BDF forward recursion `dpc_loss` uses, but returns every intermediate
    (horizon, batch) array instead of collapsing straight to a scalar -- shared by `dpc_loss`
    (training) and evaluate.py (metrics), so both read off the exact same physics.

    Args:
        x_batch: (batch, 15) -- see network.py's docstring for the column order.
        v_o: (batch, 10) -- the network's raw output for that same batch.

    Returns:
        dict of (horizon, batch) tensors: v_ref_real, v_ref_imag, vc_real, vc_imag, ift_real,
        ift_imag, i_load_real, i_load_imag (the last two broadcast to (batch,), same for all
        horizon steps -- see loss.py's module docstring on why the load is held constant).
    """
    if_alpha, if_beta = x_batch[:, 0], x_batch[:, 1]
    vc_alpha, vc_beta = x_batch[:, 2], x_batch[:, 3]
    vref_alpha, vref_beta = x_batch[:, 4], x_batch[:, 5]
    r = x_batch[:, 6]
    vref_ph = x_batch[:, 7:15]  # [alphaph, betaph, alphaph3, betaph3, alphaph4, betaph4, alphaph5, betaph5]

    # (horizon, batch) -- row k is the reference at control step k+1.
    v_ref_real = tf.stack([vref_alpha, vref_ph[:, 0], vref_ph[:, 2], vref_ph[:, 4], vref_ph[:, 6]], axis=0)
    v_ref_imag = tf.stack([vref_beta, vref_ph[:, 1], vref_ph[:, 3], vref_ph[:, 5], vref_ph[:, 7]], axis=0)

    v_meas_real, v_meas_imag = vc_alpha, vc_beta
    i_meas_real, i_meas_imag = if_alpha, if_beta

    # network output: interleaved (v_o_real, v_o_imag) per step -> (horizon, batch) each.
    v_o_real = tf.transpose(v_o[:, 0::2])
    v_o_imag = tf.transpose(v_o[:, 1::2])

    r_safe = tf.maximum(tf.abs(r), 1e-6)
    i_load_real = vc_alpha / r_safe
    i_load_imag = vc_beta / r_safe

    adf = tf.constant(ADF)
    bdf = tf.constant(BDF)

    ift_real, ift_imag, vc_real, vc_imag = [], [], [], []
    prev_ift_real = prev_ift_imag = prev_vc_real = prev_vc_imag = None
    for k in range(HORIZON):
        i_r = i_meas_real if k == 0 else prev_ift_real
        i_i = i_meas_imag if k == 0 else prev_ift_imag
        v_r = v_meas_real if k == 0 else prev_vc_real
        v_i = v_meas_imag if k == 0 else prev_vc_imag

        step_ift_real = adf[0, 0] * i_r + adf[0, 1] * v_r + bdf[0, 0] * v_o_real[k] + bdf[0, 1] * i_load_real
        step_ift_imag = adf[0, 0] * i_i + adf[0, 1] * v_i + bdf[0, 0] * v_o_imag[k] + bdf[0, 1] * i_load_imag
        step_vc_real = adf[1, 0] * i_r + adf[1, 1] * v_r + bdf[1, 0] * v_o_real[k] + bdf[1, 1] * i_load_real
        step_vc_imag = adf[1, 0] * i_i + adf[1, 1] * v_i + bdf[1, 0] * v_o_imag[k] + bdf[1, 1] * i_load_imag

        ift_real.append(step_ift_real)
        ift_imag.append(step_ift_imag)
        vc_real.append(step_vc_real)
        vc_imag.append(step_vc_imag)
        prev_ift_real, prev_ift_imag = step_ift_real, step_ift_imag
        prev_vc_real, prev_vc_imag = step_vc_real, step_vc_imag

    return {
        "v_ref_real": v_ref_real,
        "v_ref_imag": v_ref_imag,
        "vc_real": tf.stack(vc_real, axis=0),
        "vc_imag": tf.stack(vc_imag, axis=0),
        "ift_real": tf.stack(ift_real, axis=0),
        "ift_imag": tf.stack(ift_imag, axis=0),
        "i_load_real": i_load_real,
        "i_load_imag": i_load_imag,
    }


def dpc_loss(x_batch: tf.Tensor, v_o: tf.Tensor) -> tf.Tensor:
    """
    Args:
        x_batch: (batch, 15) -- see network.py's docstring for the column order.
        v_o: (batch, 10) -- the network's raw output for that same batch.

    Returns:
        Scalar loss, summed over the horizon's 5 steps and both loss terms, averaged over the
        batch (matches Model_gradients.txt's ``loss = loss / numSamples``).
    """
    s = simulate_horizon(x_batch, v_o)

    real_diff = tf.square(s["v_ref_real"] - s["vc_real"])
    imag_diff = tf.square(s["v_ref_imag"] - s["vc_imag"])
    voltage_loss = tf.reduce_sum(real_diff + imag_diff)

    ift_real_diff = tf.square(s["ift_real"] - s["i_load_real"] + CAPACITANCE_F * OMEGA_REF_RAD_S * s["v_ref_imag"])
    ift_imag_diff = tf.square(s["ift_imag"] - s["i_load_imag"] - CAPACITANCE_F * OMEGA_REF_RAD_S * s["v_ref_real"])
    current_loss = tf.reduce_sum(ift_real_diff + ift_imag_diff)

    batch_size = tf.cast(tf.shape(x_batch)[0], tf.float32)
    return (voltage_loss + current_loss) / batch_size
