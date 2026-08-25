"""Regression tests for the Keras/TF port of DPC4PowerElectronics/Custom_Loss.txt.

There is no MATLAB available in this environment to run the original toolbox, and the one
pretrained network found in DPC4PowerElectronics' git history (`trainedNet1nDPCv4_1.mat`) is
serialized as a MATLAB ``dlnetwork`` object -- scipy.io cannot deserialize it, and there is no
MATLAB here to re-export its weights. So this suite validates the *translation* itself, not a
byte-for-byte match against a MATLAB-trained reference:

1. An all-zero equilibrium: with every input and the network output at 0, every intermediate
   term in Custom_Loss.txt is 0 too, so the loss must be exactly 0. Catches basic wiring bugs.
2. A hand-computed single control step, with intentionally distinct v_o_real/v_o_imag values --
   catches interleaving-order bugs in how the network's 10 outputs get split into
   (v_o_real, v_o_imag) per horizon step (Custom_Loss.txt's
   ``s_states(1:2:end,:)``/``s_states(2:2:end,:)``).
3. A from-scratch, independent plain-numpy re-implementation of the full 5-step recursion,
   cross-checked against loss.dpc_loss on random batches -- catches recursion-indexing bugs that
   a single-step test wouldn't reach.
"""

import numpy as np
import tensorflow as tf

from driveflow.control.dpc.loss import ADF, BDF, CAPACITANCE_F, HORIZON, OMEGA_REF_RAD_S, dpc_loss
from driveflow.control.dpc.network import N_INPUTS, build_dpc_network


def _x_batch(if_a=0, if_b=0, vc_a=0, vc_b=0, vref_a=0, vref_b=0, r=50.0, vref_ph=None):
    vref_ph = vref_ph if vref_ph is not None else [0.0] * 8
    row = [if_a, if_b, vc_a, vc_b, vref_a, vref_b, r, *vref_ph]
    assert len(row) == N_INPUTS
    return tf.constant([row], dtype=tf.float32)


class TestDpcLossEquilibrium:
    def test_all_zero_input_and_output_gives_zero_loss(self):
        x = _x_batch()
        v_o = tf.zeros((1, 2 * HORIZON), dtype=tf.float32)
        loss = dpc_loss(x, v_o)
        assert abs(float(loss)) < 1e-8


class TestDpcLossSingleStep:
    def test_interleaving_order_matches_matlab_reshape(self):
        # Only the first horizon step's v_o is nonzero, real != imag, so any real/imag swap or
        # transpose bug in v_o's deinterleaving flips the sign/value of exactly this check.
        if_a, if_b, vc_a, vc_b, r = 3.0, -2.0, 100.0, -40.0, 60.0
        v_o_real_1, v_o_imag_1 = 250.0, -80.0

        i_load_real = vc_a / r
        i_load_imag = vc_b / r
        vc_real_1 = ADF[1, 0] * if_a + ADF[1, 1] * vc_a + BDF[1, 0] * v_o_real_1 + BDF[1, 1] * i_load_real
        vc_imag_1 = ADF[1, 0] * if_b + ADF[1, 1] * vc_b + BDF[1, 0] * v_o_imag_1 + BDF[1, 1] * i_load_imag

        # Set the reference for horizon step 1 to exactly vc_real_1/vc_imag_1 so THAT step's
        # voltage-tracking residual is exactly 0 -- if the port swaps real/imag it won't be.
        x = _x_batch(if_a=if_a, if_b=if_b, vc_a=vc_a, vc_b=vc_b, vref_a=float(vc_real_1), vref_b=float(vc_imag_1), r=r)
        v_o = tf.constant([[v_o_real_1, v_o_imag_1] + [0.0] * (2 * HORIZON - 2)], dtype=tf.float32)

        loss = dpc_loss(x, v_o)
        # Remaining 4 horizon steps still contribute (their v_o is 0, reference is 0, but the
        # recursion carries vc_real_1/vc_imag_1 forward) -- so the loss isn't 0, but if the
        # interleaving were swapped, THIS SPECIFIC pair of numbers would land far from what a
        # correct implementation gives. Assert against an independently hand-computed value.
        expected = _reference_full_loss(if_a, if_b, vc_a, vc_b, vc_real_1, vc_imag_1, r, [v_o_real_1, v_o_imag_1] + [0.0] * 8, np.zeros(8))
        assert np.isclose(float(loss), expected, rtol=1e-4, atol=1e-2)  # float32 vs. float64 reference


class TestDpcLossAgainstIndependentNumpy:
    def test_random_batch_matches_independent_reimplementation(self):
        rng = np.random.default_rng(0)
        batch = 8
        x_np = np.zeros((batch, N_INPUTS), dtype=np.float32)
        x_np[:, 0:2] = rng.uniform(-16, 16, (batch, 2))  # if_alpha, if_beta
        x_np[:, 2:4] = rng.uniform(-10, 10, (batch, 2))  # vc_alpha, vc_beta
        x_np[:, 4:6] = rng.uniform(-325, 325, (batch, 2))  # vref_alpha, vref_beta
        x_np[:, 6] = rng.uniform(30, 60, batch)  # R
        x_np[:, 7:15] = rng.uniform(-325, 325, (batch, 8))  # vref at future horizon steps
        v_o_np = rng.uniform(-300, 300, (batch, 2 * HORIZON)).astype(np.float32)

        tf_loss = float(dpc_loss(tf.constant(x_np), tf.constant(v_o_np)))
        np_loss = _independent_numpy_loss(x_np, v_o_np)
        assert np.isclose(tf_loss, np_loss, rtol=1e-4, atol=1e-4)


def _reference_full_loss(if_a, if_b, vc_a, vc_b, vref_a, vref_b, r, v_o_flat, vref_ph):
    x = np.array([[if_a, if_b, vc_a, vc_b, vref_a, vref_b, r, *vref_ph]], dtype=np.float32)
    v_o = np.array([v_o_flat], dtype=np.float32)
    return _independent_numpy_loss(x, v_o)


def _independent_numpy_loss(x_np: np.ndarray, v_o_np: np.ndarray) -> float:
    """Plain-numpy re-derivation of Custom_Loss.txt, written independently from loss.py's
    TF version (no shared helper code) so it can catch translation bugs loss.py itself would
    reproduce if it re-used the same buggy indexing.
    """
    batch = x_np.shape[0]
    if_alpha, if_beta = x_np[:, 0], x_np[:, 1]
    vc_alpha, vc_beta = x_np[:, 2], x_np[:, 3]
    vref_alpha, vref_beta = x_np[:, 4], x_np[:, 5]
    r = x_np[:, 6]
    vref_ph = x_np[:, 7:15]

    v_ref_real = np.stack([vref_alpha, vref_ph[:, 0], vref_ph[:, 2], vref_ph[:, 4], vref_ph[:, 6]], axis=0)
    v_ref_imag = np.stack([vref_beta, vref_ph[:, 1], vref_ph[:, 3], vref_ph[:, 5], vref_ph[:, 7]], axis=0)

    v_o_real = v_o_np[:, 0::2].T  # (5, batch)
    v_o_imag = v_o_np[:, 1::2].T

    r_safe = np.maximum(np.abs(r), 1e-6)
    i_load_real = vc_alpha / r_safe
    i_load_imag = vc_beta / r_safe

    ift_real = np.zeros((5, batch))
    ift_imag = np.zeros((5, batch))
    vc_real = np.zeros((5, batch))
    vc_imag = np.zeros((5, batch))

    ift_real[0] = ADF[0, 0] * if_alpha + ADF[0, 1] * vc_alpha + BDF[0, 0] * v_o_real[0] + BDF[0, 1] * i_load_real
    ift_imag[0] = ADF[0, 0] * if_beta + ADF[0, 1] * vc_beta + BDF[0, 0] * v_o_imag[0] + BDF[0, 1] * i_load_imag
    vc_real[0] = ADF[1, 0] * if_alpha + ADF[1, 1] * vc_alpha + BDF[1, 0] * v_o_real[0] + BDF[1, 1] * i_load_real
    vc_imag[0] = ADF[1, 0] * if_beta + ADF[1, 1] * vc_beta + BDF[1, 0] * v_o_imag[0] + BDF[1, 1] * i_load_imag

    for k in range(1, 5):
        ift_real[k] = ADF[0, 0] * ift_real[k - 1] + ADF[0, 1] * vc_real[k - 1] + BDF[0, 0] * v_o_real[k] + BDF[0, 1] * i_load_real
        ift_imag[k] = ADF[0, 0] * ift_imag[k - 1] + ADF[0, 1] * vc_imag[k - 1] + BDF[0, 0] * v_o_imag[k] + BDF[0, 1] * i_load_imag
        vc_real[k] = ADF[1, 0] * ift_real[k - 1] + ADF[1, 1] * vc_real[k - 1] + BDF[1, 0] * v_o_real[k] + BDF[1, 1] * i_load_real
        vc_imag[k] = ADF[1, 0] * ift_imag[k - 1] + ADF[1, 1] * vc_imag[k - 1] + BDF[1, 0] * v_o_imag[k] + BDF[1, 1] * i_load_imag

    voltage_loss = np.sum((v_ref_real - vc_real) ** 2 + (v_ref_imag - vc_imag) ** 2)
    current_loss = np.sum(
        (ift_real - i_load_real + CAPACITANCE_F * OMEGA_REF_RAD_S * v_ref_imag) ** 2
        + (ift_imag - i_load_imag - CAPACITANCE_F * OMEGA_REF_RAD_S * v_ref_real) ** 2
    )
    return float((voltage_loss + current_loss) / batch)


class TestDpcNetwork:
    def test_output_shape_matches_two_times_horizon(self):
        net = build_dpc_network(horizon=HORIZON)
        out = net(tf.zeros((4, N_INPUTS)))
        assert out.shape == (4, 2 * HORIZON)

    def test_hidden_layer_widths_match_matlab_topology(self):
        net = build_dpc_network()
        # featureInputLayer(15) -> fc(16) -> relu -> fc(16) -> relu -> fc(10)
        dense_layers = [layer for layer in net.layers if isinstance(layer, tf.keras.layers.Dense)]
        assert [layer.units for layer in dense_layers] == [16, 16, 2 * HORIZON]
