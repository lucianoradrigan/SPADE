"""Keras port of the DPC network architecture from DPC4PowerElectronics/Main.txt::

    layers = [
        featureInputLayer(15)
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(10)
    ];
    net = dlnetwork(layers);

Input (15 features, DPC4PowerElectronics/Custom_Loss.txt's XBatch row order): if_alpha, if_beta,
vc_alpha, vc_beta, vref_alpha, vref_beta, R, then the reference voltage's (alpha, beta) pair at
each of the 4 remaining horizon steps (vref_alphaph/betaph, ...ph3, ...ph4, ...ph5).

Output (10 = 2*horizon): interleaved (v_o_real, v_o_imag) pairs, one per horizon step -- see
loss.py's docstring for how Custom_Loss.txt reshapes this same layout (``s_states(1:2:end,:)``
/ ``s_states(2:2:end,:)``) into the two (horizon, batch) arrays it recursively unrolls.
"""

import keras

N_INPUTS = 15
HORIZON = 5


def build_dpc_network(horizon: int = HORIZON) -> keras.Model:
    """Same topology/activations as the MATLAB dlnetwork, output width generalized to
    2*horizon (MATLAB hardcodes 10 because it hardcodes horizon=5 throughout Custom_Loss.txt).
    """
    inputs = keras.Input(shape=(N_INPUTS,), name="dpc_input")
    x = keras.layers.Dense(16, activation="relu", name="fc1")(inputs)
    x = keras.layers.Dense(16, activation="relu", name="fc2")(x)
    outputs = keras.layers.Dense(2 * horizon, activation=None, name="v_o")(x)
    return keras.Model(inputs, outputs, name="dpc_mlp")
