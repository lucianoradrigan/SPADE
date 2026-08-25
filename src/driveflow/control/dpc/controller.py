"""Deploys the trained DPC network as a receding-horizon controller: at each real control step,
predict the full 5-step v_o sequence (same shape the network was trained to produce) but apply
only the FIRST step's action -- standard MPC/DPC deployment practice (re-predict every step from
the freshly measured state, discard the rest of the horizon). This is a different regime from
training: training saw i.i.d. random (state, reference) snapshots (Data4train.mat's rows), never
a real multi-step closed loop where the controller's own actions shape the next state it sees --
see evaluate_dpc_closed_loop.py for why that distinction is validated separately from the
holdout metrics in docs/macro_fase_B1_dpc.md.
"""

import tensorflow as tf

from .network import HORIZON, N_INPUTS, build_dpc_network
from .reference import RotatingReference
from driveflow.sim.vsc_system import VscState


class DpcController:
    def __init__(self, weights_path, r_ohm: float):
        self.model = build_dpc_network()
        self.model(tf.zeros((1, N_INPUTS)))  # build before loading weights
        self.model.load_weights(weights_path)
        self.r = r_ohm

    def reset(self):
        pass  # stateless controller -- every step is a fresh forward pass, no integrator/memory

    def control(self, state: VscState, reference: RotatingReference, k: int) -> tuple[float, float]:
        """Returns (v_o_real, v_o_imag) -- the first horizon step's action only."""
        horizon_refs = reference.horizon(k, horizon=HORIZON)
        row = [state.i_f_real, state.i_f_imag, state.vc_real, state.vc_imag, horizon_refs[0][0], horizon_refs[0][1], self.r]
        for vr, vi in horizon_refs[1:]:
            row += [vr, vi]
        x = tf.constant([row], dtype=tf.float32)
        v_o = self.model(x, training=False).numpy()[0]
        return float(v_o[0]), float(v_o[1])
