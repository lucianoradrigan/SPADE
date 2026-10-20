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


class ScheduledDpcController:
    """Two DPC checkpoints, selected by operating point -- TODO_CLAUDE_CODE.md C1, option 1.

    v3 (fine-tuned at the single load 8.0064 ohm) tracks better than v4 near that load and at small
    references, and v4 (fine-tuned over 3.5-1000 ohm) everywhere else
    (docs/audit/stability/dpc_operating_point_selection.md). This controller uses the *nominal*
    checkpoint inside a region of (R, |v_ref|) and the *wide* one outside it. Both inputs are
    already known to the controller: R is one of the network's inputs, and |v_ref| is the
    setpoint. Nothing is retrained.

    Hysteresis: once inside, the region is left only when R or |v_ref| moves more than
    ``hysteresis`` (a fraction of the bound) past it, so a load or setpoint sitting on a boundary
    does not make the controller alternate every step. ``set_load`` and ``set_magnitude`` update
    the operating point at run time (a load step).
    """

    def __init__(self, nominal_weights, wide_weights, r_ohm: float, region: dict, hysteresis: float = 0.05):
        self.nominal = DpcController(nominal_weights, r_ohm)
        self.wide = DpcController(wide_weights, r_ohm)
        self.region = region
        self.hysteresis = hysteresis
        self.r = r_ohm
        self.magnitude_v = None
        self.using_nominal = None
        self.switches = 0

    def reset(self):
        self.using_nominal, self.switches = None, 0

    def set_load(self, r_ohm: float):
        self.r = self.nominal.r = self.wide.r = r_ohm

    def _inside(self, magnitude_v: float, margin: float) -> bool:
        reg = self.region
        r_lo, r_hi = reg["r_min_ohm"] * (1 - margin), reg["r_max_ohm"] * (1 + margin)
        in_band = r_lo <= self.r <= r_hi and magnitude_v <= reg["band_v_max"] * (1 + margin)
        small_ref = magnitude_v <= reg["small_v_max"] * (1 + margin) and reg["small_r_min_ohm"] * (1 - margin) <= self.r <= reg["small_r_max_ohm"] * (1 + margin)
        return in_band or small_ref

    def control(self, state, reference, k: int) -> tuple[float, float]:
        mag = float(reference.magnitude_v)
        if self.using_nominal is None:
            use = self._inside(mag, 0.0)
        elif self.using_nominal:
            use = self._inside(mag, self.hysteresis)
        else:
            use = self._inside(mag, -self.hysteresis)
        if self.using_nominal is not None and use != self.using_nominal:
            self.switches += 1
        self.using_nominal = use
        return (self.nominal if use else self.wide).control(state, reference, k)
