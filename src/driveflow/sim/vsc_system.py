"""Discrete-time plant for the Voltage Source Converter (VSC) that DPC4PowerElectronics'
network was trained against -- state=[i_f, v_c] (filter current, capacitor voltage), driven
directly by the same ``Adf``/``Bdf`` matrices used in control/dpc/loss.py's training loss.

This is deliberately NOT built on SCMLSystem (sim/scml_system.py): that class assumes a
Converter -> Motor -> MechanicalLoad chain with abc/dq transforms tied to a rotor angle, none of
which exists here -- a VSC feeding an LCL filter into a resistive load has no motor, no mechanical
side, no rotor. Reusing SCMLSystem's machinery would mean forcing an unrelated physical system
through motor-shaped abstractions. This plant reuses ADF/BDF directly instead: they ARE the
identified discrete-time model (see control/dpc/loss.py's docstring), so simulating with them
*is* simulating the same plant the network was trained to control -- not a re-derived
approximation of it.

Sample period: not documented anywhere in DPC4PowerElectronics -- inferred empirically from
Data4train.mat's own horizon columns (the reference vector's phase advances by a highly
consistent ~0.032 rad per horizon step across all 10000 rows, std ~0), which matches
2*pi*50*TAU for TAU=1e-4s to within ~2%. That's also driveflow's own TAU (sim/scml_system.py,
datagen/runner.py), so this plant reuses it rather than introducing a second unrelated constant.
"""

from dataclasses import dataclass

import numpy as np

from driveflow.control.dpc.loss import ADF, BDF

TAU = 1e-4


def load_feedback_spectral_radius(r_ohm: float) -> float:
    """Spectral radius of the plant's autonomous dynamics from the resistive-load feedback alone
    (``i_load = vc/R`` folded into the state matrix, controller action v_o held at 0) -- i.e.
    whether this discrete-time plant is open-loop stable at this R, BEFORE any controller acts.

    Root cause of the closed-loop divergence documented in tests/test_dpc_robustness_grid.py: this
    crosses 1.0 at R* ~= 3.3707 ohm (solved numerically, not a round number) -- below that R, the
    identified Adf/Bdf model is open-loop unstable by itself (spectral radius grows from ~1.25 at
    R=3.0ohm to ~12.16 at R=0.5ohm), independent of which network is controlling it. This is why
    v1/v2/v3 all diverge there regardless of training data: no fine-tuning of a 1-step-ahead
    receding-horizon controller trained at a single fixed R can be expected to tame an
    already-unstable real pole of that magnitude. See MIN_STABLE_LOAD_RESISTANCE_OHM.
    """
    adf, bdf = ADF.astype(np.float64), BDF.astype(np.float64)
    a_eff = np.array(
        [
            [adf[0, 0], adf[0, 1] + bdf[0, 1] / r_ohm],
            [adf[1, 0], adf[1, 1] + bdf[1, 1] / r_ohm],
        ]
    )
    return float(np.max(np.abs(np.linalg.eigvals(a_eff))))


#: Below this R, load_feedback_spectral_radius(R) > 1 -- the plant is open-loop unstable
#: regardless of the controller (see that function's docstring for the analytic R* ~= 3.3707ohm
#: this crossing happens at). 4.0 adds ~19% margin over R* rather than sitting right at the
#: boundary. This is the floor viz/dashboard.py uses for the "Load resistance" slider, and what
#: tests/test_dpc_robustness_grid.py checks stays a safe margin above R*.
MIN_STABLE_LOAD_RESISTANCE_OHM = 4.0


@dataclass
class VscState:
    i_f_real: float
    i_f_imag: float
    vc_real: float
    vc_imag: float


class VscSystem:
    """reset()/simulate(v_o_real, v_o_imag) plant, mirroring SCMLSystem's interface shape
    without inheriting from it (see module docstring for why).
    """

    def __init__(self, load_resistance_ohm: float = 8.0064, tau: float = TAU):
        """
        Args:
            load_resistance_ohm: R in i_load = v_c/R. Default matches the constant R found in
                Data4train.mat's holdout split (see docs/macro_fase_B1_dpc.md) -- not an
                arbitrary round number.
            tau: control/sample period (s). See module docstring for how this was inferred.
        """
        self.r = load_resistance_ohm
        self.tau = tau
        self._adf = ADF.astype(np.float64)
        self._bdf = BDF.astype(np.float64)
        self.state: VscState = None

    def reset(self, i_f_real=0.0, i_f_imag=0.0, vc_real=0.0, vc_imag=0.0) -> VscState:
        self.state = VscState(i_f_real, i_f_imag, vc_real, vc_imag)
        return self.state

    def simulate(self, v_o_real: float, v_o_imag: float) -> VscState:
        """One discrete step: [i_f, v_c] <- Adf @ [i_f, v_c] + Bdf @ [v_o, i_load]."""
        s = self.state
        i_load_real = s.vc_real / self.r
        i_load_imag = s.vc_imag / self.r

        adf, bdf = self._adf, self._bdf
        new_i_f_real = adf[0, 0] * s.i_f_real + adf[0, 1] * s.vc_real + bdf[0, 0] * v_o_real + bdf[0, 1] * i_load_real
        new_i_f_imag = adf[0, 0] * s.i_f_imag + adf[0, 1] * s.vc_imag + bdf[0, 0] * v_o_imag + bdf[0, 1] * i_load_imag
        new_vc_real = adf[1, 0] * s.i_f_real + adf[1, 1] * s.vc_real + bdf[1, 0] * v_o_real + bdf[1, 1] * i_load_real
        new_vc_imag = adf[1, 0] * s.i_f_imag + adf[1, 1] * s.vc_imag + bdf[1, 0] * v_o_imag + bdf[1, 1] * i_load_imag

        self.state = VscState(new_i_f_real, new_i_f_imag, new_vc_real, new_vc_imag)
        return self.state
