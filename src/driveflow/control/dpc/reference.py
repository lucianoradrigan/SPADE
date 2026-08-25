"""Rotating reference-voltage vector, matching Data4train.mat's own reference signal: a
constant-magnitude vector rotating at grid frequency (|v_ref|=50.0V, verified directly on the
holdout split -- see docs/macro_fase_B1_dpc.md), not a randomly varying setpoint.

The 5-step horizon the network needs (network.py's ``vref_alphaph``...``vref_alphaph5`` columns)
is just this same known signal evaluated at 4 future sample instants -- since the reference is
exogenous and under our control (unlike the plant state), there is nothing to forecast/estimate.
"""

from dataclasses import dataclass

import numpy as np

from driveflow.sim.vsc_system import TAU

GRID_OMEGA_RAD_S = 2 * np.pi * 50.0
#: Reference magnitude, matching Data4train.mat exactly (see reference.py's module docstring
#: pointer and docs/macro_fase_B1_dpc.md) -- not the ±325V range an earlier abandoned draft
#: (DPC4VSC.py) sampled from.
REFERENCE_MAGNITUDE_V = 50.0


@dataclass
class RotatingReference:
    magnitude_v: float = REFERENCE_MAGNITUDE_V
    omega_rad_s: float = GRID_OMEGA_RAD_S
    tau: float = TAU
    phase0_rad: float = 0.0

    def at_step(self, k: int) -> tuple[float, float]:
        """(v_ref_real, v_ref_imag) at discrete step k (k=0 is the current step)."""
        theta = self.phase0_rad + self.omega_rad_s * self.tau * k
        return self.magnitude_v * np.cos(theta), self.magnitude_v * np.sin(theta)

    def horizon(self, k: int, horizon: int = 5) -> list[tuple[float, float]]:
        """[(v_ref_real, v_ref_imag)] for steps k, k+1, ..., k+horizon-1."""
        return [self.at_step(k + i) for i in range(horizon)]
