"""Macro-fase B: Keras port of DPC4PowerElectronics (differentiable predictive control for a
VSC). See loss.py's docstring for what the model-based loss actually does and network.py's for
the ported architecture; train.py ports Main.txt's training loop.
"""

from .loss import ADF, BDF, CAPACITANCE_F, HORIZON, OMEGA_REF_RAD_S, dpc_loss, simulate_horizon
from .network import N_INPUTS, build_dpc_network
from .train import COLUMNS, load_training_data, train_dpc, train_on_array

__all__ = [
    "ADF",
    "BDF",
    "CAPACITANCE_F",
    "HORIZON",
    "OMEGA_REF_RAD_S",
    "dpc_loss",
    "simulate_horizon",
    "N_INPUTS",
    "build_dpc_network",
    "COLUMNS",
    "load_training_data",
    "train_dpc",
    "train_on_array",
]
