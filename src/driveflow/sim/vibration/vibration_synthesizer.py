"""Orchestrates Module B into a single per-step call. Per docs/addendum_vibracion_v1.md Sec. 4:
this is invoked manually after each ``SCMLSystem.simulate()``, e.g. from ``datagen/runner.py``
(Macro-fase A.4). It is deliberately NOT a ``PhysicalSystemWrapper`` -- driveflow does not use
GEM's gymnasium.Env wrapper mechanism (Principle of design #2).

Module C (data-driven residual correction) was retired -- see docs/patch2_retiro_modulo_C.md.
``vibration_source`` is always ``"synthetic_b"``.
"""

from .modal_model import ModalVibrationModel


class VibrationSynthesizer:
    """Se invoca manualmente tras cada SCMLSystem.simulate(), NO es un
    PhysicalSystemWrapper (driveflow no usa la capa gymnasium.Env de GEM)."""

    #: Always "synthetic_b": Module C (the "synthetic_b_plus_c" alternative) was retired, see
    #: docs/patch2_retiro_modulo_C.md. Kept as a property (not a bare constant) so the dataset
    #: schema's vibration_source column has a stable place to read it from regardless of how
    #: Module B's excitation sources evolve.
    vibration_source = "synthetic_b"

    def __init__(self, modal_model: ModalVibrationModel, tau: float):
        self.modal_model = modal_model
        self.tau = tau

    def reset(self, seed=None):
        self.modal_model.reset(seed=seed)

    def set_fault(self, fault_type: str = None, severity: float = 0.0, order_override: float = None, extra_faults: list = None):
        self.modal_model.set_fault(fault_type, severity, order_override=order_override, extra_faults=extra_faults)

    def step(self, omega: float) -> tuple:
        return tuple(self.modal_model.step(omega, dt=self.tau))
