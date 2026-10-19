"""Validation bench: reusable building blocks for the stability/performance test battery
(tests/simulation, tests/control) and experiments/run_stability_battery.py.

Nothing here changes a physical model or a controller -- it only builds the existing plants with a
selectable ODE solver / step, injects documented disturbances, and measures responses, so the same
code produces both the pytest assertions and the numbers quoted in
docs/SIMULATION_STABILITY_REPORT.md.
"""

from .step_metrics import StepMetrics, step_response_metrics, tracking_metrics
from .numerics import dc_motor_linear_matrix, euler_spectral_radius, pmsm_dq_matrix, euler_max_stable_speed_pmsm
from .benches import (
    LoadTorqueStep,
    build_dc_system,
    run_dc_closed_loop,
    run_dc_open_loop,
    run_pmsm_open_loop,
    pmsm_short_circuit_current,
    run_pmsm_current_step,
    run_vsc_closed_loop,
)

__all__ = [
    "StepMetrics",
    "step_response_metrics",
    "tracking_metrics",
    "dc_motor_linear_matrix",
    "euler_spectral_radius",
    "pmsm_dq_matrix",
    "euler_max_stable_speed_pmsm",
    "LoadTorqueStep",
    "build_dc_system",
    "run_dc_closed_loop",
    "run_dc_open_loop",
    "run_pmsm_open_loop",
    "pmsm_short_circuit_current",
    "run_pmsm_current_step",
    "run_vsc_closed_loop",
]
