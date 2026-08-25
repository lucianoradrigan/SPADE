"""Orchestrates simulation runs (SCMLSystem + a controller + BearingFaultLoad + VibrationSynthesizer)
into a labeled, Parquet-exported dataset. Macro-fase A.4 of INSTRUCTIONS.md.
"""

from .export_parquet import export_parquet
from .fault_injection import BearingFaultLoad
from .runner import build_plant, build_vibration_synthesizer, run_flow, run_scenario, run_scenarios
from .scenario import Scenario, healthy_and_faulted_grid

__all__ = [
    "BearingFaultLoad",
    "Scenario",
    "healthy_and_faulted_grid",
    "build_plant",
    "build_vibration_synthesizer",
    "run_scenario",
    "run_scenarios",
    "run_flow",
    "export_parquet",
]
