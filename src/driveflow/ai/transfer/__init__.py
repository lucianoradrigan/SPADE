"""Transfer Learning Workbench for SPADE/driveflow.

Fine-tuning de modelos ya promovidos en el registry con datos nuevos (simulados o externos), y
despliegue automático a los tres tiers (PC, Raspberry Pi 5, ESP32). Phase 1: solo scaffolding --
las clases están definidas pero sin implementación (ver NotImplementedError en cada método);
Phase 2 las completa.
"""

from driveflow.ai.transfer.loader import ModelLoader
from driveflow.ai.transfer.data_merger import DataMerger
from driveflow.ai.transfer.pipeline import TransferLearningPipeline
from driveflow.ai.transfer.validator import ModelValidator

__all__ = [
    "ModelLoader",
    "DataMerger",
    "TransferLearningPipeline",
    "ModelValidator",
]
