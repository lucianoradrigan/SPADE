"""Transfer Learning Workbench for SPADE/driveflow.

Fine-tuning de modelos ya promovidos en el registry con datos nuevos (simulados o externos).
Phase 2: loader/data_merger/pipeline/validator implementados de verdad, sobre la API real de
ai/registry.py y models/*/builder.py -- no una copia paralela de esos módulos. Despliegue
automático a los tres tiers (distillation + TFLite export) sigue siendo trabajo de
experiments/train_model.py --distill + experiments/export_tflite.py (ya existentes, Sec. 8 step
8) invocados sobre el checkpoint que este módulo produce -- Phase 3 conecta eso a una UI, no está
hecho todavía. La pestaña del dashboard (Phase 3) tampoco.
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
