"""Transfer Learning Pipeline: orchestrates fine-tuning a promoted model against new data.

Phase 1 scaffolding only -- every method below raises NotImplementedError. Phase 2 implements
against the existing models/classifiers|regressors builders (keras.Model), not a new model
format.
"""

from __future__ import annotations

from enum import Enum


class FineTuneStrategy(Enum):
    FEATURE_EXTRACTOR = "feature_extractor"
    DISCRIMINATIVE_LR = "discriminative_lr"
    ADAPTER = "adapter"


class TransferLearningPipeline:
    """Orquesta: cargar modelo base -> fusionar datos -> fine-tune -> validar -> (si pasa)
    promover -> distilar automáticamente a los tiers edge. Cada paso es un método separado, no
    un único `run()` monolítico, para que Phase 3 (UI) pueda mostrar progreso por etapa."""

    def __init__(self, base_model_path: str, strategy: FineTuneStrategy = FineTuneStrategy.FEATURE_EXTRACTOR, train_config: dict | None = None):
        self.base_model_path = base_model_path
        self.strategy = strategy
        self.train_config = train_config or {}
        self.model = None
        self.history = None

    def load_base_model(self) -> None:
        raise NotImplementedError("Phase 2")

    def freeze_backbone(self, n_layers: int) -> None:
        """Estrategia FEATURE_EXTRACTOR: congela las primeras n_layers capas del modelo base."""
        raise NotImplementedError("Phase 2")

    def apply_discriminative_lr(self, base_lr: float, layer_multipliers: dict) -> None:
        """Estrategia DISCRIMINATIVE_LR: learning rate distinto por capa."""
        raise NotImplementedError("Phase 2")

    def train(self, train_data: dict, val_data: dict, epochs: int = 50, batch_size: int = 32) -> dict:
        raise NotImplementedError("Phase 2")

    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError("Phase 2")
