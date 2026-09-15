"""Model Validator: sanity checks a fine-tuned model must pass before it's eligible for promotion.

Phase 1 scaffolding only -- every method below raises NotImplementedError.
"""

from __future__ import annotations


class ModelValidator:
    """Corre los checks de sanidad post-fine-tune (accuracy no cae debajo de un umbral,
    convergencia de la loss, sin colapso de modo) antes de dejar promover un run."""

    def __init__(self, baseline_metrics: dict, thresholds: dict):
        self.baseline_metrics = baseline_metrics
        self.thresholds = thresholds
        self.validation_results: dict = {}

    def validate(self, finetuned_model, val_data: dict) -> tuple[bool, dict]:
        """Corre todos los checks. Devuelve (paso_todo, resultados_por_check)."""
        raise NotImplementedError("Phase 2")

    def check_accuracy_threshold(self, new_accuracy: float) -> bool:
        raise NotImplementedError("Phase 2")

    def check_convergence(self, loss_history: list) -> bool:
        raise NotImplementedError("Phase 2")

    def check_mode_collapse(self, predictions: list) -> bool:
        raise NotImplementedError("Phase 2")
