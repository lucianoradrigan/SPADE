"""Phase 1 scaffolding smoke test for ModelValidator."""

from driveflow.ai.transfer.validator import ModelValidator


def test_validator_initialization():
    baseline_metrics = {"accuracy": 0.895, "f1": 0.885}
    thresholds = {"min_accuracy": 0.90, "max_loss_increase": 0.1}
    validator = ModelValidator(baseline_metrics, thresholds)
    assert validator.baseline_metrics == baseline_metrics
    assert validator.thresholds == thresholds
