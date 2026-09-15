"""Phase 1 scaffolding smoke tests for TransferLearningPipeline."""

from driveflow.ai.transfer.pipeline import FineTuneStrategy, TransferLearningPipeline


def test_pipeline_initialization():
    pipeline = TransferLearningPipeline(base_model_path="/path/to/model.weights.h5", strategy=FineTuneStrategy.FEATURE_EXTRACTOR)
    assert pipeline.strategy == FineTuneStrategy.FEATURE_EXTRACTOR
    assert pipeline.model is None


def test_pipeline_accepts_every_strategy():
    for strategy in FineTuneStrategy:
        pipeline = TransferLearningPipeline(base_model_path="/path/to/model.weights.h5", strategy=strategy)
        assert pipeline.strategy == strategy
