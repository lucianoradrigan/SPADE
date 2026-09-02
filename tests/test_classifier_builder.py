"""Tests models/classifiers/schemas.py + builder.py (docs/design_ai_layer_transversal.md Sec. 6.1/
6.2, Sec. 8 step 4): a malformed config must fail at load time (schema), and a valid config must
build a working keras.Model with the right shapes -- same "fail at load, not at use" standard as
tests/test_monitoring_rules_schema.py.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

import pytest

from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import (
    ClassifierConfig,
    ClassifierConfigError,
    ConvBlockConfig,
    load_classifier_config,
)

PC_SERVER_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "classifiers" / "pc_server.yaml"


def _base_kwargs(**overrides):
    kwargs = dict(
        domain="dc_motor",
        tier="pc",
        input_window=64,
        num_classes=3,
        blocks=(ConvBlockConfig(filters=8, kernel_size=3),),
        dense_units=(16,),
        dropout=0.2,
    )
    kwargs.update(overrides)
    return kwargs


class TestConvBlockConfig:
    def test_valid_block_constructs(self):
        ConvBlockConfig(filters=8, kernel_size=3, use_se=True)

    def test_non_positive_filters_rejected(self):
        with pytest.raises(ClassifierConfigError, match="filters"):
            ConvBlockConfig(filters=0, kernel_size=3)

    def test_non_positive_kernel_size_rejected(self):
        with pytest.raises(ClassifierConfigError, match="kernel_size"):
            ConvBlockConfig(filters=8, kernel_size=0)


class TestClassifierConfigValidation:
    def test_valid_config_constructs(self):
        ClassifierConfig(**_base_kwargs())

    def test_unknown_tier_rejected(self):
        with pytest.raises(ClassifierConfigError, match="tier"):
            ClassifierConfig(**_base_kwargs(tier="raspberry"))

    def test_empty_blocks_rejected(self):
        with pytest.raises(ClassifierConfigError, match="blocks"):
            ClassifierConfig(**_base_kwargs(blocks=()))

    def test_too_few_classes_rejected(self):
        with pytest.raises(ClassifierConfigError, match="num_classes"):
            ClassifierConfig(**_base_kwargs(num_classes=1))

    def test_dropout_out_of_range_rejected(self):
        with pytest.raises(ClassifierConfigError, match="dropout"):
            ClassifierConfig(**_base_kwargs(dropout=1.0))


class TestLoadClassifierConfigRejectsMalformedFiles:
    def test_unknown_block_type_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "domain: dc_motor\ntier: pc\ninput_window: 64\nnum_classes: 3\n"
            "blocks:\n  - type: transformer\n    filters: 8\n    kernel_size: 3\n"
        )
        with pytest.raises(ClassifierConfigError, match="block type"):
            load_classifier_config(path)

    def test_missing_blocks_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("domain: dc_motor\ntier: pc\ninput_window: 64\nnum_classes: 3\n")
        with pytest.raises(ClassifierConfigError, match="blocks"):
            load_classifier_config(path)

    def test_unknown_top_level_field_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "domain: dc_motor\ntier: pc\ninput_window: 64\nnum_classes: 3\nbogus_field: 1\n"
            "blocks:\n  - type: conv1d\n    filters: 8\n    kernel_size: 3\n"
        )
        with pytest.raises(ClassifierConfigError, match="malformed config"):
            load_classifier_config(path)


class TestShippedPcServerConfig:
    def test_loads_and_validates(self):
        config = load_classifier_config(PC_SERVER_CONFIG_PATH)
        assert config.domain == "dc_motor"
        assert config.tier == "pc"
        assert len(config.blocks) == 3


class TestBuildClassifier:
    def test_output_shape_is_softmax_over_classes(self):
        config = ClassifierConfig(**_base_kwargs())
        model = build_classifier(config, n_channels=4)
        out = model(tf.zeros((2, config.input_window, 4)))
        assert out.shape == (2, 3)
        assert np.allclose(np.sum(out.numpy(), axis=1), 1.0, atol=1e-5)

    def test_multi_block_config_builds(self):
        config = ClassifierConfig(
            **_base_kwargs(blocks=(ConvBlockConfig(filters=8, kernel_size=3, use_se=True), ConvBlockConfig(filters=16, kernel_size=3)))
        )
        model = build_classifier(config, n_channels=2)
        out = model(tf.zeros((1, config.input_window, 2)))
        assert out.shape == (1, 3)

    def test_shipped_pc_server_config_builds(self):
        config = load_classifier_config(PC_SERVER_CONFIG_PATH)
        model = build_classifier(config, n_channels=5)
        out = model(tf.zeros((1, config.input_window, 5)))
        assert out.shape == (1, config.num_classes)
