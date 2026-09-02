"""Tests models/regressors/schemas.py + builder.py (docs/design_ai_layer_transversal.md Sec. 6.1/
6.3, Sec. 8 step 4) -- in particular the Sec. 9 guardrail: an ESP32-tier config must not be able
to declare recurrent_type: lstm (or gru), and must fail at load time, not once someone tries to
deploy it.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

import pytest

from driveflow.models.regressors.builder import build_forecaster
from driveflow.models.regressors.schemas import ForecasterConfig, ForecasterConfigError, load_forecaster_config

PC_FULL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "regressors" / "pc_full.yaml"


def _base_kwargs(**overrides):
    kwargs = dict(
        domain="vsc_dpc",
        tier="pc",
        input_window=32,
        horizon=8,
        recurrent_type="lstm",
        layers=(16,),
        use_attention=False,
    )
    kwargs.update(overrides)
    return kwargs


class TestForecasterConfigValidation:
    def test_valid_config_constructs(self):
        ForecasterConfig(**_base_kwargs())

    def test_unknown_tier_rejected(self):
        with pytest.raises(ForecasterConfigError, match="tier"):
            ForecasterConfig(**_base_kwargs(tier="raspberry"))

    def test_unknown_recurrent_type_rejected(self):
        with pytest.raises(ForecasterConfigError, match="recurrent_type"):
            ForecasterConfig(**_base_kwargs(recurrent_type="transformer"))

    @pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
    def test_esp32_tier_rejects_recurrent_types(self, recurrent_type):
        """The mandatory guardrail (Sec. 9): ESP32 cannot express a recurrent architecture at
        all, not just LSTM -- fails at load/construction time, not at deploy time."""
        with pytest.raises(ForecasterConfigError, match="esp32.*cannot use recurrent_type"):
            ForecasterConfig(**_base_kwargs(tier="esp32", recurrent_type=recurrent_type, layers=(4,)))

    def test_esp32_tier_accepts_none(self):
        ForecasterConfig(**_base_kwargs(tier="esp32", recurrent_type="none", layers=()))

    @pytest.mark.parametrize("tier,recurrent_type", [("pc", "gru"), ("rpi5", "lstm"), ("rpi5", "gru")])
    def test_pc_and_rpi5_accept_lstm_and_gru(self, tier, recurrent_type):
        ForecasterConfig(**_base_kwargs(tier=tier, recurrent_type=recurrent_type, layers=(8,)))

    def test_empty_layers_rejected_for_recurrent_type(self):
        with pytest.raises(ForecasterConfigError, match="layers"):
            ForecasterConfig(**_base_kwargs(layers=()))

    def test_non_positive_horizon_rejected(self):
        with pytest.raises(ForecasterConfigError, match="horizon"):
            ForecasterConfig(**_base_kwargs(horizon=0))


class TestLoadForecasterConfigRejectsMalformedFiles:
    def test_esp32_with_lstm_raises_at_load_time(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "domain: vsc_dpc\ntier: esp32\ninput_window: 32\nhorizon: 8\n"
            "recurrent_type: lstm\nlayers: [16]\n"
        )
        with pytest.raises(ForecasterConfigError, match="cannot use recurrent_type"):
            load_forecaster_config(path)

    def test_missing_recurrent_type_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("domain: vsc_dpc\ntier: pc\ninput_window: 32\nhorizon: 8\nlayers: [16]\n")
        with pytest.raises(ForecasterConfigError, match="malformed config"):
            load_forecaster_config(path)


class TestShippedPcFullConfig:
    def test_loads_and_validates(self):
        config = load_forecaster_config(PC_FULL_CONFIG_PATH)
        assert config.domain == "vsc_dpc"
        assert config.tier == "pc"
        assert config.recurrent_type == "lstm"


class TestBuildForecaster:
    def test_output_shape_is_horizon_by_channels(self):
        config = ForecasterConfig(**_base_kwargs())
        model = build_forecaster(config, n_channels=3)
        out = model(tf.zeros((2, config.input_window, 3)))
        assert out.shape == (2, config.horizon, 3)

    def test_gru_variant_builds(self):
        config = ForecasterConfig(**_base_kwargs(recurrent_type="gru"))
        model = build_forecaster(config, n_channels=2)
        out = model(tf.zeros((1, config.input_window, 2)))
        assert out.shape == (1, config.horizon, 2)

    def test_attention_variant_builds(self):
        config = ForecasterConfig(**_base_kwargs(use_attention=True, layers=(16, 8)))
        model = build_forecaster(config, n_channels=2)
        out = model(tf.zeros((1, config.input_window, 2)))
        assert out.shape == (1, config.horizon, 2)

    def test_multi_layer_stack_builds(self):
        config = ForecasterConfig(**_base_kwargs(layers=(16, 8, 4)))
        model = build_forecaster(config, n_channels=1)
        out = model(tf.zeros((1, config.input_window, 1)))
        assert out.shape == (1, config.horizon, 1)

    def test_none_recurrent_type_not_implemented_yet(self):
        config = ForecasterConfig(**_base_kwargs(tier="esp32", recurrent_type="none", layers=()))
        with pytest.raises(NotImplementedError, match="recurrent_type"):
            build_forecaster(config, n_channels=2)

    def test_shipped_pc_full_config_builds(self):
        config = load_forecaster_config(PC_FULL_CONFIG_PATH)
        model = build_forecaster(config, n_channels=4)
        out = model(tf.zeros((1, config.input_window, 4)))
        assert out.shape == (1, config.horizon, 4)

    def test_output_is_finite(self):
        """Sanity check against a silently broken graph (e.g. a bad Reshape target size) -- a
        forward pass on random input should stay finite even with untrained weights."""
        config = ForecasterConfig(**_base_kwargs())
        model = build_forecaster(config, n_channels=3)
        out = model(tf.random.normal((4, config.input_window, 3)))
        assert np.all(np.isfinite(out.numpy()))
