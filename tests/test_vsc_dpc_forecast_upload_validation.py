"""Tests viz/dpc_upload_validation.py::validate_vsc_dpc_forecast_upload -- the vsc_dpc-domain
sibling of validate_dc_motor_upload, both now delegating to _validate_channel_based_upload
(docs/design_ai_layer_transversal.md Sec. 8 steps 3/5/7). Mirrors
tests/test_dc_motor_upload_validation.py's cases with the vsc_dpc channel list instead.
"""

import pandas as pd
import pytest

from driveflow.models.common.windowing import VSC_DPC_CANDIDATE_CHANNELS
from driveflow.viz.dpc_upload_validation import DEFAULT_MAGNITUDE_BOUND, validate_vsc_dpc_forecast_upload


def _valid_row(**overrides):
    row = {"vc_real": 45.0, "vc_imag": -5.0, "i_f_real": 2.0, "i_f_imag": -1.0}
    row.update(overrides)
    return row


class TestNoKnownChannelPresent:
    def test_no_known_channels_returns_controlled_error_not_exception(self):
        df = pd.DataFrame([{"unrelated_column": 1.0}])
        df_clean, missing_channels, messages = validate_vsc_dpc_forecast_upload(df)
        assert df_clean is None
        assert missing_channels is None
        assert any(level == "error" for level, _ in messages)


class TestValidUpload:
    def test_fully_valid_upload_passes_through_with_no_messages(self):
        df = pd.DataFrame([_valid_row(), _valid_row(vc_real=46.0)])
        df_clean, missing_channels, messages = validate_vsc_dpc_forecast_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 2
        assert missing_channels == []
        assert messages == []

    def test_single_channel_present_is_enough(self):
        df = pd.DataFrame([{"vc_real": 45.0}, {"vc_real": 46.0}])
        df_clean, missing_channels, messages = validate_vsc_dpc_forecast_upload(df)
        assert df_clean is not None
        assert set(missing_channels) == set(VSC_DPC_CANDIDATE_CHANNELS) - {"vc_real"}
        assert list(df_clean.columns) == ["vc_real"]


class TestNonNumericRows:
    def test_non_numeric_value_drops_that_row_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(vc_real="not_a_number")])
        df_clean, _, messages = validate_vsc_dpc_forecast_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "non-numeric" in text for level, text in messages)


class TestPhysicallyUnreasonableRanges:
    def test_absurd_magnitude_is_dropped_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(vc_real=1e9)])
        df_clean, _, messages = validate_vsc_dpc_forecast_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "reasonable ranges" in text for level, text in messages)

    def test_negative_values_are_not_rejected(self):
        df = pd.DataFrame([_valid_row(vc_imag=-100.0)])
        assert 100.0 < DEFAULT_MAGNITUDE_BOUND
        df_clean, _, messages = validate_vsc_dpc_forecast_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
