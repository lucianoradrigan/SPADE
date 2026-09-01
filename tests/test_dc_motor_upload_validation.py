"""Tests viz/dpc_upload_validation.py::validate_dc_motor_upload -- the dc_motor-domain sibling of
validate_dpc_upload (docs/design_ai_layer_transversal.md Sec. 5.2, Sec. 8 step 3). Same
"controlled error, not an exception" standard as tests/test_dpc_upload_validation.py, but this
domain's schema is "at least one known channel", not "all N required columns" -- see that
module's docstring for why the two differ.
"""

import pandas as pd
import pytest

from driveflow.models.common.windowing import CANDIDATE_CHANNELS
from driveflow.viz.dpc_upload_validation import DEFAULT_MAGNITUDE_BOUND, validate_dc_motor_upload


def _valid_row(**overrides):
    row = {
        "acc_x": 0.01, "acc_y": -0.02, "acc_z": 0.03,
        "current_r": 5.0, "current_s": -2.5, "current_t": -2.5,
        "rpm": 1200.0, "torque_nm": 3.5,
    }
    row.update(overrides)
    return row


class TestNoKnownChannelPresent:
    def test_no_known_channels_returns_controlled_error_not_exception(self):
        df = pd.DataFrame([{"unrelated_column": 1.0}])
        df_clean, missing_channels, messages = validate_dc_motor_upload(df)  # must not raise
        assert df_clean is None
        assert missing_channels is None
        assert any(level == "error" for level, _ in messages)


class TestValidUpload:
    def test_fully_valid_upload_passes_through_with_no_messages(self):
        df = pd.DataFrame([_valid_row(), _valid_row(rpm=1250.0)])
        df_clean, missing_channels, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 2
        assert missing_channels == []
        assert messages == []

    def test_column_order_and_whitespace_do_not_matter(self):
        row = _valid_row()
        shuffled_keys = list(row.keys())[::-1]
        df = pd.DataFrame([{f" {k} ": row[k] for k in shuffled_keys}])
        df_clean, missing_channels, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert missing_channels == []

    def test_single_channel_present_is_enough(self):
        """Unlike DPC's fixed 7-column requirement, dc_motor has no trained model yet to fix a
        required input schema -- each candidate channel is independently optional."""
        df = pd.DataFrame([{"acc_x": 0.01}, {"acc_x": 0.02}])
        df_clean, missing_channels, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 2
        assert set(missing_channels) == set(CANDIDATE_CHANNELS) - {"acc_x"}
        assert list(df_clean.columns) == ["acc_x"]

    def test_extra_unknown_columns_are_ignored(self):
        df = pd.DataFrame([_valid_row(extra_sensor=123.0)])
        df_clean, missing_channels, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert "extra_sensor" not in df_clean.columns


class TestNonNumericRows:
    def test_non_numeric_value_drops_that_row_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(rpm="not_a_number")])
        df_clean, _, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "non-numeric" in text for level, text in messages)

    def test_all_rows_non_numeric_is_a_controlled_error(self):
        df = pd.DataFrame([_valid_row(rpm="garbage")])
        df_clean, _, messages = validate_dc_motor_upload(df)
        assert df_clean is None
        assert any(level == "error" for level, _ in messages)


class TestPhysicallyUnreasonableRanges:
    def test_absurd_magnitude_is_dropped_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(torque_nm=1e9)])
        df_clean, _, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "reasonable ranges" in text for level, text in messages)

    def test_negative_values_are_not_rejected(self):
        """Unlike DPC's "r", no dc_motor channel is constrained to be positive -- rpm/torque/
        current/acceleration can all legitimately be negative (direction, ripple sign)."""
        df = pd.DataFrame([_valid_row(rpm=-1200.0, current_r=-5.0)])
        df_clean, _, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert messages == []

    def test_reasonable_off_distribution_values_are_not_rejected(self):
        df = pd.DataFrame([_valid_row(torque_nm=500.0)])
        assert 500.0 < DEFAULT_MAGNITUDE_BOUND
        df_clean, _, messages = validate_dc_motor_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
