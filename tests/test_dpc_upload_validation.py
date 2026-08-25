"""Tests viz/dpc_upload_validation.py::validate_dpc_upload -- the dashboard's "Evaluate your
dataset" upload validator, kept in its own Streamlit-free module specifically so it can be
unit-tested this way (see that module's docstring). Patch 8 Sec. 8: at minimum, a malformed-schema
(missing column) case must be covered, and shown to be handled in a controlled way (a returned
error message, not an exception escaping the function).
"""

import numpy as np
import pandas as pd
import pytest

from driveflow.viz.dpc_upload_validation import DPC_AUTOFILL_COLUMNS, DPC_REQUIRED_COLUMNS, validate_dpc_upload


def _valid_row(**overrides):
    row = {
        "if_alpha": 0.0, "if_beta": 0.0, "vc_alpha": 0.0, "vc_beta": 0.0,
        "vref_alpha": 50.0, "vref_beta": 0.0, "r": 8.0064,
        "vref_alphaph": 49.9, "vref_betaph": 1.6,
        "vref_alphaph3": 49.6, "vref_betaph3": 3.2,
        "vref_alphaph4": 49.1, "vref_betaph4": 4.8,
        "vref_alphaph5": 48.5, "vref_betaph5": 6.4,
    }
    row.update(overrides)
    return row


class TestMissingRequiredColumn:
    """Patch 8 Sec. 8's explicit minimum: a malformed-schema (missing column) upload must be
    handled in a controlled way -- a returned error message, not an uncontrolled exception."""

    def test_missing_required_column_returns_controlled_error_not_exception(self):
        df = pd.DataFrame([_valid_row()]).drop(columns=["r"])
        df_clean, missing_horizon, messages = validate_dpc_upload(df)  # must not raise
        assert df_clean is None
        assert missing_horizon is None
        assert any(level == "error" and "r" in text for level, text in messages)

    def test_missing_multiple_required_columns_all_named_in_the_error(self):
        df = pd.DataFrame([_valid_row()]).drop(columns=["r", "if_alpha"])
        df_clean, _, messages = validate_dpc_upload(df)
        assert df_clean is None
        error_text = next(text for level, text in messages if level == "error")
        assert "r" in error_text and "if_alpha" in error_text


class TestValidUpload:
    def test_fully_valid_upload_passes_through_with_no_messages(self):
        df = pd.DataFrame([_valid_row(), _valid_row(if_alpha=1.0)])
        df_clean, missing_horizon, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 2
        assert missing_horizon == []
        assert messages == []

    def test_column_order_and_whitespace_do_not_matter(self):
        row = _valid_row()
        shuffled_keys = list(row.keys())[::-1]
        df = pd.DataFrame([{f" {k} ": row[k] for k in shuffled_keys}])  # reversed + padded names
        df_clean, missing_horizon, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert missing_horizon == []

    def test_only_required_columns_present_autofills_the_rest(self):
        df = pd.DataFrame([_valid_row()])[DPC_REQUIRED_COLUMNS]
        df_clean, missing_horizon, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert set(missing_horizon) == set(DPC_AUTOFILL_COLUMNS)
        assert list(df_clean.columns) == DPC_REQUIRED_COLUMNS  # horizon cols not yet added at this stage


class TestNonNumericRows:
    def test_non_numeric_value_drops_that_row_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(r="not_a_number")])
        df_clean, _, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "non-numeric" in text for level, text in messages)

    def test_all_rows_non_numeric_is_a_controlled_error(self):
        df = pd.DataFrame([_valid_row(r="garbage")])
        df_clean, _, messages = validate_dpc_upload(df)
        assert df_clean is None
        assert any(level == "error" for level, _ in messages)


class TestPhysicallyUnreasonableRanges:
    def test_non_positive_resistance_is_dropped_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(r=0.0), _valid_row(r=-5.0)])
        df_clean, _, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "R <= 0" in text for level, text in messages)

    def test_absurd_magnitude_is_dropped_with_a_warning(self):
        df = pd.DataFrame([_valid_row(), _valid_row(vc_alpha=1e9)])
        df_clean, _, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
        assert any(level == "warning" and "reasonable ranges" in text for level, text in messages)

    def test_reasonable_off_distribution_values_are_not_rejected(self):
        """The range check must not be so strict it rejects legitimate off-distribution stress
        tests -- e.g. a much larger resistance/reference than training data, still physically
        sane."""
        df = pd.DataFrame([_valid_row(r=500.0, vref_alpha=200.0)])
        df_clean, _, messages = validate_dpc_upload(df)
        assert df_clean is not None
        assert len(df_clean) == 1
