"""Validation for a user-uploaded dataset, evaluated against a trained model
(viz/dashboard.py's DPC "Evaluate your dataset" tab, Fase B) or destined for one (the dc_motor/
Fase C side, which has no trained classifier/regressor yet -- INSTRUCTIONS.md Sec. 5). Deliberately
has NO Streamlit import and does no rendering: `viz.dashboard` imports `import streamlit as st` and
runs `st.set_page_config` and other calls at module import time, which crashes (KeyError on
`_PHASES[None]`) when imported outside a real Streamlit session/runtime -- meaning dashboard.py
itself cannot be imported from a plain pytest test. Keeping the validation logic here, with no
Streamlit dependency, is what makes it directly unit-testable (tests/test_dpc_upload_validation.py)
without a browser or a simulated app session. `_render_dpc_upload_eval` in dashboard.py is a thin
wrapper that calls `validate_dpc_upload` and renders the returned messages via
st.error/st.warning/st.info.

Generalized to both domains (docs/design_ai_layer_transversal.md Sec. 5.2, Sec. 8 step 3):
`validate_dpc_upload` (vsc_dpc domain) is unchanged in behavior/messages -- still the only
validator wired into the dashboard today -- but its "columns present -> numeric dtype ->
physically-reasonable ranges" mechanics are now factored into `_coerce_numeric` and
`_drop_out_of_range_rows`, shared with the new `validate_dc_motor_upload` (dc_motor domain).
The module keeps its original filename -- the dc_motor entry point is additive, not a rename --
because dashboard.py already imports DPC_REQUIRED_COLUMNS/DPC_AUTOFILL_COLUMNS/validate_dpc_upload
from this exact path and there is no reason to disturb that.

The two domains validate differently, not just with different column lists: `validate_dpc_upload`
requires ALL 7 DPC_REQUIRED_COLUMNS (they're the fixed, hardcoded input row of one specific
trained network, control/dpc/controller.py) and treats "r" specially (must be > 0, the plant
divides by it). `validate_dc_motor_upload` requires only that AT LEAST ONE of
models.common.windowing.CANDIDATE_CHANNELS is present -- there is no trained dc_motor model yet to
fix a required input schema, and even once Fase C trains one, individual channels being
independently present-or-absent is the same tolerance discover_channels() already applies when
building training windows.
"""

import pandas as pd

from driveflow.models.common.windowing import CANDIDATE_CHANNELS as DC_MOTOR_CANDIDATE_CHANNELS

#: The true minimum to run the network at all: the measured state (i_f, v_c), R, and the
#: reference's CURRENT value. Order matches driveflow.control.dpc.COLUMNS[:7].
DPC_REQUIRED_COLUMNS = ["if_alpha", "if_beta", "vc_alpha", "vc_beta", "vref_alpha", "vref_beta", "r"]
#: The remaining 8 inputs are the reference's horizon steps 2-5 -- genuinely optional, because
#: they're derivable from vref_alpha/vref_beta plus an assumed rotation frequency (see
#: dashboard.py's _autofill_horizon_columns) if the caller doesn't have them. Order matches
#: driveflow.control.dpc.COLUMNS[7:].
DPC_AUTOFILL_COLUMNS = [
    "vref_alphaph", "vref_betaph", "vref_alphaph3", "vref_betaph3",
    "vref_alphaph4", "vref_betaph4", "vref_alphaph5", "vref_betaph5",
]

#: Generic "clearly wrong data" bound (unit errors, placeholder sentinels like -999) shared by
#: both domains -- not a physical validation, see _drop_out_of_range_rows.
DEFAULT_MAGNITUDE_BOUND = 1e4


def _coerce_numeric(df: pd.DataFrame) -> tuple:
    """Coerces every column to numeric (non-numeric -> NaN) and drops rows with any NaN.

    Returns (df, n_dropped)."""
    df = df.apply(pd.to_numeric, errors="coerce")
    n_nonnumeric = int(df.isna().any(axis=1).sum())
    return df.dropna(), n_nonnumeric


def _drop_out_of_range_rows(df: pd.DataFrame, positive_columns: tuple, magnitude_bound: float) -> tuple:
    """Drops rows where a `positive_columns` value is <= 0, or any other column's |value| exceeds
    magnitude_bound -- generous bounds meant to catch clearly wrong data (unit errors, non-positive
    resistance, placeholder sentinels like -999), not to second-guess a legitimate off-distribution
    stress test (the whole point of these evaluators).

    Returns (df, n_dropped)."""
    if df.empty:
        return df, 0
    invalid = pd.Series(False, index=df.index)
    for c in positive_columns:
        if c in df.columns:
            invalid = invalid | (df[c] <= 0)
    signal_cols = [c for c in df.columns if c not in positive_columns]
    if signal_cols:
        invalid = invalid | (df[signal_cols].abs() > magnitude_bound).any(axis=1)
    n_invalid = int(invalid.sum())
    return df[~invalid], n_invalid


def validate_dpc_upload(df_raw: pd.DataFrame):
    """Checks, in order: required columns present (schema) -> numeric dtype (coerced, non-numeric
    dropped) -> physically-reasonable ranges (R > 0, generous magnitude bound on every other
    column) -- Patch 8 Sec. 8's three explicit asks (columns present, correct types, reasonable
    ranges), each producing a controlled, informative message rather than an uncontrolled
    exception reaching the caller.

    Returns:
        (df_clean, missing_horizon_cols, messages). df_clean is None iff there's a hard error
        (missing required columns, or zero valid rows survive validation) -- callers must check
        this before using it. messages is a list of (level, text) with level in
        {"error","warning","info"}, in display order.
    """
    messages = []
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_required = [c for c in DPC_REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        messages.append((
            "error",
            f"Missing required column(s): {', '.join(missing_required)}. These {len(DPC_REQUIRED_COLUMNS)} "
            "are the minimum needed to run the network at all -- expand 'Required format' above for the full schema.",
        ))
        return None, None, messages

    present_horizon_cols = [c for c in DPC_AUTOFILL_COLUMNS if c in df.columns]
    missing_horizon_cols = [c for c in DPC_AUTOFILL_COLUMNS if c not in df.columns]

    df = df[DPC_REQUIRED_COLUMNS + present_horizon_cols]
    df, n_nonnumeric = _coerce_numeric(df)
    df, n_range_invalid = _drop_out_of_range_rows(df, positive_columns=("r",), magnitude_bound=DEFAULT_MAGNITUDE_BOUND)

    if n_nonnumeric:
        messages.append(("warning", f"Dropped {n_nonnumeric} row(s) with non-numeric or missing values."))
    if n_range_invalid:
        messages.append((
            "warning",
            f"Dropped {n_range_invalid} row(s) outside physically reasonable ranges -- R <= 0 "
            "(the plant divides by it), or any current/voltage column with |value| > 10000.",
        ))
    if df.empty:
        messages.append(("error", "No valid rows left to evaluate after validation."))
        return None, None, messages

    return df, missing_horizon_cols, messages


def validate_dc_motor_upload(df_raw: pd.DataFrame):
    """Same "columns present -> numeric dtype -> reasonable ranges" pattern as
    validate_dpc_upload, generalized to the dc_motor domain's diagnosis channels
    (models.common.windowing.CANDIDATE_CHANNELS) -- see module docstring for how the two domains'
    schemas genuinely differ, not just their column names.

    Returns:
        (df_clean, missing_channels, messages) -- same 3-tuple shape as validate_dpc_upload.
        missing_channels lists which of CANDIDATE_CHANNELS were not found in the upload
        (informational: unlike DPC's horizon columns, there is no autofill step for these).
    """
    messages = []
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    present_channels = [c for c in DC_MOTOR_CANDIDATE_CHANNELS if c in df.columns]
    if not present_channels:
        messages.append((
            "error",
            f"None of the known diagnosis channels ({', '.join(DC_MOTOR_CANDIDATE_CHANNELS)}) were found in "
            "the upload -- at least one is needed to evaluate anything.",
        ))
        return None, None, messages
    missing_channels = [c for c in DC_MOTOR_CANDIDATE_CHANNELS if c not in df.columns]

    df = df[present_channels]
    df, n_nonnumeric = _coerce_numeric(df)
    df, n_range_invalid = _drop_out_of_range_rows(df, positive_columns=(), magnitude_bound=DEFAULT_MAGNITUDE_BOUND)

    if n_nonnumeric:
        messages.append(("warning", f"Dropped {n_nonnumeric} row(s) with non-numeric or missing values."))
    if n_range_invalid:
        messages.append((
            "warning",
            f"Dropped {n_range_invalid} row(s) outside physically reasonable ranges -- a channel with "
            f"|value| > {DEFAULT_MAGNITUDE_BOUND:.0f}.",
        ))
    if df.empty:
        messages.append(("error", "No valid rows left to evaluate after validation."))
        return None, None, messages

    return df, missing_channels, messages
