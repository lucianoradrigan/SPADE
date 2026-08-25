"""Validation for a user-uploaded dataset evaluated against the trained DPC network
(viz/dashboard.py's "Evaluate your dataset" tab, Fase B). Deliberately has NO Streamlit import and
does no rendering: `viz.dashboard` imports `import streamlit as st` and runs `st.set_page_config`
and other calls at module import time, which crashes (KeyError on `_PHASES[None]`) when imported
outside a real Streamlit session/runtime -- meaning dashboard.py itself cannot be imported from a
plain pytest test. Keeping the validation logic here, with no Streamlit dependency, is what makes
it directly unit-testable (tests/test_dpc_upload_validation.py) without a browser or a simulated
app session. `_render_dpc_upload_eval` in dashboard.py is a thin wrapper that calls
`validate_dpc_upload` and renders the returned messages via st.error/st.warning/st.info.
"""

import pandas as pd

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
    df = df.apply(pd.to_numeric, errors="coerce")
    n_nonnumeric = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    # Physically-reasonable range checks -- generous bounds meant to catch clearly wrong data
    # (unit errors, non-positive resistance, placeholder sentinels like -999), not to second-guess
    # a legitimate off-distribution stress test (the whole point of this evaluator).
    if not df.empty:
        r_invalid = df["r"] <= 0  # VscSystem.simulate() divides by R (i_load = v_c/R)
        signal_cols = [c for c in df.columns if c != "r"]
        magnitude_invalid = (df[signal_cols].abs() > 1e4).any(axis=1)
        n_range_invalid = int((r_invalid | magnitude_invalid).sum())
        df = df[~(r_invalid | magnitude_invalid)]
    else:
        n_range_invalid = 0

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
