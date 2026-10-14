"""Alert log table shared by the Monitoring and Alert History pages: one formatting of the
ServerAgent log (UTC time, fired/resolved/acknowledged event), so both pages show the same columns."""

import datetime as dt

import pandas as pd
import streamlit as st

COLUMNS = ["time_utc", "domain", "rule_name", "source_id", "severity", "event", "action"]


def alert_log_frame(log: list) -> pd.DataFrame:
    """log: ServerAgent.alert_history() entries. Adds time_utc and the derived event column."""
    df = pd.DataFrame(log)
    df["time_utc"] = [dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds") for t in df["timestamp"]]
    df["event"] = ["acknowledged" if a == "acknowledged" else ("resolved" if r else "fired") for a, r in zip(df["action"], df["resolved"])]
    return df


def alert_columns(df: pd.DataFrame) -> list:
    return COLUMNS + (["by"] if "by" in df else [])


def alerts_table(df: pd.DataFrame) -> None:
    st.dataframe(df[alert_columns(df)], hide_index=True, width="stretch")
