"""Alert History page: the session's full ServerAgent log, filterable and exportable."""

import datetime as dt

import pandas as pd
import streamlit as st

from driveflow.viz.adapters.session import server_agent
from driveflow.viz.components.page_header import page_header, source_badge_html


def render() -> None:
    page_header(
        "Operations / Alert history",
        "Alert history",
        "Every alert event, resolution and acknowledgement recorded by this session's ServerAgent, in arrival order.",
        [source_badge_html("simulated")],
    )
    sa = server_agent()
    log = sa.alert_history()
    if not log:
        st.info("No alert recorded in this session yet — run a monitoring pass on the Monitoring page.", icon=":material/info:")
        st.page_link("pages/monitoring.py", label="Go to Monitoring", icon=":material/monitor_heart:")
        return
    df = pd.DataFrame(log)
    df["time_utc"] = [dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds") for t in df["timestamp"]]
    df["event"] = ["acknowledged" if a == "acknowledged" else ("resolved" if r else "fired") for a, r in zip(df["action"], df["resolved"])]
    c1, c2, c3 = st.columns(3)
    domains = c1.multiselect("Domain", sorted(df["domain"].unique()), key="ah_domain")
    severities = c2.multiselect("Severity", sorted(df["severity"].astype(str).unique()), key="ah_severity")
    events = c3.multiselect("Event", ["fired", "resolved", "acknowledged"], key="ah_event")
    view = df
    if domains:
        view = view[view["domain"].isin(domains)]
    if severities:
        view = view[view["severity"].astype(str).isin(severities)]
    if events:
        view = view[view["event"].isin(events)]
    cols = ["time_utc", "domain", "rule_name", "source_id", "severity", "event", "action"] + (["by"] if "by" in view else [])
    st.dataframe(view[cols], hide_index=True, width="stretch")
    st.caption(f"{len(view)} of {len(df)} event(s). Acknowledge = an operator has seen the alert; it does not resolve it.")
    st.download_button("Export CSV", view[cols].to_csv(index=False), file_name="spade_alert_history.csv", key="ah_export", icon=":material/download:")
