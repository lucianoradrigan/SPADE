"""Promotion readiness panel: the fail-closed gate decision (ai/promotion.py) rendered as one
status badge, the blocking reasons and every gate item -- never a bare "ready" badge."""

import streamlit as st

from driveflow.viz.components.metric_cards import check_list
from driveflow.viz.components.page_header import status_badge_html


def promotion_gate(decision) -> None:
    """decision: ai.promotion.GateDecision (approved, blocking, items)."""
    st.markdown("##### Promotion readiness")
    st.html('<div class="spade-root spade-badges">' + status_badge_html("approved" if decision.approved else "blocked") + "</div>")
    if not decision.approved:
        st.error("Cannot promote yet — " + "; ".join(f"{i.name}: {i.status}" for i in decision.blocking), icon=":material/block:")
    check_list([(i.name.replace("validation:", "validation · ").replace("_", " "), i.status.lower(), i.detail) for i in decision.items])
