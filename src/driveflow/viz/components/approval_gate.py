"""ApprovalGate (spec Sec. 9, 10 "Approval"): the human decision on a controller/default candidate.
PromotionGate stays the gate for models. Enforcement lives in the backend
(driveflow.runs.store.RunStore.record_approval refuses a blocked approval); the disabled button here
is only a convenience."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from driveflow.viz.components.state_panel import ComponentView, render_state


def approval_view(candidate_run: str | None, gate_report: dict | None, outcome: dict | None = None, loading: bool = False) -> ComponentView:
    if loading:
        return ComponentView("loading", "Checking evidence…")
    if outcome and outcome.get("error"):
        return ComponentView("error", "Approval record not saved", outcome["error"])
    if outcome and outcome.get("record"):
        rec = outcome["record"]
        return ComponentView("success", "Human approval recorded", f"{rec.decision} by {rec.approver} at {rec.user_confirmed_at} UTC")
    if not candidate_run:
        return ComponentView("empty", "No candidate selected")
    blocking = (gate_report or {}).get("blocking") or []
    if blocking:
        return ComponentView("negative", "Approval blocked. No default has been changed.", "; ".join(map(str, blocking)))
    return ComponentView("success", "Human approval required. Review safety constraints and evaluation results.")


def approval_gate(subject: str, candidate_run: str | None, gate_report: dict | None, on_submit: Callable[[str, str], dict], key: str = "approval") -> None:
    """on_submit(approver, decision) -> {"record": ApprovalRecord} or {"error": str}; the result is
    kept in st.session_state[f"{key}_outcome"] for this subject/candidate."""
    outcome_key = f"{key}_outcome"
    outcome = st.session_state.get(outcome_key)
    if outcome and outcome.get("candidate") != candidate_run:
        outcome = None
    render_state(approval_view(candidate_run, gate_report, outcome))
    if not candidate_run or (outcome and outcome.get("record")):
        return
    blocked = bool((gate_report or {}).get("blocking"))
    approver = st.text_input("Approver name", key=f"{key}_approver", help="Approvals are recorded with the name of the person who made them.")
    confirmed = st.checkbox("I reviewed the evidence and approve this change.", key=f"{key}_confirm", disabled=blocked)
    c1, c2 = st.columns(2)
    if c1.button("Record approval", key=f"{key}_approve", disabled=blocked or not confirmed or not approver.strip(), type="primary"):
        st.session_state[outcome_key] = {"candidate": candidate_run, **on_submit(approver, "approved")}
        st.rerun()
    if c2.button("Record rejection", key=f"{key}_reject", disabled=not approver.strip()):
        st.session_state[outcome_key] = {"candidate": candidate_run, **on_submit(approver, "rejected")}
        st.rerun()
