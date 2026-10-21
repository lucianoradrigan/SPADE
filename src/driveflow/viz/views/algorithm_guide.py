"""Help / Algorithm Guide (spec Sec. 8.6; expansion task E12). Three questions -> an algorithm family
with why / assumptions / limitations -> the SPADE page that implements it, or "Not available in this
version". The table is configs/algorithm_guide.yaml; this view only renders it."""

from __future__ import annotations

import streamlit as st

from driveflow.viz.adapters import guide_adapter as ga
from driveflow.viz.components.page_header import badge_html, page_header


def render() -> None:
    page_header("Help / Algorithm Guide", "Which algorithm family fits my problem?",
                "Answer three questions to get an algorithm family, why it fits, what it assumes, where it stops working, and the SPADE page that implements it.",
                [badge_html("GUIDANCE", "training", "◆", "A starting point, not a guarantee that the method suits your system")])
    guide = ga.load_guide()
    ctrl, view = st.columns([1, 2], gap="medium")
    with ctrl:
        with st.container(border=True):
            phase = st.radio("1 · Life-cycle phase", list(guide["phases"]), format_func=guide["phases"].get, key="guide_phase")
            tasks = ga.tasks_for(phase, guide)
            labels = dict(tasks)
            task = st.radio("2 · Task", [t for t, _ in tasks], format_func=labels.get, key=f"guide_task_{phase}")
            source = st.radio("3 · Data source", list(guide["data_sources"]), format_func=guide["data_sources"].get, key="guide_source")
    rec = ga.recommend(phase, task, source, guide)
    with view:
        with st.container(border=True):
            st.caption("Recommended family")
            st.markdown(f"#### {rec.family}")
            st.markdown(f"**Why.** {rec.why}")
            st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in rec.assumptions))
            st.markdown("**Limitations**\n" + "\n".join(f"- {x}" for x in rec.limitations))
            if rec.page:
                st.page_link(rec.page, label=f"Open {rec.page_title}", icon=":material/arrow_forward:")
            else:
                st.info(f"**{rec.unavailable_reason}.** SPADE does not implement this for the chosen inputs yet.", icon=":material/block:")
        st.caption(f"Structure (what → which → how) after the companion selector of: {ga.citation_text(guide)} "
                   "The recommendations are SPADE's own, written for what SPADE implements; none of the selector's code or text is copied.")
