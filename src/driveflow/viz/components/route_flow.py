"""Route diagram: one lane per way of using a tool, each a left-to-right chain of steps. Plain HTML
and CSS (st.html strips <svg>); the steps stack on a narrow screen."""

from html import escape

import streamlit as st


def route_flow(lanes: list) -> None:
    """lanes: [{"title", "needs", "tone" (hex accent), "steps": [(step, detail)]}]. The last step is
    the lane's result and is drawn as such."""
    rows = []
    for lane in lanes:
        steps = []
        for i, (step, detail) in enumerate(lane["steps"]):
            if i:
                steps.append('<div class="spade-route-arrow" aria-hidden="true">→</div>')
            result = " is-result" if i == len(lane["steps"]) - 1 else ""
            steps.append(f'<div class="spade-route-step{result}"><b>{escape(step)}</b><span>{escape(detail)}</span></div>')
        rows.append(f'<div class="spade-route" style="--tone:{lane["tone"]}"><div class="spade-route-head"><b>{escape(lane["title"])}</b><span>{escape(lane["needs"])}</span></div>'
                    f'<div class="spade-route-steps">{"".join(steps)}</div></div>')
    st.html(f'<div class="spade-root spade-routes">{"".join(rows)}</div>')
