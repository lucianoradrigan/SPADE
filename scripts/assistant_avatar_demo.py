"""Test page of the assistant's avatar (task FE-3). Not registered in the app's navigation.

    streamlit run scripts/assistant_avatar_demo.py
    # for screenshots: ?state=thinking  (any of the 11 states)  &compact=1  &no_assets=1

Shows the 11 states with buttons, and replays a recorded event sequence through the same pure
mapping the Assistant page will use (viz/adapters/assistant_adapter.avatar_state).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from driveflow.assistant import events as ev  # noqa: E402
from driveflow.viz.adapters.assistant_adapter import STATES, avatar_state  # noqa: E402
from driveflow.viz.components.radrigan_avatar import radrigan_avatar  # noqa: E402

#: a recorded configuration conversation (FE-1 test sequence): propose, confirm, run, results
RECORDED = [ev.TurnStarted(1), ev.LlmCall("start"), ev.LlmCall("end", 120), ev.ToolCallEvent("start", "describe_plant"), ev.ToolCallEvent("end", "describe_plant", True),
            ev.ToolCallEvent("start", "estimate_cost"), ev.ToolCallEvent("end", "estimate_cost", True), ev.ProposalReady("abc", 5, 32.0), ev.TurnEnded(),
            ev.TurnStarted(2), ev.UserConfirmed("abc", 2), ev.LlmCall("start"), ev.ToolCallEvent("start", "run_experiments"), ev.ToolCallEvent("end", "run_experiments", True),
            ev.JobEvent("j", "queued", 0, 5), ev.TurnEnded(), ev.JobEvent("j", "running", 3, 5), ev.JobEvent("j", "completed", 5, 5),
            ev.TurnStarted(3), ev.ToolCallEvent("start", "summarize_results"), ev.ToolCallEvent("end", "summarize_results", True), ev.ResultsReady("r"), ev.TurnEnded()]

st.set_page_config(page_title="Assistant avatar (test page)", layout="wide")
qp = st.query_params
state = qp.get("state") or st.session_state.get("demo_state", "idle")
compact = qp.get("compact") == "1"
avatar_dir = "/nonexistent-avatar-dir" if qp.get("no_assets") == "1" else ""

left, right = st.columns([2, 3])
with left:
    slot = st.empty()
with right:
    st.caption("Test page of the avatar (FE-3). Not part of the app's navigation.")
    cols = st.columns(4)
    for i, s in enumerate(STATES):
        if cols[i % 4].button(s, key=f"b_{s}"):
            st.session_state["demo_state"] = state = s
    replay = st.button("Replay a recorded conversation")

labels = {"running": "Simulation running: 3/5", "using_tool": "Checking the plant…", "error": "describe_plant: unknown plant 'x'"}
with slot:
    res = radrigan_avatar(state, label=labels.get(state), progress=0.6 if state == "running" else None, compact=compact, avatar_dir=avatar_dir, report_env=True)
env = getattr(res, "env", None) if res is not None else None
if env:
    st.caption(f"component reports: reduced_motion={env.get('reduced_motion')} state={env.get('state')} has_assets={env.get('has_assets')}")
if replay:
    for i in range(len(RECORDED)):
        s, label, progress = avatar_state(RECORDED[: i + 1])
        with slot:
            radrigan_avatar(s, label=label, progress=progress, compact=compact, avatar_dir=avatar_dir, beat=i)
        time.sleep(0.4)
