"""Assistant page (tasks F1.10 + FE-4): chat with SPADE's local assistant.

Layout (ESPEC_FRONTEND, ADENDA FE-4): the avatar column, the chat column and the draft panel with
**Run**. On narrow screens the columns stack and a compact avatar shows in the header (CSS).

- The loop is called with ``on_event``; every event repaints the avatar in an ``st.empty()``:
  ``thinking`` is painted before the model call.
- The answer is shown word by word and the avatar's ``beat`` advances every few words (the model's
  reply arrives whole from Ollama; the streaming is visual only).
- Job progress refreshes in an ``st.fragment(run_every="2s")``. When a job changes state (the
  worker button, or a worker elsewhere), ``Agent.job_updates`` updates the avatar and, for a job that
  ended, adds the assistant's fixed message with the results card to the chat (demo fixes, 2026-11-08).
- What the user reads: the assistant's sentence, the CARDS (SPADE data verbatim with its source) and
  the draft panel (each value with its origin chip).
- The confirmation is set by THIS layer only: the Run button, or a yes typed right after a proposal.
- Ollama down or no model: state ``unavailable`` with how to start it; the page keeps working.

Fixed texts: the first message, and "All results are SIMULATED." in the footer.
"""

from __future__ import annotations

import time

import streamlit as st

from driveflow.assistant import answer
from driveflow.assistant.cli import FIRST_MESSAGE, build_agent, unavailable_message
from driveflow.assistant.core import llm
from driveflow.assistant.core.policy import is_affirmative
from driveflow.viz.adapters.assistant_adapter import avatar_state
from driveflow.viz.components.radrigan_avatar import radrigan_avatar

FOOTER = "All results are SIMULATED."
WORDS_PER_BEAT = 4
STREAM_DELAY_S = 0.02
#: FE polish (2026-11-12, ESPEC §1.3): "thinking" stays on screen at least this long, so a fast reply does not flicker.
MIN_THINKING_S = 0.4
#: FE polish (2026-11-12): the draft panel's origin chips get a colour; the text stays, so colour is never the only signal.
#: Light background + dark text (contrast > 7:1 in both themes, the chip carries its own background).
ORIGIN_STYLE = {"USER": ("#dbeafe", "#1e3a8a"), "DERIVED": ("#ede9fe", "#4c1d95"), "SUGGESTED": ("#fef3c7", "#78350f"),
                "DEFAULT": ("#e5e7eb", "#111827"), "ASSUMED": ("#fee2e2", "#7f1d1d"), "DATASHEET": ("#dcfce7", "#14532d")}
ORIGIN_LEGEND = ("Origin: USER = you gave it · DERIVED = computed from your values · SUGGESTED = proposed by the assistant · "
                 "DEFAULT = SPADE's default · ASSUMED = an assumption to verify · DATASHEET = read from a document")
#: Tests replace this with a factory returning a ReplayClient; None = the configured OllamaClient.
CLIENT_FACTORY = None
NARROW_CSS = """
<style>
.st-key-assistant_header_avatar {display: none;}
@media (max-width: 900px) {
  .st-key-assistant_header_avatar {display: block;}
  .st-key-assistant_side_avatar {display: none;}
}
</style>
"""


def _config() -> dict:
    from driveflow.assistant.config import load_config

    return load_config()


def _session():
    ss = st.session_state
    if "assistant_agent" not in ss:
        ss.assistant_events = []
        ss.assistant_history = []  # [{"role", "text", "cards"}]
        ss.assistant_unavailable = None
        client = CLIENT_FACTORY() if CLIENT_FACTORY else None
        ss.assistant_agent = build_agent(client=client, config=_config(), on_event=ss.assistant_events.append)
        if isinstance(ss.assistant_agent.client, llm.OllamaClient):
            try:
                ss.assistant_agent.client.check()
            except (llm.RuntimeUnavailable, llm.ModelNotAvailable) as exc:
                ss.assistant_unavailable = unavailable_message(exc, _config())
    return ss.assistant_agent


def _avatar(placeholder, compact=False, beat=0, speaking=False, key="assistant_avatar"):
    ss = st.session_state
    if ss.get("assistant_unavailable"):
        state, label, progress = "unavailable", "Assistant unavailable", None
    else:
        state, label, progress = avatar_state(ss.get("assistant_events", []))
    # A placeholder repainted during one run needs a new element key each time.
    n = ss.get("_assistant_avatar_paints", 0) + 1
    ss["_assistant_avatar_paints"] = n
    with placeholder.container():
        radrigan_avatar(state, label=label, progress=progress, compact=compact, beat=beat, speaking=speaking, key=f"{key}_{n}")


def _render_cards(cards: list) -> None:
    for c in cards:
        with st.container(border=True):
            st.markdown(f"**{c['title']}**")
            st.caption(f"Source: {c['source']}")
            for ln in c.get("lines", []):
                if ln:
                    st.text(ln)  # verbatim: never interpreted as markdown


def _answered_in(wall_s: float) -> str:
    """FE polish (2026-11-12): the visible response time of a turn (model and tools, not the visual streaming)."""
    return f"Answered in {wall_s:.1f} s · local model"


def _render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m["text"]:
            st.markdown(m["text"])
        if m.get("cards"):
            _render_cards(m["cards"])
        if m.get("wall_s") is not None:
            st.caption(_answered_in(m["wall_s"]))


def _turn(agent, text: str, how: str | None, avatar_slot) -> None:
    ss = st.session_state
    ss.assistant_history.append({"role": "user", "text": text, "cards": []})
    _render_message(ss.assistant_history[-1])
    agent.on_event = lambda e: (ss.assistant_events.append(e), _avatar(avatar_slot))
    t0 = time.monotonic()
    try:
        res = agent.user_turn(text, confirmed_how=how)
        wall_s = time.monotonic() - t0
        if wall_s < MIN_THINKING_S:  # keep "thinking" visible at least MIN_THINKING_S (no flicker on fast replies)
            time.sleep(MIN_THINKING_S - wall_s)
    except (llm.RuntimeUnavailable, llm.ModelNotAvailable) as exc:
        ss.assistant_unavailable = unavailable_message(exc, _config())
        ss.assistant_history.append({"role": "assistant", "text": ss.assistant_unavailable, "cards": []})
        _avatar(avatar_slot)
        return
    finally:
        agent.on_event = ss.assistant_events.append
    ss.assistant_unavailable = None
    with st.chat_message("assistant"):
        box = st.empty()
        words, shown = res.text.split(), []
        for i, w in enumerate(words):
            shown.append(w)
            box.markdown(" ".join(shown))
            if i % WORDS_PER_BEAT == 0:
                _avatar(avatar_slot, beat=i // WORDS_PER_BEAT + 1, speaking=True)
            time.sleep(STREAM_DELAY_S)
        if res.cards:
            _render_cards(res.cards)
        st.caption(_answered_in(wall_s))
    ss.assistant_history.append({"role": "assistant", "text": res.text, "cards": res.cards, "wall_s": wall_s})
    _avatar(avatar_slot)


def _apply_job_updates(agent) -> bool:
    """Read the jobs; add the assistant's message for every job that ended. True if anything changed
    (the avatar or the chat must be repainted)."""
    ss = st.session_state
    before = len(ss.get("assistant_events", []))
    for u in agent.job_updates():
        ss.assistant_history.append({"role": "assistant", "text": u["text"], "cards": u["cards"]})
    return len(ss.get("assistant_events", [])) != before


@st.fragment(run_every="2s")
def _jobs_panel() -> None:
    agent = st.session_state.get("assistant_agent")
    if agent is None or not agent.state.jobs:
        return
    if _apply_job_updates(agent):
        st.rerun()
    from driveflow.jobs.service import JobService

    svc = JobService()
    for j in agent.state.jobs:
        try:
            job = svc.get(j["job_id"])
        except Exception as exc:  # noqa: BLE001 -- the panel must not break the page
            st.caption(f"{j['job_id']}: {exc}")
            continue
        total = job.progress_total or 1
        st.progress(min(1.0, (job.progress_done or 0) / total), text=f"{j['job_id']} · {job.state.value} · {job.progress_done or 0}/{job.progress_total or '?'}")
    try:  # a link only exists inside the app's navigation; the view rendered alone (tests, demo) gets a caption
        st.page_link("pages/experiment_runs.py", label="Open Experiment Runs", icon=":material/open_in_new:")
    except Exception:  # noqa: BLE001 -- StreamlitAPIException / KeyError when the page is not registered
        st.caption("The runs are listed on the Experiment Runs page.")


def _origin_css(origin: str) -> str:
    """CSS for one origin cell; several origins ("DEFAULT/USER") take the first one's colour."""
    bg, fg = ORIGIN_STYLE.get(str(origin).split("/")[0], ("", ""))
    return f"background-color: {bg}; color: {fg}; font-weight: 600" if bg else ""


def _draft_panel(agent) -> None:
    st.markdown("**Draft**")
    rows = answer.draft_panel(agent.state.draft)
    if rows:
        import pandas as pd

        df = pd.DataFrame([{"Field": r["field"], "Value": str(r["value"]) + (f" {r['unit']}" if r.get("unit") else ""), "Origin": r["origin"]} for r in rows])
        st.dataframe(df.style.map(_origin_css, subset=["Origin"]), hide_index=True, use_container_width=True)
        st.caption(ORIGIN_LEGEND)
    else:
        st.caption("Empty: describe what you want to simulate.")
    p = agent.state.proposal
    pending = p is not None and p.draft_hash == agent.state.draft.hash() and agent.state.confirmation is None
    if st.button("Run", type="primary", disabled=not pending, use_container_width=True, help="Confirm the last proposal (the assistant never runs anything without it)"):
        st.session_state.assistant_pending = ("/run", "button")
        st.rerun()
    _jobs_panel()
    if agent.state.jobs and st.button("Process queued jobs here (once)", use_container_width=True,
                                      help="Runs the queued jobs in this process; otherwise start a worker: python -m driveflow.jobs.worker"):
        from driveflow.jobs.worker import Worker
        from driveflow.runs.store import workspace_root

        Worker(workspace_root()).serve(once=True, poll_s=0.05)
        _apply_job_updates(agent)
        st.rerun()


def render() -> None:
    st.markdown(NARROW_CSS, unsafe_allow_html=True)
    agent = _session()
    ss = st.session_state
    cfg = _config()
    status = (cfg.get("model") or {}).get("status")
    st.title("Assistant")
    model = cfg.get("model") or {}
    st.caption(f"Local model `{model.get('tag')}` via Ollama" + (f" · status: {status}" if status else ""),
               help=f"Model {model.get('tag')} · digest {model.get('digest') or 'not pinned'} · runs on this machine; no data leave it.")
    with st.container(key="assistant_header_avatar"):
        _avatar(st.empty(), compact=True, key="assistant_avatar_compact")
    col_avatar, col_chat, col_draft = st.columns([1.1, 2.6, 1.5])
    with col_avatar:
        with st.container(key="assistant_side_avatar"):
            avatar_slot = st.empty()
            _avatar(avatar_slot)
    with col_chat:
        _render_message({"role": "assistant", "text": FIRST_MESSAGE, "cards": []})
        if ss.get("assistant_unavailable"):
            st.warning(ss.assistant_unavailable)
        for m in ss.assistant_history:
            _render_message(m)
        prompt = st.chat_input("Describe what you want to simulate, tune or know about SPADE")
        pending = ss.pop("assistant_pending", None)
        if pending:
            _turn(agent, pending[0], pending[1], avatar_slot)
        elif prompt:
            how = f"typed: {prompt}" if agent.state.proposal is not None and is_affirmative(prompt) else None
            _turn(agent, prompt, how, avatar_slot)
    with col_draft:  # filled after the turn so it shows the draft as the turn left it
        _draft_panel(agent)
    st.divider()
    st.caption(FOOTER)
