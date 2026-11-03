"""Assistant events -> avatar state (task FE-1, ESPEC_FRONTEND_PHD_RADRIGAN §1.3 and §2).

``avatar_state(events)`` is a pure function: it replays a list of ``AgentEvent`` dicts (or objects)
and returns ``(state, label, progress)`` for the radrigan_avatar component. When several states are
active at the end of the list, the highest priority wins:
unavailable > error > confirmed > extracting > using_tool > thinking > running > awaiting_confirmation
> results_ready > idle. The 400 ms minimum of ``thinking`` is the UI's job, not this function's.

FE-D6: only ``user_confirmed`` (the user's confirmation recorded by the interface) leads to
``confirmed``; ``results_ready`` never does, and it carries no approval gesture.
"""

from __future__ import annotations

STATES = ("idle", "listening", "thinking", "using_tool", "extracting", "awaiting_confirmation", "confirmed", "running", "results_ready", "error", "unavailable")
PRIORITY = ("unavailable", "error", "confirmed", "extracting", "using_tool", "thinking", "running", "awaiting_confirmation", "results_ready", "idle")
TERMINAL_JOB_STATES = ("completed", "partially_completed", "failed", "cancelled")

#: Readable chip text per tool (English, like the rest of the GUI). Every registered tool needs one
#: (tests/assistant/test_assistant_events.py).
TOOL_LABELS = {
    "set_mode": "Understanding the request…",
    "recommend_plant": "Choosing the plant…",
    "describe_plant": "Checking the plant…",
    "suggest_value": "Looking up a suggested value…",
    "update_draft": "Updating the draft…",
    "validate_draft": "Validating the experiments…",
    "estimate_cost": "Estimating cost…",
    "propose_experiments": "Preparing the proposal…",
    "run_experiments": "Launching the simulations…",
    "get_job_status": "Checking the job…",
    "summarize_results": "Summarizing the results…",
    "describe_tunable": "Checking what can be tuned…",
    "update_objective": "Writing the requirements…",
    "propose_tuning": "Preparing the tuning proposal…",
    "run_tuning": "Launching the tuning…",
    "check_requirements": "Checking the requirements…",
    "explain_spade": "Looking it up in SPADE's catalog…",
    "search_docs": "Searching SPADE's documentation…",
}
LABELS = {"thinking": "Thinking…", "awaiting_confirmation": "Waiting for your confirmation", "confirmed": "Launched", "results_ready": "Results ready (simulated)",
          "unavailable": "Assistant unavailable: local model not running"}


def _d(e) -> dict:
    return e if isinstance(e, dict) else e.to_dict()


def tool_label(tool: str) -> str:
    return TOOL_LABELS.get(tool, f"Running {tool}…")


def avatar_state(events: list) -> tuple[str, str | None, float | None]:
    active: dict[str, bool] = {s: False for s in PRIORITY}
    tool_stack: list[str] = []
    error_msg = unavailable_msg = None
    job = None
    for raw in events:
        e = _d(raw)
        t = e.get("type")
        if t == "turn_started":
            active["error"] = active["confirmed"] = active["results_ready"] = False
            error_msg = None
        elif t == "llm_call":
            active["thinking"] = e.get("phase") == "start"
        elif t == "tool_call":
            if e.get("phase") == "start":
                tool_stack.append(e.get("tool", ""))
            elif tool_stack:
                tool_stack.pop()
        elif t == "extraction":
            active["extracting"] = e.get("phase") != "end"
        elif t == "proposal_ready":
            active["awaiting_confirmation"] = True
            active["results_ready"] = False
        elif t == "user_confirmed":
            active["confirmed"] = True
            active["awaiting_confirmation"] = False
        elif t == "job":
            job = e
            active["running"] = e.get("state") not in TERMINAL_JOB_STATES
        elif t == "results_ready":
            active["results_ready"] = True
            active["running"] = False
        elif t == "error":
            active["error"] = True
            error_msg = (e.get("message") or e.get("where") or "error").splitlines()[0][:160]
        elif t == "runtime_unavailable":
            active["unavailable"] = True
            unavailable_msg = e.get("reason")
        elif t == "turn_ended":
            # "confirmed" is a one-shot (900 ms in the UI); after the turn the job shows as running.
            active["thinking"] = active["confirmed"] = False
            tool_stack.clear()
            if e.get("stopped") not in (None, "runtime_unavailable"):
                active["error"] = True
                error_msg = error_msg or f"stopped: {e.get('stopped')}"
    active["using_tool"] = bool(tool_stack)
    active["idle"] = True
    state = next(s for s in PRIORITY if active[s])
    progress = None
    label = LABELS.get(state)
    if state == "using_tool":
        label = tool_label(tool_stack[-1])
    elif state == "error":
        label = error_msg
    elif state == "unavailable":
        label = LABELS["unavailable"] + (f" ({unavailable_msg})" if unavailable_msg else "")
    elif state == "running" and job:
        done, total = job.get("done"), job.get("total")
        label = f"Simulation running: {done}/{total}" if done is not None and total else "Simulation running"
        progress = (done / total) if done is not None and total else None
    elif state == "idle":
        label = None
    return state, label, progress
