"""Which tools the model sees, and what each tool needs before it runs (task F1.3).

Two rules live here, both in Python and neither decided by the model:

1. **Tools by state (C2).** The model only sees the tools that make sense now, by mode
   (``undetermined``, ``configuration``, ``objective``, ``question``) and by the state of the
   session: ``run_experiments`` is visible only when the draft is ready AND a launch was proposed;
   the objective-mode tools only in objective mode; ``explain_spade`` and ``search_docs`` always.
   Changing mode is a tool (``set_mode``), never an assumption of the model.
2. **Confirmation.** ``run_experiments`` / ``run_tuning`` need a confirmation mark whose
   ``message_seq`` is the LAST user message and whose draft hash equals both the proposal's and the
   current draft's. The mark is set only by the interface layer (CLI / page) through
   ``SessionState.confirm`` -- there is no tool for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from driveflow.assistant.draft import Draft, Ledger

MODES = ("undetermined", "configuration", "objective", "question")
ALWAYS = ("set_mode", "explain_spade", "search_docs")
#: mode -> tools visible in that mode regardless of the draft's state (besides ALWAYS).
BY_MODE = {
    "undetermined": ("recommend_plant", "describe_plant"),
    "question": (),
    "configuration": ("recommend_plant", "describe_plant", "suggest_value", "update_draft", "validate_draft", "estimate_cost"),
    "objective": ("recommend_plant", "describe_plant", "suggest_value", "update_draft", "describe_tunable", "update_objective", "validate_draft"),
}
LAUNCH_TOOLS = {"configuration": "run_experiments", "objective": "run_tuning"}
PROPOSE_TOOLS = {"configuration": "propose_experiments", "objective": "propose_tuning"}
NEEDS_CONFIRMATION = ("run_experiments", "run_tuning")


@dataclass
class Proposal:
    tool: str  # the launch tool this proposal is for
    draft_hash: str
    after_message_seq: int  # the user's answer must come after this message


@dataclass
class Confirmation:
    message_seq: int
    draft_hash: str
    how: str  # "button" or "typed: <text>"


@dataclass
class SessionState:
    draft: Draft = field(default_factory=Draft)
    ledger: Ledger = field(default_factory=Ledger)
    mode: str = "undetermined"
    proposal: Proposal | None = None
    confirmation: Confirmation | None = None
    jobs: list = field(default_factory=list)  # [{"job_id", "kind", "status", "run_id"?}]
    last_user_seq: int = 0
    seq: int = 0  # every message of the conversation gets the next number

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    # --- interface layer only ----------------------------------------------------------------
    def confirm(self, message_seq: int, how: str) -> bool:
        """Set by the CLI / page when the user pressed Run or answered yes right after a proposal.
        Returns False (and sets nothing) when there is no proposal to confirm."""
        if self.proposal is None or message_seq <= self.proposal.after_message_seq or message_seq != self.last_user_seq:
            return False
        self.confirmation = Confirmation(message_seq=message_seq, draft_hash=self.proposal.draft_hash, how=how)
        return True

    def propose(self, mode: str) -> Proposal:
        """Called by the propose_* tools: freezes the draft hash the user will confirm."""
        self.proposal = Proposal(tool=LAUNCH_TOOLS[mode], draft_hash=self.draft.hash(), after_message_seq=self.seq)
        self.confirmation = None
        return self.proposal


def draft_ready(state: SessionState) -> bool:
    try:
        return bool(state.draft.status()["ready"])
    except Exception:  # noqa: BLE001 -- a broken draft is simply not ready
        return False


def visible_tools(state: SessionState) -> list[str]:
    """Names of the tools the model may see in this state (order: always, mode, state)."""
    names = list(ALWAYS) + list(BY_MODE[state.mode])
    if state.mode in LAUNCH_TOOLS:
        ready = draft_ready(state) if state.mode == "configuration" else (draft_ready(state) and bool((state.draft.objective or {}).get("complete")))
        if ready:
            names.append(PROPOSE_TOOLS[state.mode])
            p = state.proposal
            if p is not None and p.tool == LAUNCH_TOOLS[state.mode] and p.draft_hash == state.draft.hash():
                names.append(LAUNCH_TOOLS[state.mode])
    if state.jobs:
        names.append("get_job_status")
        if state.mode == "configuration" or any(j.get("kind") == "scenario_sweep" for j in state.jobs):
            names.append("summarize_results")
        if any(j.get("kind") == "controller_tuning" for j in state.jobs):
            names.append("check_requirements")
    return list(dict.fromkeys(names))


def check_call(state: SessionState, tool: str) -> str | None:
    """Why this tool may not run now (None = it may). Applies to every call the model makes."""
    if tool not in visible_tools(state):
        return f"tool {tool!r} is not available now (mode {state.mode}); available: {visible_tools(state)}"
    if tool in NEEDS_CONFIRMATION:
        c, p = state.confirmation, state.proposal
        if p is None:
            return "nothing was proposed: call the propose tool, show the plan and wait for the user's yes"
        if c is None:
            return "the user has not confirmed this proposal; wait for their answer (the interface records it)"
        if c.message_seq != state.last_user_seq:
            return "the confirmation is not in the user's latest message; ask again"
        if not (c.draft_hash == p.draft_hash == state.draft.hash()):
            return "the draft changed after the user confirmed; propose again"
    return None


#: Short answers the interface reads as "yes" right after a proposal. Anything longer or different
#: is a normal message (the model then asks again); the Run button is the unambiguous path.
AFFIRMATIVE = ("si", "sí", "s", "yes", "y", "ok", "dale", "confirmo", "adelante", "go", "/run", "sí, lánzalo", "si, lanzalo", "yes, run it")


def is_affirmative(text: str) -> bool:
    """Interface helper (CLI / page), not a tool: is this message a bare yes?"""
    return text.strip().lower().rstrip(".!") in AFFIRMATIVE
