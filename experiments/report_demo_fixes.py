"""Demo fixes (Luciano, 2026-11-08): the two milestone conversations again, on the Assistant page with the
pinned local model, plus a question about the results WHILE the job is still queued (the case that showed
``error`` in the milestone demo).

    python experiments/report_demo_fixes.py
Writes docs/audit/assistant/demo_fixes_live.json. Descriptive evidence (not F1.11). What is checked:
1. every text the user read after the number filter is made of whole sentences / lines of the model's
   text, and no line ends in ":" without content (truncated text);
2. no model call offered summarize_results / check_requirements before the job ended, and no ``error``
   event or avatar state while it ran (results too early);
3. after the worker button the chat shows the job's message and results card without a new question
   (text not refreshed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import report_f1_10_demo as demo  # noqa: E402

OUT = ROOT / "docs" / "audit" / "assistant" / "demo_fixes_live.json"
EARLY = {"A_configuration": "¿Ya están los resultados?", "B_objective": "¿Ya se cumplen los requisitos?"}
RESULT_TOOLS = ("summarize_results", "check_requirements")


def _checks(agent) -> dict:
    from driveflow.assistant import answer

    ev = agent.trace.events
    post = [e for e in ev if e["kind"] == "postprocess" and e.get("what") == "strip_draft_numbers"]
    whole, dangling = [], []
    for e in post:
        raw_units = {u.strip() for u, _ in answer.units(e["raw"])}
        whole.append(e["shown"] in answer.SEE_PANEL.values() or all(u.strip() in raw_units for u, _ in answer.units(e["shown"])))
    for e in post:  # headers left without content by the filter (the model's own unfiltered text is not checked)
        us = [u for u, _ in answer.units(e["shown"]) if u.strip()]
        for i, u in enumerate(us):
            if answer._HEADER.search(u) and (i == len(us) - 1 or answer._indent(us[i + 1]) is None):
                dangling.append(u)
    first_done = next((i for i, e in enumerate(ev) if e["kind"] == "job_update"), len(ev))
    launched = next((i for i, e in enumerate(ev) if e["kind"] == "tool_result" and e["name"] in ("run_experiments", "run_tuning")), None)
    early_offers = [i for i, e in enumerate(ev) if e["kind"] == "model" and launched is not None and launched < i < first_done
                    and any(t in (e.get("visible_tools") or []) for t in RESULT_TOOLS)]
    refused = [e["reason"] for e in ev if e["kind"] == "tool_refused" and e["name"] in RESULT_TOOLS]
    errors = [e for e in agent.trace.agent_events if e["type"] == "error"]
    updates = [e for e in ev if e["kind"] == "job_update"]
    return {"error_detail": [e for e in agent.trace.agent_events if e["type"] in ("error", "turn_ended") and (e["type"] == "error" or e.get("stopped"))],
            "filtered_texts": len(post), "filtered_texts_made_of_whole_units": sum(whole), "dangling_headers": dangling,
            "result_tools_offered_while_running": len(early_offers), "result_tools_refused_while_running": refused, "error_events": len(errors),
            "job_updates": [{"state": u["state"], "text": u["text"], "cards": u["cards"]} for u in updates]}


def main() -> None:
    from driveflow.assistant.config import load_config

    cfg = load_config()
    res = {"command": "python experiments/report_demo_fixes.py", "model": cfg["model"], "think": cfg.get("think"), "conversations": {}}
    for name, steps in demo.CONVERSATIONS.items():
        steps = list(steps)
        steps.insert(steps.index("WORKER"), EARLY[name])
        captured = {}
        orig = demo._snapshot

        def snap(at, _orig=orig, _c=captured):
            _c["agent"] = at.session_state["assistant_agent"]
            _c["states"] = list(at.session_state["assistant_events"])
            return _orig(at)

        demo._snapshot = snap
        try:
            conv = demo.run_conversation(name, steps)
        finally:
            demo._snapshot = orig
        conv["checks"] = _checks(captured["agent"])
        conv["raw_and_shown"] = [{k: e.get(k) for k in ("what", "raw", "shown", "removed")} for e in captured["agent"].trace.events if e["kind"] == "postprocess"]
        res["conversations"][name] = conv
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: {"states": v["avatar_states"], "checks": {c: x for c, x in v["checks"].items() if c != "job_updates"}, "total_s": round(v["total_s"])}
                      for k, v in res["conversations"].items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
