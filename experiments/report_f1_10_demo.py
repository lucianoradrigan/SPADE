"""F1.10 + FE-4, milestone "Demo interna del asistente": the Assistant page driven end to end with the
pinned local model (qwen3.5:9b, Ollama), through Streamlit's AppTest (the real page script, no browser).

Two conversations, each in a fresh session and a temporary workspace:
  A. configuration: request -> (questions) -> proposal -> Run button -> job -> results;
  B. objective: requirements -> proposal -> Run button -> tuning job -> requirements check.
Recorded per turn: the user's input, the assistant's sentence, the cards' titles and sources, the draft
panel, the confirmation set by the page, the wall time, and the avatar states the events went through.

    python experiments/report_f1_10_demo.py
Writes docs/audit/assistant/f1_10_demo.{md,json}. Descriptive evidence for the milestone (not F1.11).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "assistant" / "f1_10_demo"
SCRIPT = "import driveflow.viz.views.assistant as v\nv.STREAM_DELAY_S = 0\nv.render()\n"
CONVERSATIONS = {
    "A_configuration": ["Simula el motor DC de stock con un controlador PI a 100 rad/s durante 0,5 s.", "RUN", "WORKER", "¿Cuáles son los resultados?"],
    "B_objective": ["Sintoniza el PI del motor DC de stock para que el sobrepaso sea menor a 5 % y el tiempo de establecimiento menor a 0,2 s.", "RUN", "WORKER",
                    "¿Se cumplen los requisitos?"],
}
MAX_EXTRA_TURNS = 2  # if the assistant asks something before proposing, answer with the agreed fallback
FALLBACK = {"A_configuration": "Usa el motor dc_perm_ex_v1 con los valores que te di.", "B_objective": "Usa el motor dc_perm_ex_v1 y el algoritmo por defecto."}


def _states(events: list) -> list:
    from driveflow.viz.adapters.assistant_adapter import avatar_state

    seq = []
    for i in range(len(events)):
        s = avatar_state(events[: i + 1])[0]
        if not seq or seq[-1] != s:
            seq.append(s)
    return seq


def _snapshot(at) -> dict:
    agent = at.session_state["assistant_agent"]
    hist = at.session_state["assistant_history"]
    last = next((m for m in reversed(hist) if m["role"] == "assistant"), None)
    from driveflow.assistant.answer import draft_panel

    return {"assistant": last["text"] if last else None, "cards": [{"title": c["title"], "source": c["source"]} for c in (last or {}).get("cards", [])],
            "draft": [{k: r[k] for k in ("field", "value", "unit", "origin")} for r in draft_panel(agent.state.draft)],
            "proposal_pending": agent.state.proposal is not None, "jobs": [dict(j) for j in agent.state.jobs],
            "exceptions": [e.message for e in at.exception]}


def run_conversation(name: str, steps: list) -> dict:
    from streamlit.testing.v1 import AppTest

    os.environ["SPADE_WORKSPACE"] = tempfile.mkdtemp(prefix=f"spade_demo_{name}_")
    at = AppTest.from_string(SCRIPT, default_timeout=900)
    t0 = time.time()
    at.run()
    turns = []
    extra = 0
    queue = list(steps)
    while queue:
        step = queue.pop(0)
        ts = time.time()
        if step == "RUN":
            btn = [b for b in at.button if b.label == "Run"]
            if not btn or btn[0].disabled:
                if extra < MAX_EXTRA_TURNS:  # nothing proposed yet: the assistant asked something
                    extra += 1
                    queue.insert(0, "RUN")
                    step = FALLBACK[name]
                    at.chat_input[0].set_value(step).run()
                else:
                    turns.append({"input": "RUN", "skipped": "no proposal to confirm", "wall_s": 0.0})
                    continue
            else:
                btn[0].click().run()
        elif step == "WORKER":
            btn = [b for b in at.button if b.label.startswith("Process queued jobs")]
            if btn:
                btn[0].click().run()
            else:
                turns.append({"input": step, "skipped": "no job", "wall_s": 0.0})
                continue
        else:
            at.chat_input[0].set_value(step).run()
        snap = _snapshot(at)
        snap.update({"input": step, "wall_s": time.time() - ts})
        turns.append(snap)
        print(name, step[:50], f"{snap['wall_s']:.1f}s", (snap["assistant"] or "")[:120].replace("\n", " "), flush=True)
    agent = at.session_state["assistant_agent"]
    events = agent.trace.agent_events
    conf = [e for e in agent.trace.events if e["kind"] == "confirmation"]
    return {"turns": turns, "total_s": time.time() - t0, "avatar_states": _states(events), "confirmations": conf,
            "launch_without_confirmation": sum(1 for e in agent.trace.events if e["kind"] == "tool_result" and e["name"] in ("run_experiments", "run_tuning")) - sum(c["recorded"] for c in conf),
            "notes": [e["key"] for e in agent.trace.events if e["kind"] == "note"]}


def main() -> None:
    from driveflow.assistant.config import load_config

    cfg = load_config()
    res = {"command": "python experiments/report_f1_10_demo.py", "model": cfg["model"], "think": cfg.get("think"), "guards": cfg.get("guards"), "conversations": {}}
    for name, steps in CONVERSATIONS.items():
        res["conversations"][name] = run_conversation(name, steps)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: {"states": v["avatar_states"], "total_s": round(v["total_s"])} for k, v in res["conversations"].items()}, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
