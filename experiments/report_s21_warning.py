"""S21-3: record a conversation with the pinned local model in which the assistant proposes runs on
dc_pmg132_v1 at light load, and check that the proposal card carries the S-21 warning.

    python experiments/report_s21_warning.py
Writes tests/assistant/cassettes/s21_warning_pmg132.json (replayed strictly by
tests/assistant/test_assistant_s21_warning.py) and docs/audit/assistant/s21_3_warning.json.
The agent is built exactly as the CLI and the Assistant page build it (cli.build_agent).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CASSETTE = ROOT / "tests" / "assistant" / "cassettes" / "s21_warning_pmg132.json"
OUT = ROOT / "docs" / "audit" / "assistant" / "s21_3_warning.json"
USER_TURNS = ["Simula el motor dc_pmg132_v1 con un controlador PI a 120 rad/s durante 0,5 s, sin carga."]
FALLBACK = "Sí, usa esos valores y propónmelo."  # if the assistant asks before proposing (at most twice)


def main() -> None:
    from driveflow.assistant.cli import build_agent
    from driveflow.assistant.config import load_config
    from driveflow.assistant.core.llm import OllamaClient, RecordingClient
    from driveflow.assistant.draft import audit_trace

    os.environ["SPADE_WORKSPACE"] = tempfile.mkdtemp(prefix="spade_s21_3_")
    cfg = load_config()
    inner = OllamaClient(cfg)
    info = inner.check()
    rec = RecordingClient(inner, model_id=inner.model_id(), runtime_version=info["runtime_version"])
    a = build_agent(client=rec, config=cfg)
    turns, cards = list(USER_TURNS), []
    transcript = []
    while turns:
        text = turns.pop(0)
        res = a.user_turn(text)
        cards += res.cards
        transcript.append({"user": text, "assistant": res.text, "cards": res.cards, "stopped": res.stopped})
        if a.state.proposal is None and len(transcript) < 3:
            turns.append(FALLBACK)
    proposal = [c for c in cards if c["title"].startswith("Proposal:")]
    warned = bool(proposal) and any(ln.startswith("WARNING S-21") for ln in proposal[-1]["lines"])
    cas = rec.cassette(a.trace.draft_events)
    cas["meta"] = {"user_turns": [t["user"] for t in transcript], "command": "python experiments/report_s21_warning.py"}
    CASSETTE.parent.mkdir(parents=True, exist_ok=True)
    CASSETTE.write_text(json.dumps(cas, indent=1, ensure_ascii=False) + "\n")
    res = {"command": "python experiments/report_s21_warning.py", "model": cfg["model"], "ollama": info["runtime_version"], "transcript": transcript,
           "proposed": a.state.proposal is not None, "launched": len(a.state.jobs), "proposal_card_has_s21_warning": warned,
           "draft_status_warnings": a.state.draft.status()["warnings"], "audit_trace_problems": audit_trace(a.trace.draft_events),
           "exchanges": len(rec.exchanges), "cassette": str(CASSETTE.relative_to(ROOT))}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("transcript", "draft_status_warnings")}, indent=1, default=str))
    for t in transcript:
        print("you>", t["user"], "\nassistant>", t["assistant"])
        for c in t["cards"]:
            print("  card:", c["title"], *[f"\n    {ln}" for ln in c["lines"]])


if __name__ == "__main__":
    main()
