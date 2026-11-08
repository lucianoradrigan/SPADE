"""Terminal chat with the SPADE assistant (task F1.9): ``python -m driveflow.assistant``.

After every turn it prints the assistant's sentence, the cards (SPADE data shown verbatim with its
source) and the draft panel (every value with its origin chip). The confirmation is the user's own
input, set by THIS layer and never by the model: ``sí`` / ``yes`` (policy.AFFIRMATIVE) right after a
proposal, or ``/run``. Commands: /run, /draft, /worker (run queued jobs once), /help, /quit.

If Ollama does not answer or the pinned model is missing, the CLI says how to fix it and keeps going.
``--replay <cassette.json>`` runs on a recorded conversation (ReplayClient) instead of Ollama.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from driveflow.assistant import answer, cards
from driveflow.assistant.core import llm
from driveflow.assistant.core.policy import is_affirmative

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_v1.md"
FIRST_MESSAGE = "I'm SPADE's AI assistant. I set up and run simulations; my suggestions require your verification. All results are SIMULATED."
HELP = "Commands: /run (confirm the last proposal), /draft (show the draft), /worker (run queued jobs once), /help, /quit."


def system_prompt() -> str:
    return PROMPT_PATH.read_text()


def build_agent(client=None, config: dict | None = None, on_event=None):
    from driveflow.assistant import catalog, objective, tools
    from driveflow.assistant.config import load_config
    from driveflow.assistant.core.loop import Agent

    cfg = config or load_config()
    client = client or llm.OllamaClient(cfg)
    return Agent(client, tools.CONFIG_TOOLS + objective.OBJECTIVE_TOOLS + catalog.CATALOG_TOOLS, system_prompt=system_prompt(),
                 limits=cfg.get("limits"), seed=int(cfg.get("seed", 0)), guards=cfg.get("guards"), on_event=on_event)


def unavailable_message(exc: Exception, cfg: dict) -> str:
    tag = (cfg.get("model") or {}).get("tag")
    if isinstance(exc, llm.ModelNotAvailable):
        return f"Assistant unavailable: {exc}. Download the model with: ollama pull {tag}"
    return f"Assistant unavailable: {exc}. Start the local runtime with: systemctl start ollama (or `ollama serve`)."


def render_turn(agent, res) -> str:
    out = [f"assistant> {res.text}"]
    if res.cards:
        out.append(cards.render_text(res.cards))
    out.append("draft:\n" + answer.render_panel(answer.draft_panel(agent.state.draft)))
    if agent.state.jobs:
        out.append("jobs: " + ", ".join(f"{j['job_id']} ({j.get('kind')})" for j in agent.state.jobs))
    return "\n".join(out)


def run(lines, out=print, client=None, config: dict | None = None) -> object:
    """Process user lines (an iterable); returns the agent. Used by main() and the tests."""
    from driveflow.assistant.config import load_config

    cfg = config or load_config()
    agent = build_agent(client=client, config=cfg)
    status = (cfg.get("model") or {}).get("status")
    out(FIRST_MESSAGE + (f" [model {cfg['model']['tag']}: {status}]" if status else ""))
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/help":
            out(HELP)
            continue
        if text == "/draft":
            out(answer.render_panel(answer.draft_panel(agent.state.draft)))
            continue
        if text == "/worker":
            from driveflow.jobs.worker import Worker
            from driveflow.runs.store import workspace_root

            Worker(workspace_root()).serve(once=True, poll_s=0.05)
            out("worker: queued jobs processed")
            for u in agent.job_updates():  # demo fixes: the assistant's message and the results card
                out(f"assistant> {u['text']}" + (("\n" + cards.render_text(u["cards"])) if u["cards"] else ""))
            continue
        how = None
        if agent.state.proposal is not None and (text == "/run" or is_affirmative(text)):
            how = "command: /run" if text == "/run" else f"typed: {text}"
        try:
            res = agent.user_turn(text, confirmed_how=how)
        except (llm.RuntimeUnavailable, llm.ModelNotAvailable) as exc:
            out(unavailable_message(exc, cfg))
            continue
        out(render_turn(agent, res))
    return agent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, default=None, help="run on a recorded cassette (ReplayClient) instead of Ollama")
    args = ap.parse_args(argv)
    client = llm.ReplayClient(args.replay, strict=False) if args.replay else None

    def lines():
        while True:
            try:
                yield input("you> ")
            except (EOFError, KeyboardInterrupt):
                return

    run(lines(), out=lambda s: print(s, flush=True), client=client)


if __name__ == "__main__":
    main(sys.argv[1:])
