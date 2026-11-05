"""F1.9: the terminal chat (driveflow.assistant.cli) on recorded replies (ReplayClient): a complete
configuration conversation (request -> proposal -> "sí" -> job -> results), the confirmation set by the
CLI and never by the model, the draft panel with origin chips, the cards, Ollama down, and the prompt."""

from driveflow.assistant import cli
from driveflow.assistant.config import load_config
from driveflow.assistant.core import llm

OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"},
       {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}, {"op": "set", "field": "duration_s", "value": 0.1, "unit": "s"}]


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


def replay(*replies):
    return llm.ReplayClient({"format": 1, "exchanges": list(replies)}, strict=False)


def test_complete_configuration_conversation():
    client = replay(reply("", call("set_mode", mode="configuration"), call("describe_plant", plant="dc_perm_ex_v1"), call("update_draft", ops=OPS)),
                    reply("", call("propose_experiments")), reply("¿Confirmas la simulación de 100 rad/s?"),
                    reply("", call("run_experiments")), reply("Lanzado."),
                    reply("", call("get_job_status"), call("summarize_results")), reply("Listo, revisa los resultados."))
    out = []
    agent = cli.run(["Simula el motor DC de stock con PI a 100 rad/s durante 0,1 s.", "sí", "/worker", "¿resultados?", "/quit"], out=out.append, client=client)
    text = "\n".join(out)
    assert out[0].startswith("I'm SPADE's AI assistant") and "internal-demo" in out[0]
    assert "omega_ref_rad_s  100 rad/s  [USER]" in text  # the draft panel, with the origin chip
    assert "Proposal: 1 experiment(s) on dc_perm_ex_v1 (PI)" in text and "Plant dc_perm_ex_v1" in text
    assert "¿Confirmas la simulación?" not in text and "assistant> ¿Confirmas la simulación de 100 rad/s?" not in text  # draft number removed
    conf = [e for e in agent.trace.events if e["kind"] == "confirmation"]
    assert conf and conf[0]["how"] == "typed: sí" and conf[0]["recorded"] is True
    assert agent.state.jobs and "Results (SIMULATED · AI-PLANNED)" in text


def test_a_yes_without_a_proposal_is_not_a_confirmation():
    client = replay(reply("", call("set_mode", mode="question")), reply("¿En qué te ayudo?"))
    agent = cli.run(["sí"], out=lambda s: None, client=client)
    assert not [e for e in agent.trace.events if e["kind"] == "confirmation"]


def test_ollama_down_is_reported_and_the_cli_keeps_going():
    cfg = load_config(overrides={"base_url": "http://127.0.0.1:9", "timeout_s": 1})
    out = []
    cli.run(["hola", "/help", "/quit"], out=out.append, config=cfg)
    assert any(s.startswith("Assistant unavailable") and "ollama" in s for s in out) and any(s.startswith("Commands:") for s in out)


def test_system_prompt_rules():
    p = cli.system_prompt()
    assert "never speak in the" in p and "Luciano" in p  # FE-4: not in the first person as Luciano
    assert "datasheet" not in p.lower()  # the datasheet step is phase 2
    assert "verbatim" in p and "Never invent" in p
