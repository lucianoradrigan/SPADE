"""F1.1 round 2: loop guards and nudges (core/loop.py DEFAULT_GUARDS), with scripted model replies.

Designed by reading the F1.1 conversations (docs/audit/assistant/f1_1_model_choice_round2_rule.md):
identical calls refused, per-turn caps on search_docs / explain_spade, a final answer without tools on
any limit and never an empty answer, update_objective hidden until a plant is set, a note when the
model answers without recording anything, and the look-up / citation notes for questions about SPADE."""

from driveflow.assistant import catalog, objective, tools
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import DEFAULT_GUARDS, EMPTY_FALLBACK, GUARDS_OFF, NOTE_PREFIX, STOP_MESSAGES, Agent
from driveflow.assistant.core.policy import SessionState, visible_tools

ALL = tools.CONFIG_TOOLS + objective.OBJECTIVE_TOOLS + catalog.CATALOG_TOOLS


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


class Recording(llm.ReplayClient):
    """ReplayClient (non-strict) that also records the tool names offered on each call."""

    def __init__(self, *replies):
        super().__init__({"format": 1, "exchanges": list(replies)}, strict=False)
        self.offered = []

    def chat(self, messages, tools, *, seed=0):
        self.offered.append([t["function"]["name"] for t in tools])
        self.last_messages = list(messages)
        return super().chat(messages, tools, seed=seed)


def agent(*replies, guards=DEFAULT_GUARDS, mode=None):
    client = Recording(*replies)
    st = SessionState(mode=mode) if mode else None
    return Agent(client, ALL, guards=guards, state=st), client


def kinds(a, kind):
    return [e for e in a.trace.events if e["kind"] == kind]


def test_identical_call_in_the_same_turn_is_refused():
    a, _ = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="abstention threshold")),
                 reply("", call("search_docs", query="abstention threshold")), reply("Según docs/AI_TECHNICAL_AUDIT.md, …"))
    a.user_turn("What is the abstention threshold?")
    assert [e["name"] for e in kinds(a, "tool_result")].count("search_docs") == 1
    assert [e["reason"] for e in kinds(a, "tool_refused")] == ["repeat"]


def test_search_docs_is_capped_at_two_per_turn():
    a, _ = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="converter artifact"), call("search_docs", query="one quadrant"),
                       call("search_docs", query="negative current")), reply("Fuente: docs/SIMULATION_STABILITY_REPORT.md, S-21."))
    a.user_turn("¿Qué es S-21?")
    assert [e["name"] for e in kinds(a, "tool_result")].count("search_docs") == 2
    assert [e["reason"] for e in kinds(a, "tool_refused")] == ["cap"]


def test_a_limit_ends_with_a_final_answer_without_tools():
    bad = reply("", call("nope"), call("nope2"), call("nope3"))
    a, client = agent(bad, reply("No pude completar el borrador; dime la planta."))
    res = a.user_turn("algo")
    assert res.stopped == "errors" and res.text == "No pude completar el borrador; dime la planta."
    assert client.offered[-1] == [] and client.last_messages[-1]["content"].startswith(NOTE_PREFIX)
    assert kinds(a, "model")[-1].get("final") is True


def test_no_turn_ends_with_an_empty_answer():
    a, client = agent(reply("", call("set_mode", mode="question")), reply(""), reply("SPADE simula motores y convertidores (docs/README)."))
    assert a.user_turn("hola").text.startswith("SPADE simula")
    assert client.offered[-1] == []
    b, _ = agent(reply(""), reply(""))
    assert b.user_turn("hola").text == EMPTY_FALLBACK


def test_update_objective_hidden_until_a_plant_is_set():
    s = SessionState(mode="objective")
    s.objective_needs_plant = True
    assert "update_objective" not in visible_tools(s)
    s.ledger.add_user_message(1, "motor DC")
    assert s.draft.patch([{"op": "set_plant", "plant": "dc_perm_ex_v1"}], s.ledger)["ok"]
    assert "update_objective" in visible_tools(s)
    s.objective_needs_plant = False
    assert "update_objective" in visible_tools(SessionState(mode="objective"))
    a, client = agent(reply("", call("set_mode", mode="objective")), reply("¿Qué motor?"), reply("¿Qué motor quieres sintonizar?"))
    a.user_turn("quiero un PI con sobrepaso < 5 %")
    assert all("update_objective" not in offered for offered in client.offered)


def test_text_only_answer_without_recording_gets_one_note():
    a, client = agent(reply("", call("set_mode", mode="configuration")), reply("Entendido, simularé eso."), reply("¿Qué planta quieres usar?"))
    res = a.user_turn("simula el motor DC con PI")
    notes = kinds(a, "note")
    assert [n["key"] for n in notes] == ["draft"] and res.text == "¿Qué planta quieres usar?"


def test_question_about_spade_requires_a_lookup_and_a_citation():
    a, _ = agent(reply("S-21 es un conjunto de decisiones."),
                 reply("", call("set_mode", mode="question"), call("search_docs", query="S-21 converter")),
                 reply("S-21 es un artefacto del convertidor."),
                 reply("S-21 es un artefacto del convertidor de un cuadrante (docs/SIMULATION_STABILITY_REPORT.md, S-21)."))
    res = a.user_turn("¿Qué es el hallazgo S-21?")
    assert [n["key"] for n in kinds(a, "note")] == ["docs", "cite"]
    assert "SIMULATION_STABILITY_REPORT" in res.text


def test_notes_are_not_evidence_for_the_draft():
    a, _ = agent(reply("", call("set_mode", mode="configuration")), reply("ok"), reply("¿Qué velocidad?"))
    a.user_turn("simula el motor DC")
    assert all(not m["text"].startswith(NOTE_PREFIX) for m in a.state.ledger.user_messages) if hasattr(a.state.ledger, "user_messages") else True
    assert [e for e in a.trace.draft_events if e["role"] == "user"] == [{"role": "user", "seq": 1, "text": "simula el motor DC"}]


def test_guards_off_keeps_the_previous_behaviour():
    a, client = agent(reply("", call("nope"), call("nope2"), call("nope3")), guards=GUARDS_OFF)
    res = a.user_turn("algo")
    assert res.text == STOP_MESSAGES["errors"] and len(client.offered) == 1
    assert Agent(Recording(), ALL).guards == GUARDS_OFF  # library default


def test_output_token_cap_is_sent_to_ollama(monkeypatch):
    from driveflow.assistant.config import load_config

    c = llm.OllamaClient({**load_config(), "model": {"tag": "x:1", "digest": "d"}, "num_predict": 1536})
    c.checked = True
    sent = {}
    monkeypatch.setattr(c, "_request", lambda path, payload=None, timeout=None: sent.update(payload) or {"message": {"role": "assistant", "content": "hi"}})
    c.chat([{"role": "user", "content": "hola"}], [])
    assert sent["options"]["num_predict"] == 1536
