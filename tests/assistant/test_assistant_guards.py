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

    def chat(self, messages, tools, *, seed=0, think=None):
        self.offered.append([t["function"]["name"] for t in tools])
        self.thinks = getattr(self, "thinks", []) + [think]
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


ROUND3 = {**DEFAULT_GUARDS, "spade_answer_cards": False, "strip_draft_numbers": False}  # round-3 behaviour, before F1.9


def test_question_about_spade_requires_a_lookup_and_a_verified_citation():
    a, _ = agent(reply("Es un conjunto de decisiones."),
                 reply("", call("set_mode", mode="question"), call("search_docs", query="S-21 converter")),
                 reply("Es un artefacto del convertidor (según los documentos)."),
                 reply("Es un artefacto del convertidor de un cuadrante (docs/SIMULATION_STABILITY_REPORT.md)."), guards=ROUND3)
    res = a.user_turn("¿Qué es el hallazgo S-21?")
    assert [n["key"] for n in kinds(a, "note")] == ["docs", "verified"]  # "según los documentos" is not a citation
    assert "SIMULATION_STABILITY_REPORT" in res.text


def test_verified_citation_needs_an_existing_path_or_id():
    from driveflow.assistant.glossary import verified_citation

    assert verified_citation("ver docs/VALIDATION_SUMMARY.md") and verified_citation("ver AI-N12")
    assert not verified_citation("según los documentos") and not verified_citation("docs/NO_EXISTE.md") and not verified_citation("AI-N999")


def test_not_documented_only_after_two_empty_distinct_searches():
    # search_docs returned hits: "not documented" gets a note
    a, _ = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="abstention threshold")),
                 reply("No lo tengo documentado."), reply("Es el umbral de AI-N12 (docs/VALIDATION_SUMMARY.md)."))
    res = a.user_turn("What does the abstention threshold of the classifiers mean?")
    assert "nodoc" in [n["key"] for n in kinds(a, "note")] and "AI-N12" in res.text
    # two distinct queries with no hits: allowed
    b, _ = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="zzqx wvyk"), call("search_docs", query="qqzz kkvw")),
                 reply("No lo tengo documentado."))
    b.user_turn("¿Qué es el módulo zzqx de SPADE?")
    assert "nodoc" not in [n["key"] for n in kinds(b, "note")]


def test_glossary_term_triggers_the_lookup():
    a, _ = agent(reply("It is when the classifier does not answer."), reply("", call("set_mode", mode="question"), call("search_docs", query="abstention threshold")),
                 reply("See AI-N12 in docs/VALIDATION_SUMMARY.md."))
    a.user_turn("What does the abstention threshold of the classifiers mean?")
    assert [n["key"] for n in kinds(a, "note")][0] == "docs"


def test_tool_results_become_verbatim_cards_and_named_plants_get_catalog_cards():
    a, _ = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="abstention threshold")),
                 reply("El umbral de abstención está en AI-N12 (docs/VALIDATION_SUMMARY.md); el motor dc_perm_ex_v1 no se usa aquí."), guards=ROUND3)
    res = a.user_turn("What does the abstention threshold of the classifiers mean?")
    srcs = [c["source"] for c in res.cards]
    assert "docs/VALIDATION_SUMMARY.md" in srcs
    plant = [c for c in res.cards if c["title"] == "Plant dc_perm_ex_v1"]
    assert plant and any(ln.startswith("Nominal values: omega 300") for ln in plant[0]["lines"])  # from the catalog, not the model
    sent = [m for m in a.messages if m["role"] == "tool" and m.get("tool_name") == "search_docs"][0]["content"]
    assert "shown to the user verbatim" in sent


def test_spade_answer_is_the_card_plus_one_checked_sentence():
    """F1.9: the first sentence stays only if its numbers, IDs, parameter names and negations are in the cards."""
    q = "What does the abstention threshold of the classifiers mean?"
    look = reply("", call("set_mode", mode="question"), call("search_docs", query="abstention threshold"))
    a, _ = agent(look, reply("AI-N12 covers the abstention threshold. It also explains many other things at length."))
    res = a.user_turn(q)
    assert res.text == "AI-N12 covers the abstention threshold." and any(c["source"] == "docs/VALIDATION_SUMMARY.md" for c in res.cards)
    # an inverted claim (a negation the card does not make) is dropped: only the card remains
    b, _ = agent(look, reply("Abstention never applies to ball faults since 2019."))
    res = b.user_turn(q)
    assert res.text == "From SPADE's documentation (see the card):" and res.cards
    assert [e for e in b.trace.events if e["kind"] == "postprocess"][0]["kept_sentence"] is False
    # a parameter name that is not in the card is dropped too
    c, _ = agent(look, reply("Set `pi_setpoint_weight` to tune it."))
    assert c.user_turn(q).text.startswith("From SPADE")


def test_draft_numbers_are_not_repeated_by_the_model():
    ops = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"}, {"op": "set", "field": "duration_s", "value": 0.5, "unit": "s"},
           {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}]
    a, _ = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=ops)),
                 reply("Listo: PI a 100 rad/s durante 0,5 s. ¿Lo propongo?"))
    res = a.user_turn("Simula el motor DC de stock con PI a 100 rad/s durante 0,5 s.")
    assert res.text == "¿Lo propongo?"
    from driveflow.assistant.answer import draft_panel

    rows = {r["field"]: r for r in draft_panel(a.state.draft)}
    assert rows["omega_ref_rad_s"]["value"] == 100 and rows["omega_ref_rad_s"]["origin"] == "USER"
    b, _ = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=ops)), reply("PI a 100 rad/s."))
    assert b.user_turn("Simula el motor DC de stock con PI a 100 rad/s durante 0,5 s.").text == "Revisa el borrador en el panel y confirma la propuesta si está bien."


def test_update_draft_retries_are_capped_then_the_user_is_asked():
    bad = [{"op": "set", "field": "omega_ref_rad_s", "value": 777, "unit": "rad/s"}]  # not in the user's words: refused
    seq = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=[{"op": "set_plant", "plant": "dc_perm_ex_v1"}]))]
    seq += [reply("", call("update_draft", ops=bad + [{"op": "set", "field": "duration_s", "value": 0.1 * k, "unit": "s"}])) for k in range(1, 4)]
    seq += [reply("¿A qué velocidad?")]
    a, client = agent(*seq)
    res = a.user_turn("Simula el motor DC con PI a su velocidad nominal.")
    assert res.stopped == "update_draft_retries" and res.text == "¿A qué velocidad?"
    assert client.offered[-1] == [] and sum(1 for e in kinds(a, "tool_result") if e["name"] == "update_draft") == 4  # 1 ok + 3 failed (1 try + 2 retries)


def test_final_answer_without_tools_has_thinking_off():
    a, client = agent(reply("", call("nope"), call("nope2"), call("nope3")), reply("No pude; dime la planta."))
    a.user_turn("algo")
    assert client.offered[-1] == [] and client.thinks[-1] is False and client.thinks[0] is None


def test_objective_mode_shows_a_tunables_card():
    ops = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"}]
    a, _ = agent(reply("", call("set_mode", mode="objective"), call("update_draft", ops=ops), call("describe_tunable")), reply("¿Qué sobrepaso máximo quieres?"))
    res = a.user_turn("quiero que el motor DC responda más rápido")
    card = [c for c in res.cards if c["title"].startswith("Tunable parameters")][0]
    assert any(ln.startswith("setpoint_weight: range") for ln in card["lines"])


def test_cards_render_for_the_cli():
    from driveflow.assistant.cards import plant_card, render_text

    out = render_text([plant_card("dc_perm_ex_v1")])
    assert "source: configs/plant_catalog.yaml" in out and "Nominal values" in out


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
