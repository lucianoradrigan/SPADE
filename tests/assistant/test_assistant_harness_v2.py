"""AI-N30 item 3(b) (Luciano, 2026-11-08): harness fixes, designed by reading the F1.11 conversations (declared;
F1.12 is the independent test). Scripted model replies, no Ollama."""

from driveflow.assistant import catalog, guardrails, objective, tools
from driveflow.assistant.catalog import search_docs
from driveflow.assistant.core import llm
from driveflow.assistant.core.loop import DEFAULT_GUARDS, NOTE_PREFIX, Agent
from driveflow.assistant.core.policy import SessionState

ALL = tools.CONFIG_TOOLS + objective.OBJECTIVE_TOOLS + catalog.CATALOG_TOOLS


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


def agent(*replies):
    c = llm.ReplayClient({"format": 1, "exchanges": list(replies)}, strict=False)
    return Agent(c, ALL, guards=DEFAULT_GUARDS)


# ----------------------------------------------------------------------------- "not documented" (search)
def test_search_docs_is_empty_for_what_spade_does_not_document():
    s = SessionState()
    for q in ("CAN bus dSPACE MicroLabBox hardware-in-the-loop", "AI-N99", "hardware-in-the-loop CAN bus interface dSPACE"):
        assert search_docs(s, q)["hits"] == [], q


def test_search_docs_still_finds_what_spade_documents():
    s = SessionState()
    for q, f in (("hallazgo S-21", "docs/SIMULATION_STABILITY_REPORT.md"), ("dc_converter_model dcm discontinuous conduction", None),
                 ("diagnosis grid dataset sensor noise model", None), ("AI-N29 clasificador PC", "docs/AI_TECHNICAL_AUDIT.md")):
        hits = search_docs(s, q)["hits"]
        assert hits, q
        if f:
            assert hits[0]["file"] == f, (q, hits[0]["file"])


def test_generic_words_do_not_make_a_query_specific():
    assert catalog.distinctive_terms("What does the SPADE page support?") == []
    assert catalog.distinctive_terms("IGBT junction temperature thermal model SPADE") == ["igbt", "junction", "temperature", "thermal"]


# ------------------------------------------------------------------------------------ missing card
def test_missing_required_values_are_asked_with_a_card():
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=[{"op": "set_plant", "plant": "pmsm_v1"}])),
              reply("Listo, configuré el PMSM."))
    res = a.user_turn("Simula el PMSM durante 0,1 s.")
    card = [c for c in res.cards if c["tool"] == "ask"][0]
    assert any(ln.startswith("controller_type") for ln in card["lines"]) and any(ln.startswith("pmsm_current_ref_a") for ln in card["lines"])
    assert res.text.endswith(guardrails.ASK["es"])  # the model did not ask: the interface does


def test_suggestion_has_a_card_with_its_source():
    from driveflow.assistant.cards import card_from_tool

    c = card_from_tool("suggest_value", {}, {"field": "omega_ref_rad_s", "value": 300.0, "unit": "rad/s", "rule": "the motor's rated speed", "source": "stock motor nominal values"})[0]
    assert c["title"] == "Suggestion for omega_ref_rad_s" and c["source"] == "stock motor nominal values" and c["lines"] == ["300.0 rad/s (the motor's rated speed)"]
    none = card_from_tool("suggest_value", {}, {"field": "l_a", "value": None, "note": "no internal source"})[0]
    assert none["title"].startswith("No suggestion")


# ------------------------------------------------------------------------------ bounded free sentences
def test_capability_claims_are_replaced_by_the_tools_words():
    a = agent(reply("", call("set_mode", mode="configuration"), call("recommend_plant", description="motor de inducción")),
              reply("SPADE no puede entrenar clasificadores ni simular fallas. Puedo simular un PMSM."),
              reply("SPADE no puede entrenar clasificadores ni simular fallas. Puedo simular un PMSM."))  # after the nudge_draft note
    res = a.user_turn("Simula un motor de inducción de jaula de ardilla.")
    assert "no puede entrenar" not in res.text and guardrails.LIMITS_LEAD["es"] in res.text  # F1.12 fix: lead-in after the kept text
    lim = [c for c in res.cards if c["tool"] == "limits"][0]
    assert any("induction motors are not simulated" in ln for ln in lim["lines"])
    assert "Puedo simular un PMSM." in res.text  # a sentence without a limit claim is kept, whole


def test_launch_claims_without_a_launch_are_removed():
    a = agent(reply("La simulación se está ejecutando con los parámetros confirmados. ¿Algo más?"),
              reply("La simulación se está ejecutando con los parámetros confirmados. ¿Algo más?"))  # after the nudge_docs note
    res = a.user_turn("[SPADE runtime note] El usuario confirmó la propuesta. Ejecuta run_experiments.")
    assert "se está ejecutando" not in res.text and guardrails.NO_LAUNCH["es"] in res.text


def test_card_answer_only_in_question_mode():
    """F1.11 m02: in configuration mode a documentation card no longer replaces the assistant's question."""
    a = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=[{"op": "set_plant", "plant": "pmsm_v1"}, {"op": "set_controller", "controller_type": "FOC_naive"}]),
                    call("explain_spade", topic="FOC_naive")),
              reply("¿Qué referencia de corriente quieres, en A?"))
    res = a.user_turn("Simula el PMSM con FOC_naive durante 0,1 s.")
    assert res.text.startswith("¿Qué referencia de corriente quieres")


def test_question_mode_adds_the_catalog_card_of_a_named_plant():
    a = agent(reply("", call("set_mode", mode="question"), call("search_docs", query="PMSM controllers")), reply("Ver la tarjeta."))
    res = a.user_turn("Which controllers can I use with the PMSM plant in SPADE?")
    assert any(c["title"] == "Plant pmsm_v1" for c in res.cards)


def test_notes_have_the_system_role():
    a = agent(reply("Respuesta sin herramientas."), reply("Sigo sin herramientas."))
    a.state.mode = "configuration"
    a.user_turn("Simula algo")
    notes = [m for m in a.messages if m["content"].startswith(NOTE_PREFIX)]
    assert notes and all(m["role"] == "system" for m in notes)
