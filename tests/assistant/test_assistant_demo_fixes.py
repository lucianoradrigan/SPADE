"""Demo fixes (Luciano, 2026-11-08), after the milestone demo "Demo interna del asistente" (f1_10_demo.md):

1. Truncated text: the number filter removes WHOLE sentences / lines, never cuts one; a header left
   without content goes with them; the tuning proposal shows its requirements and values in the card.
2. Results too early: summarize_results / check_requirements are not visible to the model until the
   job is terminal; while it runs the avatar stays ``running``, never ``error``.
3. Text not refreshed: processing the jobs (worker button, /worker) updates the assistant's message and
   its state, without waiting for the next question.
"""

import json
import re
import textwrap

from streamlit.testing.v1 import AppTest

from driveflow.assistant import answer, cards, catalog, objective, tools
from driveflow.assistant.core import llm, policy
from driveflow.assistant.core.loop import DEFAULT_GUARDS, Agent
from driveflow.assistant.draft import Draft
from driveflow.viz.adapters.assistant_adapter import avatar_state

ALL = tools.CONFIG_TOOLS + objective.OBJECTIVE_TOOLS + catalog.CATALOG_TOOLS
OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v2"}, {"op": "set_controller", "controller_type": "PI"},
       {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}, {"op": "set", "field": "duration_s", "value": 0.1, "unit": "s"}]
USER = "Simula el motor DC de stock con PI a 100 rad/s durante 0,1 s."


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


class Recording(llm.ReplayClient):
    def __init__(self, *replies):
        super().__init__({"format": 1, "exchanges": list(replies)}, strict=False)
        self.offered = []

    def chat(self, messages, tools, *, seed=0, think=None):
        self.offered.append([t["function"]["name"] for t in tools])
        return super().chat(messages, tools, seed=seed)


def agent(*replies):
    client = Recording(*replies)
    return Agent(client, ALL, guards=DEFAULT_GUARDS), client


def worker():
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    return Worker(workspace_root()).serve(once=True, poll_s=0.05)


def objective_draft():
    d = Draft()
    d.objective = {"requirements": [{"metric": "overshoot_pct", "value": 5}, {"metric": "settling_time_s", "value": 0.2}], "evaluations": {"value": 48, "origin": "default"}}
    return d


# ------------------------------------------------------------------------------------ 1. truncated text
#: what the model wrote in the milestone demo (B_objective, turn 2), with its line breaks
DEMO_RAW = ("He preparado la propuesta de sintonización para el motor DC de stock (dc_perm_ex_v2) con controlador PI:\n"
            "- **Objetivo**: Minimizar el criterio J estándar\n- **Algoritmo**: PSO (por defecto)\n- **Evaluaciones**: 48 (por defecto)\n"
            "- **Requisitos**:\n  - Sobrepaso < 5 %\n  - Tiempo de establecimiento < 0,2 s\n¿Confirmas que deseas lanzar esta sintonización?")


def _kept_units_are_whole(raw, out):
    whole = {u.strip() for u, _ in answer.units(raw)}
    return all(u.strip() in whole for u, _ in answer.units(out))


def test_the_demo_proposal_loses_its_requirement_lines_and_their_header():
    out, removed = answer.strip_draft_numbers(DEMO_RAW, objective_draft())
    assert out == ("He preparado la propuesta de sintonización para el motor DC de stock (dc_perm_ex_v2) con controlador PI:\n"
                   "- **Objetivo**: Minimizar el criterio J estándar\n- **Algoritmo**: PSO (por defecto)\n¿Confirmas que deseas lanzar esta sintonización?")
    assert removed == 4  # evaluations, the two requirement lines and the "Requisitos" header left empty
    assert "Requisitos" not in out and _kept_units_are_whole(DEMO_RAW, out)


def test_an_inline_header_with_its_numbers_goes_whole():
    raw = "- **Requisitos**: sobrepaso < 5 % y establecimiento < 0,2 s.\n¿Confirmas?"
    assert answer.strip_draft_numbers(raw, objective_draft()) == ("¿Confirmas?", 1)


def test_a_header_whose_content_survives_stays():
    raw = "Plan:\n- PSO con 48 evaluaciones\n- criterio J\n¿Sigo?"
    assert answer.strip_draft_numbers(raw, objective_draft())[0] == "Plan:\n- criterio J\n¿Sigo?"


def test_abbreviations_and_decimals_never_split_a_sentence():
    d = objective_draft()
    assert answer.strip_draft_numbers("Usaré aprox. 48 evaluaciones, e.g. PSO. ¿Sigo?", d) == ("¿Sigo?", 1)
    assert answer.strip_draft_numbers("Uso PSO, e.g. Tuning Studio. Son 48 evaluaciones. ¿Sigo?", d)[0] == "Uso PSO, e.g. Tuning Studio. ¿Sigo?"
    assert answer.strip_draft_numbers("Asentamiento < 0.2 s. Listo.", d)[0] == "Listo."
    for raw in (DEMO_RAW, "Usaré aprox. 48 evaluaciones, e.g. PSO. ¿Sigo?", "A 5 %. B. C: 0,2 s! D?"):
        out, _ = answer.strip_draft_numbers(raw, d)
        assert _kept_units_are_whole(raw, out), (raw, out)


def test_units_rejoin_to_the_text():
    for raw in (DEMO_RAW, "Uno. Dos!  ¿Tres?\n\n- cuatro\n  - cinco", "e.g. 5 s. Fin."):
        assert "".join(u + sep for u, sep in answer.units(raw)) == raw


def test_the_tuning_proposal_card_lists_every_requirement_with_its_value():
    result = {"plant": "dc_perm_ex_v2", "base_motor": "stock motor", "tunes": "speed PI", "objective": "J", "algorithm": "pso",
              "evaluations": {"value": 48, "origin": "default", "source": "x"}, "seed": 0,
              "requirements": [{"metric": "overshoot_pct", "op": "<=", "value": 5, "unit": "%"}, {"metric": "settling_time_s", "op": "<=", "value": 0.2, "unit": "s"}],
              "verification": "evaluation grid", "draft_hash": "h", "proposal": "show"}
    lines = cards.card_from_tool("propose_tuning", {}, result)[0]["lines"]
    assert "requirement: overshoot_pct <= 5 %" in lines and "requirement: settling_time_s <= 0.2 s" in lines
    assert "evaluations: 48 (default)" in lines and "objective: J" in lines and "algorithm: pso" in lines


def test_objective_conversation_requirements_in_the_card_not_in_the_text():
    reqs = [{"metric": "overshoot_pct", "op": "<=", "value": 5, "unit": "%"}, {"metric": "settling_time_s", "op": "<=", "value": 0.2, "unit": "s"}]
    a, _ = agent(reply("", call("set_mode", mode="objective"), call("update_draft", ops=OPS[:2]), call("update_objective", requirements=reqs, use_defaults=True)),
                 reply("", call("propose_tuning")), reply(DEMO_RAW))
    res = a.user_turn("Sintoniza el PI del motor DC de stock para que el sobrepaso sea menor a 5 % y el tiempo de establecimiento menor a 0,2 s.")
    assert a.state.proposal is not None
    assert "Requisitos" not in res.text and not re.search(r"\b(5|0,2|48)\b", res.text) and res.text.endswith("¿Confirmas que deseas lanzar esta sintonización?")
    card = [c for c in res.cards if c["title"] == "Tuning proposal"][0]
    assert "requirement: overshoot_pct <= 5 %" in card["lines"] and "requirement: settling_time_s <= 0.2 s" in card["lines"]


# ------------------------------------------------------------------------------ 2. results too early
def test_result_tools_hidden_until_the_job_is_terminal():
    s = policy.SessionState(mode="configuration")
    s.jobs.append({"job_id": "j1", "kind": "scenario_sweep", "status": "queued"})
    for st in ("queued", "preparing", "running", "cancelling"):
        s.jobs[0]["status"] = st
        assert "summarize_results" not in policy.visible_tools(s) and "get_job_status" in policy.visible_tools(s)
        assert "has not finished" in policy.check_call(s, "summarize_results")
    for st in ("completed", "partially_completed", "failed", "cancelled"):
        s.jobs[0]["status"] = st
        assert "summarize_results" in policy.visible_tools(s) and policy.check_call(s, "summarize_results") is None


def test_after_run_the_model_cannot_summarize_and_the_avatar_stays_running():
    a, client = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo?"),
                      reply("", call("run_experiments")), reply("", call("get_job_status"), call("summarize_results")), reply("Está corriendo."))
    a.user_turn(USER)
    a.user_turn("/run", confirmed_how="button")
    assert a.state.jobs and a.state.jobs[0]["status"] == "queued"
    assert all("summarize_results" not in offered for offered in client.offered)  # never offered while the job is not terminal
    refused = [e for e in a.trace.events if e["kind"] == "tool_refused" and e["name"] == "summarize_results"]
    assert refused and "has not finished" in refused[0]["reason"]
    assert not [e for e in a.trace.agent_events if e["type"] == "error"]
    assert avatar_state(a.trace.agent_events)[0] == "running"


def test_the_loop_reads_the_job_state_before_each_model_call():
    a, client = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo?"),
                      reply("", call("run_experiments")), reply("Lanzado."), reply("", call("summarize_results")), reply("Listo."))
    a.user_turn(USER)
    a.user_turn("/run", confirmed_how="button")
    assert worker() == 1
    a.user_turn("¿Resultados?")  # nobody called get_job_status: the loop read the state itself
    assert "summarize_results" in client.offered[-2]
    assert [e for e in a.trace.events if e["kind"] == "tool_result" and e["name"] == "summarize_results"]
    assert avatar_state(a.trace.agent_events)[0] == "results_ready"


# --------------------------------------------------------------------------- 3. text not refreshed
def test_processing_the_jobs_updates_the_message_and_the_state():
    a, _ = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo?"),
                 reply("", call("run_experiments")), reply("Lanzado."))
    a.user_turn(USER)
    a.user_turn("/run", confirmed_how="button")
    assert a.job_updates() == []  # still queued: nothing to say
    assert worker() == 1
    ups = a.job_updates()
    assert len(ups) == 1 and ups[0]["state"] == "completed"
    assert ups[0]["text"] == f"El job {a.state.jobs[0]['job_id']} terminó. Los resultados (simulados) están en la tarjeta."
    assert ups[0]["cards"][0]["title"].startswith("Results (SIMULATED · AI-PLANNED)")
    assert avatar_state(a.trace.agent_events)[0] == "results_ready"
    assert a.messages[-1] == {"role": "assistant", "content": ups[0]["text"]}  # the model knows the job ended
    assert a.job_updates() == []  # once per transition


def test_a_failed_job_says_so_without_results():
    a, _ = agent()
    a.state.mode = "configuration"
    a.state.jobs.append({"job_id": "job-x", "kind": "scenario_sweep", "status": "running"})
    a.job_refresher = lambda st: [{"job_id": "job-x", "kind": "scenario_sweep", "from": "running", "to": "failed", "done": 0, "total": 1}]
    ups = a.job_updates(lang="en")
    assert ups == [{"job_id": "job-x", "state": "failed", "text": "Job job-x failed; there are no results.", "cards": []}]


def test_a_finished_tuning_job_gets_the_requirements_card(monkeypatch):
    a, _ = agent()
    a.state.mode = "objective"
    a.state.jobs.append({"job_id": "job-t", "kind": "controller_tuning", "status": "running"})
    a.job_refresher = lambda st: [{"job_id": "job-t", "kind": "controller_tuning", "from": "running", "to": "completed", "done": 48, "total": 48}]
    fake = {"origin": "SIMULATED · AI-PLANNED", "run_id": "run-t", "verdict": "candidate 'best' meets all 2 requirements on the evaluation grid", "all_met": True}
    monkeypatch.setattr(a.tools["check_requirements"], "fn", lambda state, job_id=None: fake)
    ups = a.job_updates(lang="es")
    assert ups[0]["cards"][0]["title"] == "Requirements check (SIMULATED)" and any("meets all 2" in ln for ln in ups[0]["cards"][0]["lines"])
    assert avatar_state(a.trace.agent_events)[0] == "results_ready"


def test_cli_worker_prints_the_update():
    from driveflow.assistant import cli

    cas = {"format": 1, "exchanges": [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")),
                                      reply("¿Lanzo?"), reply("", call("run_experiments")), reply("Lanzado.")]}
    out = []
    cli.run([USER, "/run", "/worker"], out=out.append, client=llm.ReplayClient(cas, strict=False))
    joined = "\n".join(out)
    assert "terminó. Los resultados (simulados) están en la tarjeta." in joined and "Results (SIMULATED · AI-PLANNED)" in joined


def test_page_worker_button_updates_the_chat_and_the_avatar(tmp_path):
    ex = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo?"),
          reply("", call("run_experiments")), reply("Lanzado.")]
    cas = tmp_path / "cassette.json"
    cas.write_text(json.dumps({"format": 1, "exchanges": ex}))
    at = AppTest.from_string(textwrap.dedent(f"""
        from driveflow.assistant.core import llm
        import driveflow.viz.views.assistant as v
        v.CLIENT_FACTORY = lambda: llm.ReplayClient({str(cas)!r}, strict=False)
        v.STREAM_DELAY_S = 0
        v.render()
    """), default_timeout=120).run()
    at.chat_input[0].set_value(USER).run()
    [b for b in at.button if b.label == "Run"][0].click().run()
    agent_ = at.session_state["assistant_agent"]
    assert avatar_state(at.session_state["assistant_events"])[0] == "running"
    [b for b in at.button if b.label.startswith("Process queued jobs")][0].click().run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert f"El job {agent_.state.jobs[0]['job_id']} terminó." in md  # without a new question
    assert any(m.value.startswith("**Results (SIMULATED · AI-PLANNED)") for m in at.markdown)
    assert avatar_state(at.session_state["assistant_events"])[0] == "results_ready"


def test_a_turn_stopped_by_repeated_status_calls_leaves_the_avatar_running():
    """The milestone re-run (demo_fixes_live.json, A): after Run the model repeated get_job_status
    until the turn stopped after 3 refused calls; the job was queued, so the avatar stays running."""
    a, _ = agent(reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo?"),
                 reply("", call("run_experiments"), call("get_job_status")), reply("", call("get_job_status")), reply("", call("get_job_status")),
                 reply("", call("get_job_status")), reply("Sigue en cola."))
    a.user_turn(USER)
    res = a.user_turn("/run", confirmed_how="button")
    assert res.stopped == "errors"
    assert not [e for e in a.trace.agent_events if e["type"] == "error"]
    assert avatar_state(a.trace.agent_events)[0] == "running"


def test_another_tool_error_while_a_job_runs_still_shows_error():
    from driveflow.assistant import events as ev

    events = [ev.JobEvent("j", "running", 0, 1), ev.TurnStarted(2), ev.ErrorEvent("tool:update_draft", "bad op"), ev.TurnEnded()]
    assert avatar_state(events)[:2] == ("error", "bad op")
    assert avatar_state([ev.JobEvent("j", "running", 0, 1), ev.TurnStarted(2), ev.TurnEnded(stopped="max_tool_calls_per_turn")])[0] == "running"
    assert avatar_state([ev.TurnStarted(2), ev.TurnEnded(stopped="max_tool_calls_per_turn")])[0] == "error"  # no job: unchanged


def test_a_table_whose_rows_were_removed_goes_whole():
    """The re-run (demo_fixes_live.json, B, last turn): the rows carried the draft's thresholds."""
    raw = ("**Sí, los requisitos se cumplen** con \"best\":\n\n| Requisito | Límite | best |\n|---|---|---|\n| Sobrepaso | ≤ 5% | 4.87% |\n"
           "| Establecimiento | ≤ 0,2 s | 0,053 s |\n\n**Configuración:**\n- damping 1.04\n¿Sigo?")
    out, _ = answer.strip_draft_numbers(raw, objective_draft())
    assert out == "**Configuración:**\n- damping 1.04\n¿Sigo?"
    raw2 = "Tabla:\n| a | b |\n|---|---|\n| x | 5 |\n| y | 7 |\nFin."
    assert answer.strip_draft_numbers(raw2, objective_draft())[0] == "Tabla:\n| a | b |\n|---|---|\n| y | 7 |\nFin."
