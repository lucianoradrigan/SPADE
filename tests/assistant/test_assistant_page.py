"""F1.10 + FE-4: the Assistant page (viz/views/assistant.py) in AppTest, on recorded replies.

A complete configuration conversation (request -> proposal -> Run button -> job -> results), the
confirmation set by the page only, the draft panel with origin chips, the cards, the fixed texts,
and Ollama down (state unavailable, the page keeps working)."""

import json
import textwrap

from streamlit.testing.v1 import AppTest

OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"},
       {"op": "set", "field": "omega_ref_rad_s", "value": 100, "unit": "rad/s"}, {"op": "set", "field": "duration_s", "value": 0.1, "unit": "s"}]


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg, "prompt_eval_count": 3, "eval_count": 2}}


def page(tmp_path, exchanges=None, url=None):
    cas = tmp_path / "cassette.json"
    cas.write_text(json.dumps({"format": 1, "exchanges": exchanges or []}))
    factory = (f"lambda: llm.OllamaClient(load_config(overrides={{'base_url': {url!r}, 'timeout_s': 1}}))" if url
               else f"lambda: llm.ReplayClient({str(cas)!r}, strict=False)")
    script = textwrap.dedent(f"""
        from driveflow.assistant.config import load_config
        from driveflow.assistant.core import llm
        import driveflow.viz.views.assistant as v
        v.CLIENT_FACTORY = {factory}
        v.STREAM_DELAY_S = 0
        v.render()
    """)
    return AppTest.from_string(script, default_timeout=120)


def texts(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.text] + [w.value for w in at.warning])


def test_complete_configuration_conversation_with_the_run_button(tmp_path):
    ex = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")),
          reply("¿Confirmas la simulación de 100 rad/s?"),
          reply("", call("run_experiments")), reply("Lanzado; avísame cuando termine."),
          reply("", call("get_job_status"), call("summarize_results")), reply("Listo.")]
    at = page(tmp_path, ex).run()
    assert not at.exception
    assert "I'm SPADE's AI assistant" in texts(at) and "All results are SIMULATED." in texts(at)
    run_btn = [b for b in at.button if b.label == "Run"][0]
    assert run_btn.disabled  # nothing proposed yet
    at.chat_input[0].set_value("Simula el motor DC de stock con PI a 100 rad/s durante 0,1 s.").run()
    assert not at.exception
    t = texts(at)
    assert "Proposal: 1 experiment(s) on dc_perm_ex_v1 (PI)" in t and "Revisa el borrador en el panel" in t  # the draft number was not repeated
    df = at.dataframe[0].value
    assert list(df[df["Field"] == "omega_ref_rad_s"]["Origin"]) == ["USER"]
    run_btn = [b for b in at.button if b.label == "Run"][0]
    assert not run_btn.disabled
    run_btn.click().run()
    assert not at.exception
    agent = at.session_state["assistant_agent"]
    conf = [e for e in agent.trace.events if e["kind"] == "confirmation"]
    assert agent.state.jobs and conf[-1]["how"] == "button" and conf[-1]["recorded"]  # set by the page, consumed by the launch
    [b for b in at.button if b.label.startswith("Process queued jobs")][0].click().run()
    at.chat_input[0].set_value("¿Resultados?").run()
    assert not at.exception and "Results (SIMULATED · AI-PLANNED)" in texts(at)


def test_ollama_down_page_keeps_working(tmp_path):
    at = page(tmp_path, url="http://127.0.0.1:9").run()
    assert not at.exception
    assert "Assistant unavailable" in texts(at) and "systemctl start ollama" in texts(at)
    at.chat_input[0].set_value("hola").run()
    assert not at.exception and "Assistant unavailable" in texts(at)


def test_page_is_registered_in_intelligence_without_touching_other_pages():
    from driveflow.viz.app import PAGES

    assert ("pages/assistant.py", "Assistant", ":material/forum:") in PAGES["Intelligence"]
    assert [p for p, _t, _i in PAGES["Intelligence"]][:4] == ["pages/ai_diagnostics.py", "pages/edge_deployment.py", "pages/transfer_learning.py", "pages/model_registry.py"]
