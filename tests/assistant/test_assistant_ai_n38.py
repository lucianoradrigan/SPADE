"""AI-N38 (corrected 2026-11-15): when the draft tool refuses a frozen v1 DC plant (decision E3, S-21), the Assistant page
adds a fixed card that says why and what is simulated instead; the text is the page's, not the model's."""

import driveflow.viz.views.assistant as v
from tests.assistant.test_assistant_page import call, page, reply, texts


def test_the_refusal_is_detected_only_from_a_tool_refusal():
    err = {"kind": "tool_error", "name": "update_draft", "error": "dc_perm_ex_v1 is the frozen GEM-faithful plant (one-quadrant converter artifact S-21), ..."}
    assert v.v1_refused_in([err])
    res = {"kind": "tool_result", "name": "update_draft", "result": {"ok": False, "errors": [{"op": 0, "error": "dc_pmg132_v1 is the frozen GEM-faithful plant ..."}]}}
    assert v.v1_refused_in([res])
    assert not v.v1_refused_in([{"kind": "tool_result", "name": "update_draft", "result": {"ok": True, "errors": []}}])
    assert not v.v1_refused_in([{"kind": "model", "text": "frozen GEM-faithful plant"}, {"kind": "tool_error", "error": "unknown plant 'x'"}])


def test_the_page_shows_the_fixed_card_after_a_v1_request(tmp_path):
    ex = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=[{"op": "set_plant", "plant": "dc_perm_ex_v1"}])),
          reply("", call("update_draft", ops=[{"op": "set_plant", "plant": "dc_perm_ex_v2"}])), reply("Usaré la planta v2.")]
    at = page(tmp_path, ex).run()
    at.chat_input[0].set_value("Simula el motor dc_perm_ex_v1 con PI a 100 rad/s.").run()
    assert not at.exception
    t = texts(at)
    assert v.V1_REFUSED_CARD["title"] in t and "S-21" in t
    assert at.session_state["assistant_history"][-1]["cards"][-1]["title"] == v.V1_REFUSED_CARD["title"]


def test_no_card_without_a_v1_request(tmp_path):
    ex = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=[{"op": "set_plant", "plant": "dc_perm_ex_v2"}])), reply("Listo.")]
    at = page(tmp_path, ex).run()
    at.chat_input[0].set_value("Simula el motor DC de stock.").run()
    assert not at.exception and v.V1_REFUSED_CARD["title"] not in texts(at)
