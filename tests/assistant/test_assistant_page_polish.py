"""FE polish (M6.3, 2026-11-12; branch feat/FE-assistant-polish, NOT merged, M-D6): presentation only.
Visible response time, a minimum time on screen for "thinking", coloured origin chips with a legend, the model digest
in the header's tooltip and a link from job progress to Experiment Runs. The assistant's logic is not touched."""

import time

import driveflow.viz.views.assistant as v
from tests.assistant.test_assistant_page import OPS, call, page, reply, texts


def test_each_reply_shows_its_response_time(tmp_path):
    ex = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Confirmas?")]
    at = page(tmp_path, ex).run()
    t0 = time.monotonic()
    at.chat_input[0].set_value("Simula el motor DC de stock con PI a 100 rad/s durante 0,1 s.").run()
    assert not at.exception
    assert time.monotonic() - t0 >= v.MIN_THINKING_S  # the fast replayed reply still shows "thinking" for the minimum
    assert any(c.value.startswith("Answered in ") and c.value.endswith(" s · local model") for c in at.caption)
    hist = at.session_state["assistant_history"]
    assert hist[-1]["role"] == "assistant" and hist[-1]["wall_s"] >= 0


def test_origin_chips_have_a_colour_and_a_legend_and_keep_their_text(tmp_path):
    ex = [reply("", call("set_mode", mode="configuration"), call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Confirmas?")]
    at = page(tmp_path, ex).run()
    at.chat_input[0].set_value("Simula el motor DC de stock con PI a 100 rad/s durante 0,1 s.").run()
    df = at.dataframe[0].value
    assert list(df[df["Field"] == "omega_ref_rad_s"]["Origin"]) == ["USER"]  # the text stays: colour is never the only signal
    assert v.ORIGIN_LEGEND in texts(at)
    assert "background-color: #dbeafe" in v._origin_css("USER") and v._origin_css("DEFAULT/USER").startswith("background-color: #e5e7eb")
    assert v._origin_css("—") == ""


def _contrast(fg, bg):
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def test_origin_chip_colours_meet_wcag_aa():
    for name, (bg, fg) in v.ORIGIN_STYLE.items():
        assert _contrast(fg, bg) >= 4.5, name


def test_header_tooltip_carries_the_model_digest(tmp_path):
    at = page(tmp_path).run()
    head = [c for c in at.caption if c.value.startswith("Local model")][0]
    assert "digest" in (head.help or "")
