"""Tests for the landing page's system diagram and its per-component detail panel (added because
the diagram alone -- boxes/arrows with no click-to-select support in Streamlit's plotly_chart --
left what the Raspberry Pi 5/ESP32 boxes actually do unclear).

There is no selectbox anymore (removed per direct feedback that it was uncomfortable -- click a
box in the diagram instead) and AppTest cannot simulate an actual click on a Plotly chart (no
plotly-chart element type in streamlit.testing.v1 at all), so these tests set
`diagram_component_detail` directly in session_state before `.run()` -- exactly the state a real
click would have written, exercising the same rendering path regardless of how that state got
set."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "dashboard.py")


class TestSystemDiagram:
    def test_landing_page_renders_diagram_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception

    def test_no_selection_yet_shows_a_prompt_instead_of_a_blank_or_default_panel(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        assert not list(at.info)
        caption_texts = " ".join(el.value for el in at.caption)
        assert "click any box above" in caption_texts.lower()

    def test_selecting_raspberry_pi_5_explains_distillation_not_training_from_scratch(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.session_state["diagram_component_detail"] = "Raspberry Pi 5"
        at.run()
        assert not at.exception
        detail_text = " ".join(el.value for el in at.info)
        assert "distilled" in detail_text.lower()
        assert "float16" in detail_text

    def test_selecting_esp32_explains_int8_and_no_recurrent_layers(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.session_state["diagram_component_detail"] = "ESP32"
        at.run()
        assert not at.exception
        detail_text = " ".join(el.value for el in at.info)
        assert "int8" in detail_text
        assert "TCN" in detail_text

    def test_esp32_watchdog_is_distinguished_from_the_esp32_ml_tier(self):
        """The two ESP32 boxes mean different things (a trained-and-distilled classifier/
        regressor vs. a rule engine with no ML at all) -- regression test that the detail text
        doesn't conflate them."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.session_state["diagram_component_detail"] = "ESP32 watchdog"
        at.run()
        assert not at.exception
        detail_text = " ".join(el.value for el in at.info)
        assert "not a model" in detail_text.lower()

    def test_every_diagram_box_has_a_detail_entry_and_renders_it(self):
        """Covers all 12 components in one pass -- catches a future box added to the diagram
        without a matching detail entry (the module also has a self-enforcing assertion for
        this, but this confirms the rendering path itself doesn't choke on any of them)."""
        for key in [
            "DC motor + PMSM (Fase A)", "VSC + trained DPC network (Fase B)", "Common data layer",
            "Classifier (CNN)", "Regressor (LSTM/GRU/TCN)", "Model registry", "Raspberry Pi 5", "ESP32",
            "Dashboard — IA tab", "ESP32 watchdog", "GatewayAgent — RPi5", "ServerAgent — PC",
        ]:
            at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
            at.session_state["diagram_component_detail"] = key
            at.run()
            assert not at.exception, f"selecting {key!r} raised: {at.exception}"
            assert list(at.info), f"selecting {key!r} produced no detail panel"
