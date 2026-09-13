"""Tests for the landing page's system diagram and its per-component detail panel (added because
the diagram alone -- boxes/arrows with no click-to-select support in Streamlit's plotly_chart --
left what the Raspberry Pi 5/ESP32 boxes actually do unclear)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "dashboard.py")


class TestSystemDiagram:
    def test_landing_page_renders_diagram_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception

    def test_component_detail_selectbox_covers_every_diagram_box(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        options = at.selectbox(key="diagram_component_detail").options
        for expected in ["Raspberry Pi 5", "ESP32", "ESP32 watchdog", "Model registry", "Dashboard — IA tab"]:
            assert expected in options

    def test_selecting_raspberry_pi_5_explains_distillation_not_training_from_scratch(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.selectbox(key="diagram_component_detail").set_value("Raspberry Pi 5").run()
        assert not at.exception
        detail_text = " ".join(el.value for el in at.info)
        assert "distilled" in detail_text.lower()
        assert "float16" in detail_text

    def test_selecting_esp32_explains_int8_and_no_recurrent_layers(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.selectbox(key="diagram_component_detail").set_value("ESP32").run()
        assert not at.exception
        detail_text = " ".join(el.value for el in at.info)
        assert "int8" in detail_text
        assert "TCN" in detail_text

    def test_esp32_watchdog_is_distinguished_from_the_esp32_ml_tier(self):
        """The two ESP32 boxes mean different things (a trained-and-distilled classifier/
        regressor vs. a rule engine with no ML at all) -- regression test that the detail text
        doesn't conflate them."""
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.selectbox(key="diagram_component_detail").set_value("ESP32 watchdog").run()
        assert not at.exception
        detail_text = " ".join(el.value for el in at.info)
        assert "not a model" in detail_text.lower()
