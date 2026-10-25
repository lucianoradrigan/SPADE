"""Tests for the Overview page's system diagram + detail panel, and for both entry points.

The single-file dashboard's landing page became the Overview page of the multipage app
(driveflow.viz.app). The diagram behaviour is unchanged: AppTest cannot click a Plotly chart, so
these tests set `diagram_component_detail` in session_state before `.run()` -- exactly the state a
real click writes."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

VIZ = Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz"
APP_PATH = str(VIZ / "app.py")
DASHBOARD_PATH = str(VIZ / "dashboard.py")


class TestEntryPoints:
    def test_legacy_dashboard_command_still_works_and_opens_overview(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        assert any("Click any box above" in c.value for c in at.caption)

    def test_app_entry_point_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        assert not at.exception


class TestSystemDiagram:
    def test_no_selection_yet_shows_a_prompt_instead_of_a_default_panel(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        caption_texts = " ".join(el.value for el in at.caption)
        assert "click any box above" in caption_texts.lower()
        assert not [i for i in at.info if "distilled" in i.value.lower()]

    def test_selecting_raspberry_pi_5_explains_distillation_not_training_from_scratch(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.session_state["diagram_component_detail"] = "Raspberry Pi 5"
        at.run()
        detail_text = " ".join(el.value for el in at.info)
        assert "distilled" in detail_text.lower()
        assert "float16" in detail_text

    def test_selecting_esp32_explains_int8_and_no_recurrent_layers(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.session_state["diagram_component_detail"] = "ESP32"
        at.run()
        detail_text = " ".join(el.value for el in at.info)
        assert "int8" in detail_text
        assert "TCN" in detail_text

    def test_esp32_watchdog_is_distinguished_from_the_esp32_ml_tier(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.session_state["diagram_component_detail"] = "ESP32 watchdog"
        at.run()
        assert "not a model" in " ".join(el.value for el in at.info).lower()

    def test_every_diagram_box_has_a_detail_entry_and_renders_it(self):
        from driveflow.viz.views.system_diagram import _DIAGRAM_COMPONENT_DETAILS

        for key in _DIAGRAM_COMPONENT_DETAILS:
            at = AppTest.from_file(APP_PATH, default_timeout=60)
            at.session_state["diagram_component_detail"] = key
            at.run()
            assert not at.exception, f"selecting {key!r} raised: {at.exception}"
            assert any(_DIAGRAM_COMPONENT_DETAILS[key][:40] in i.value for i in at.info)


class TestOverviewHonesty:
    def test_motor_thumbnail_is_unknown_before_any_monitoring(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        assert any(c.value.startswith("UNKNOWN") for c in at.caption)

    def test_no_engine_ready_badge_without_checks(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        html = " ".join(str(getattr(h, "body", "")) for h in at.get("html"))
        assert "ENGINE READY" not in html.upper()
        assert "CHECKS PASS" in html


class TestSystemDiagramNumbers:
    """The diagram quotes promoted runs' metrics in prose; these checks fail when a promotion in
    configs/registry.yaml changes a number the text still carries."""

    @staticmethod
    def _metrics(key: str) -> dict:
        import json

        import yaml

        root = Path(__file__).resolve().parents[1]
        entries = yaml.safe_load((root / "configs" / "registry.yaml").read_text())["entries"]
        return json.loads((root / entries[key] / "metrics.json").read_text())

    def test_quoted_metrics_match_the_promoted_runs(self):
        from driveflow.viz.views.system_diagram import _DIAGRAM_COMPONENT_DETAILS as d

        pc_cls = self._metrics("dc_motor/pc/classifier")
        assert f"{100 * pc_cls['test_accuracy']:.1f} %" in d["Classifier (CNN)"]
        assert f"{len(pc_cls['classes'])} classes" in d["Classifier (CNN)"]
        assert f"{self._metrics('vsc_dpc/pc/regressor')['test_rmse']:.3f}" in d["Regressor (LSTM/TCN)"]
        assert f"{100 * self._metrics('dc_motor/rpi5/classifier')['test_accuracy']:.1f} %" in d["Raspberry Pi 5"]
        assert f"{self._metrics('vsc_dpc/rpi5/regressor')['test_rmse']:.3f}" in d["Raspberry Pi 5"]
        esp_cls = self._metrics("dc_motor/esp32/classifier")
        assert f"{100 * esp_cls['test_accuracy']:.1f} %" in d["ESP32"]
        assert f"{len(esp_cls['classes'])}-class" in d["ESP32"]
        assert f"{self._metrics('vsc_dpc/esp32/regressor')['test_rmse']:.2f}" in d["ESP32"]

    def test_every_registered_environment_is_named(self):
        import driveflow.envs  # noqa: F401  (registers the SPADE/* IDs)
        from driveflow.envs.registry import ENVIRONMENTS
        from driveflow.viz.views.system_diagram import _DIAGRAM_COMPONENT_DETAILS as d

        for env_id in ENVIRONMENTS:
            if env_id.endswith("Surrogate-v0") and env_id != "SPADE/BuckSurrogate-v0":
                continue  # per-PLECS-sweep surrogates (A1), covered by SPADE/<Profile>Surrogate-v0
            assert f"`{env_id}`" in d["Gymnasium environments"], env_id
