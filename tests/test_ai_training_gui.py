"""GUI: AI Diagnostics → Train a model (several fault classes, simulation-page options, a worker job,
the candidate selectable in Evaluate) and the Transfer Learning base-model sources and transfer modes."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.ai import custom_training as ct
from driveflow.ai.registry import DEFAULT_REGISTRY_PATH
from driveflow.ai.transfer.pipeline import FineTuneStrategy
from driveflow.jobs.worker import Worker
from driveflow.runs.store import workspace_root

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open(page):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page(f"pages/{page}.py").run()
    assert not at.exception
    return at


def _text(at):
    return " ".join(str(e.value) for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind))


class TestTrainTab:
    @pytest.mark.parametrize("domain", ["dc_motor", "vsc_dpc"])
    def test_renders_with_a_run_summary(self, domain):
        at = _open("ai_diagnostics")
        at.selectbox(key="ia_domain").set_value(domain).run()
        assert not at.exception
        assert "runs" in _text(at) and at.button(key=f"tr_{domain}_start")

    def test_more_simulation_options_than_before(self):
        at = _open("ai_diagnostics")
        for key in ("ia_controller", "ia_fault_type"):
            assert at.selectbox(key=key)
        assert at.radio(key="ia_control_mode")
        at.radio(key="ia_control_mode").set_value("Torque (τ_ref)").run()
        at.selectbox(key="ia_controller").set_value("MPC").run()
        assert not at.exception

    def test_one_class_is_refused_before_submitting(self):
        at = _open("ai_diagnostics")
        at.multiselect(key="tr_dc_classes").set_value(["outer_race"]).run()
        assert any("at least two classes" in e.value for e in at.error)
        assert at.button(key="tr_dc_motor_start").disabled

    def test_train_then_evaluate_the_candidate(self, tmp_path, monkeypatch):
        import driveflow.viz.views.ai_diagnostics as page

        monkeypatch.setattr(ct, "CUSTOM_ROOT", tmp_path / "models")
        monkeypatch.setattr(ct, "DATASET_ROOT", tmp_path / "data")
        monkeypatch.setattr(page, "list_classifier_runs", lambda domain: ct.list_model_runs(domain, root=tmp_path / "models"))
        at = _open("ai_diagnostics")
        at.multiselect(key="tr_dc_classes").set_value(["healthy", "outer_race", "inner_race"]).run()
        at.number_input(key="tr_dc_speed_n").set_value(1).run()
        at.multiselect(key="tr_dc_scales").set_value([1.0]).run()
        at.toggle(key="tr_dc_drift_psi").set_value(False).run()
        at.toggle(key="tr_dc_drift_j").set_value(False).run()
        at.number_input(key="tr_dc_motor_dur").set_value(0.3).run()
        at.number_input(key="tr_dc_motor_skip").set_value(0.05).run()
        at.number_input(key="tr_dc_motor_reps").set_value(3).run()
        at.selectbox(key="tr_dc_motor_preset").set_value("esp32_dscnn").run()
        at.select_slider(key="tr_dc_motor_window").set_value(512).run()
        at.number_input(key="tr_dc_motor_epochs").set_value(1).run()
        at.text_input(key="tr_dc_motor_name").set_value("gui_test").run()
        at.slider(key="tr_dc_motor_workers").set_value(1).run()
        assert "9 runs" in _text(at)
        at.button(key="tr_dc_motor_start").click().run()
        assert any("Queued as job" in s.value for s in at.success)
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
        at.button(key="tr_dc_motor_refresh").click().run()
        assert not at.exception
        assert "CANDIDATE — NOT PROMOTED" in " ".join(str(h.body) for h in at.get("html"))
        run_id = at.selectbox(key="tr_dc_motor_run").value
        at.button(key=f"tr_use_{run_id}").click().run()
        assert "gui_test" in str(at.selectbox(key="ia_model").value)
        at.button(key="ia_generate").click().run()
        assert not at.exception
        assert "CANDIDATE — NOT PROMOTED" in " ".join(str(h.body) for h in at.get("html"))


class TestTransferLearningSources:
    def test_import_lists_the_supported_formats(self):
        at = _open("transfer_learning")
        at.radio(key="tl_base_source").set_value("Import a model file").run()
        assert not at.exception
        table = next(d.value for d in at.dataframe if "Format" in getattr(d.value, "columns", []))
        assert {"Keras v3 model", "ONNX / PyTorch", "TensorFlow Lite"} <= set(table["Format"])

    def test_another_run_with_a_new_class_only_offers_a_new_head(self):
        registry_before = DEFAULT_REGISTRY_PATH.read_text()
        at = _open("transfer_learning")
        at.radio(key="tl_base_source").set_value("Another model on this machine").run()
        base = next(o for o in at.selectbox(key="tl_base_run").options if "pc_server/2026-10-18_run01" in str(o))
        at.selectbox(key="tl_base_run").set_value(base).run()
        at.multiselect(key="tl_faults").set_value(["healthy", "outer_race", "cage"]).run()
        assert "New class(es) for this base model: cage" in _text(at)
        at.slider(key="tl_n_scenarios").set_value(9).run()
        at.slider(key="tl_duration").set_value(0.3).run()
        at.button(key="tl_prepare_button").click().run()
        assert not at.exception and at.radio(key="tl_step").value == "2 · Setup"
        from driveflow.viz.views.transfer_learning import STRATEGY_LABELS

        assert list(at.radio(key="tl_strategy").options) == [STRATEGY_LABELS[FineTuneStrategy.NEW_HEAD]]
        assert at.radio(key="tl_strategy").value is FineTuneStrategy.NEW_HEAD
        at.slider(key="tl_epochs").set_value(1).run()
        at.button(key="tl_train_button").click().run()
        assert not at.exception and at.radio(key="tl_step").value == "3 · Validate"
        at.radio(key="tl_step").set_value("4 · Promote").run()
        assert any("acceptance_floors" in e.value or "min_macro_f1" in e.value or "min_per_class_recall" in e.value for e in at.error)
        assert at.button(key="tl_promote_button").disabled
        assert DEFAULT_REGISTRY_PATH.read_text() == registry_before

    def test_full_transfer_trains_on_the_promoted_model(self):
        at = _open("transfer_learning")
        at.multiselect(key="tl_faults").set_value(["healthy", "outer_race", "inner_race"]).run()
        at.slider(key="tl_n_scenarios").set_value(9).run()
        at.slider(key="tl_duration").set_value(0.3).run()
        at.button(key="tl_prepare_button").click().run()
        at.radio(key="tl_strategy").set_value(FineTuneStrategy.FULL_FINETUNE).run()
        assert "full transfer learning" in _text(at)
        at.slider(key="tl_epochs").set_value(1).run()
        at.button(key="tl_train_button").click().run()
        assert not at.exception and at.radio(key="tl_step").value == "3 · Validate"
