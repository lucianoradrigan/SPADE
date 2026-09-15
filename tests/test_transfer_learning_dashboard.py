"""Tests for the Transfer Learning dashboard tab (Phase 3). AppTest exercises the landing page and
the full Train flow (data generation -> merge -> window -> fit -> validate, all in-memory, no
filesystem writes) against the REAL project registry (dc_motor/pc/classifier is genuinely
promoted there, same as tests/test_ai_dashboard.py already relies on). The Promote step (writes
config.yaml/weights/metrics.json to disk AND to configs/registry.yaml) is deliberately NOT
exercised through the live button -- that would promote a low-quality tiny test model into the
REAL production registry. It's tested directly instead, against a tmp_path registry, the same way
tests/test_registry.py tests driveflow.ai.registry.promote() itself.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from driveflow.ai.registry import promote, resolve
from driveflow.ai.transfer.pipeline import FineTuneStrategy, TransferLearningPipeline
from driveflow.viz.transfer_learning_dashboard import _next_run_dir, _prepare_windows, _save_finetuned_artifacts

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "dashboard.py")


class TestLandingPage:
    def test_tl_card_is_present(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        assert at.button(key="enter_phase_TL").label == "Enter Fase TL →"


class TestEnterTlPhase:
    def test_renders_without_exceptions_for_both_domains(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=60)
        at.run()
        at.button(key="enter_phase_TL").click().run()
        assert not at.exception
        at.sidebar.selectbox(key="tl_domain").set_value("vsc_dpc").run()
        assert not at.exception


class TestTrainFlow:
    """No filesystem writes -- see module docstring. Small scenario count/epochs for speed."""

    def test_dc_motor_classifier_trains_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_TL").click().run()
        at.sidebar.slider(key="tl_n_scenarios").set_value(4).run()
        at.sidebar.slider(key="tl_epochs").set_value(2).run()
        at.button(key="tl_train_button").click().run()
        assert not at.exception
        success_texts = " ".join(el.value for el in at.success)
        assert "Training complete" in success_texts

    def test_vsc_dpc_regressor_trains_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_TL").click().run()
        at.sidebar.selectbox(key="tl_domain").set_value("vsc_dpc").run()
        at.sidebar.slider(key="tl_n_scenarios").set_value(4).run()
        at.sidebar.slider(key="tl_epochs").set_value(2).run()
        at.button(key="tl_train_button").click().run()
        assert not at.exception
        success_texts = " ".join(el.value for el in at.success)
        assert "Training complete" in success_texts

    def test_discriminative_lr_strategy_trains_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_TL").click().run()
        at.sidebar.radio(key="tl_strategy").set_value(FineTuneStrategy.DISCRIMINATIVE_LR).run()
        at.sidebar.slider(key="tl_n_scenarios").set_value(4).run()
        at.sidebar.slider(key="tl_epochs").set_value(2).run()
        at.button(key="tl_train_button").click().run()
        assert not at.exception

    def test_adapter_strategy_trains_without_exceptions(self):
        at = AppTest.from_file(DASHBOARD_PATH, default_timeout=120)
        at.run()
        at.button(key="enter_phase_TL").click().run()
        at.sidebar.radio(key="tl_strategy").set_value(FineTuneStrategy.ADAPTER).run()
        at.sidebar.slider(key="tl_n_scenarios").set_value(4).run()
        at.sidebar.slider(key="tl_epochs").set_value(2).run()
        at.button(key="tl_train_button").click().run()
        assert not at.exception


class TestPrepareWindowsChannelGuard:
    """Regression test for the real bug found by testing the actual Train button: windowing must
    use the BASE MODEL's own channel list, not re-discover channels from the new (possibly
    smaller/less varied) sample -- a fresh small dataset can legitimately have fewer live
    channels, and feeding a mismatched channel count into model.fit() raises a shape error."""

    def test_missing_channel_is_reported_clearly_instead_of_crashing_later(self):
        import pandas as pd

        df = pd.DataFrame({"acc_x": [1.0] * 20, "label": ["normal"] * 10 + ["outer_race"] * 10})
        windows, error = _prepare_windows(
            "classifier", df, input_window=8, horizon=None,
            channels=["acc_x", "current_r"], classes=["normal", "outer_race"],
        )
        assert windows is None
        assert "current_r" in error


class TestPromoteAndSaveArtifacts:
    """Exercises the exact save+promote path _render_fase_tl's own "Promote to registry" button
    calls, against a tmp_path registry -- never the real one (see module docstring)."""

    def test_finetuned_run_saves_and_promotes(self, tmp_path):
        from driveflow.models.classifiers.builder import build_classifier
        from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig

        # A minimal stand-in "base" config/run, mirroring what ModelLoader.load_latest() would
        # hand _render_fase_tl in production.
        base_config_dir = tmp_path / "configs" / "classifiers" / "pc_server" / "2026-09-01_run01"
        base_config_dir.mkdir(parents=True)
        config_yaml = base_config_dir / "config.yaml"
        config_yaml.write_text(
            "domain: dc_motor\ntier: pc\ninput_window: 8\nnum_classes: 2\n"
            "blocks:\n  - type: conv1d\n    filters: 4\n    kernel_size: 3\ndense_units: []\ndropout: 0.0\n"
        )

        config = ClassifierConfig(domain="dc_motor", tier="pc", input_window=8, num_classes=2, blocks=(ConvBlockConfig(filters=4, kernel_size=3),))
        model = build_classifier(config, n_channels=2)
        pipeline = TransferLearningPipeline.from_loaded_model(model, strategy=FineTuneStrategy.FEATURE_EXTRACTOR)

        run_dir = _next_run_dir(base_config_dir.parent.parent / f"{base_config_dir.parent.name}_tl.yaml")
        assert run_dir.parent.name == "pc_server_tl"

        _save_finetuned_artifacts(run_dir, config_yaml, pipeline, metrics={"domain": "dc_motor", "tier": "pc", "channels": ["a", "b"], "classes": ["normal", "outer_race"], "test_accuracy": 0.5})
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "model.weights.h5").exists()
        assert (run_dir / "metrics.json").exists()

        registry_path = tmp_path / "registry.yaml"
        entry = promote(run_dir, registry_path=registry_path)
        assert entry.domain == "dc_motor"
        assert entry.tier == "pc"
        assert entry.block == "classifier"
        assert resolve("dc_motor", "pc", "classifier", registry_path=registry_path) == run_dir.resolve()
