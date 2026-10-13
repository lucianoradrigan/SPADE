"""Tests for the Transfer Learning wizard page (driveflow.viz.views.transfer_learning).

AppTest drives the real flow -- Prepare dataset -> Train candidate -> Validate -> Promote step --
against the REAL registry for reading only. "Save as candidate" / "Promote" are never clicked here
(they would write under configs/); the save/promote path is covered against tmp_path in
tests/test_ai_audit_regressions.py (T-AI-06) and below via driveflow.ai.transfer.workflow.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from driveflow.ai.registry import DEFAULT_REGISTRY_PATH
from driveflow.ai.transfer.pipeline import FineTuneStrategy

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open(domain="dc_motor"):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/transfer_learning.py").run()
    if domain != "dc_motor":
        at.selectbox(key="tl_domain").set_value(domain).run()
    return at


def _prepare_and_train(at, n=9, epochs=2, strategy=None):
    at.slider(key="tl_n_scenarios").set_value(n).run()
    at.slider(key="tl_duration").set_value(0.3).run()  # short runs keep the test fast
    at.button(key="tl_prepare_button").click().run()
    assert not at.exception
    assert at.radio(key="tl_step").value == "2 · Setup"
    if strategy is not None:
        at.radio(key="tl_strategy").set_value(strategy).run()
    at.slider(key="tl_epochs").set_value(epochs).run()
    at.button(key="tl_train_button").click().run()
    assert not at.exception
    assert at.radio(key="tl_step").value == "3 · Validate"


class TestWizard:
    def test_renders_for_both_domains(self):
        at = _open()
        assert not at.exception
        at.selectbox(key="tl_domain").set_value("vsc_dpc").run()
        assert not at.exception

    @pytest.mark.parametrize("domain", ["dc_motor", "vsc_dpc"])
    def test_full_flow_ends_blocked_until_a_reviewer_approves(self, domain):
        registry_before = DEFAULT_REGISTRY_PATH.read_text()
        at = _open(domain)
        _prepare_and_train(at)
        at.radio(key="tl_step").set_value("4 · Promote").run()
        assert not at.exception
        assert at.button(key="tl_promote_button").disabled
        assert any("human_approval: PENDING" in e.value for e in at.error)
        assert DEFAULT_REGISTRY_PATH.read_text() == registry_before  # nothing written by navigating

    @pytest.mark.parametrize("strategy", [FineTuneStrategy.DISCRIMINATIVE_LR, FineTuneStrategy.ADAPTER])
    def test_other_strategies_train(self, strategy):
        at = _open()
        _prepare_and_train(at, strategy=strategy)

    def test_too_few_acquisitions_fail_explicitly_instead_of_leaking(self):
        at = _open()
        at.slider(key="tl_n_scenarios").set_value(3).run()  # 1 per class -> cannot fill val/test per class
        at.button(key="tl_prepare_button").click().run()
        assert not at.exception
        assert any("Dataset not prepared" in e.value for e in at.error)
        assert at.radio(key="tl_step").value == "1 · Data"

    def test_adapter_is_labelled_as_not_lora(self):
        at = _open()
        at.slider(key="tl_n_scenarios").set_value(9).run()
        at.button(key="tl_prepare_button").click().run()
        labels = at.radio(key="tl_strategy").options
        assert any("NOT LoRA" in str(o) for o in labels)


class TestMissingChannelGuard:
    def test_missing_channel_is_reported_clearly(self):
        import pandas as pd

        from driveflow.ai.transfer import workflow as wf

        df = pd.DataFrame({"timestamp_s": [i * 1e-4 for i in range(1, 41)], "acc_x": [1.0] * 40, "label": ["normal"] * 40, "source_file": "a"})
        with pytest.raises(ValueError, match="current_r"):
            wf.prepare_dataset("dc_motor", "classifier", {"channels": ["acc_x", "current_r"], "classes": ["normal"]}, 8, None, df)
