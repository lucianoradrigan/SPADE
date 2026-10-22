"""Full vs. feature transfer learning, a new output head for a new class vocabulary, and importing a
third-party Keras model as the base (driveflow.ai.transfer.pipeline / workflow / external)."""

import json

import keras
import numpy as np
import pandas as pd
import pytest
import yaml

from driveflow.ai import registry
from driveflow.ai.transfer import external as ext
from driveflow.ai.transfer import workflow as wf
from driveflow.ai.transfer.loader import ModelLoader
from driveflow.ai.transfer.pipeline import FineTuneStrategy as S
from driveflow.ai.transfer.pipeline import with_new_head
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import load_classifier_config

WIN = 64
FREQ = {"normal": 40.0, "outer_race": 90.0, "inner_race": 160.0, "unbalance": 260.0}


def _data(labels, n_runs=4, n=WIN * 6, seed=0):
    """Two channels whose dominant frequency depends on the class: separable in a few epochs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1e-4
    frames = []
    for lab in labels:
        for k in range(n_runs):
            sig = np.sin(2 * np.pi * FREQ[lab] * 10 * t + rng.uniform(0, 6)) + 0.1 * rng.standard_normal(n)
            frames.append(pd.DataFrame({"timestamp_s": t + 1e-4, "current_r": sig, "acc_x": np.roll(sig, 3) + 0.1 * rng.standard_normal(n),
                                        "label": lab, "source_file": f"{lab}_{k}"}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def spade_base(tmp_path):
    """A tiny SPADE builder run: 2 classes, 2 channels, 64-sample window."""
    run_dir = tmp_path / "configs" / "classifiers" / "tiny" / "2026-01-01_run01"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"domain": "dc_motor", "tier": "pc", "input_window": WIN, "num_classes": 2,
                                                         "blocks": [{"type": "conv1d", "filters": 8, "kernel_size": 5}], "dense_units": [8], "dropout": 0.0}))
    model = build_classifier(load_classifier_config(run_dir / "config.yaml"), n_channels=2)
    model.save_weights(run_dir / "model.weights.h5")
    (run_dir / "metrics.json").write_text(json.dumps({"kind": "classifier", "domain": "dc_motor", "tier": "pc", "channels": ["current_r", "acc_x"],
                                                      "classes": ["normal", "outer_race"], "test_macro_f1": 0.5}))
    return ModelLoader(registry.DEFAULT_REGISTRY_PATH, "dc_motor", "classifier").load_by_generation(run_dir)


def _prepare(base, labels, **kw):
    return wf.prepare_dataset("dc_motor", "classifier", base["metrics"], WIN, None, _data(labels), seed=0, val_frac=0.25, test_frac=0.25, **kw)


class TestNewHead:
    def test_new_head_matches_the_builder_architecture_and_freezes_the_features(self, spade_base, tmp_path):
        new = with_new_head(spade_base["model"], 3, unfreeze_last=1)
        raw = yaml.safe_load((spade_base["run_dir"] / "config.yaml").read_text())
        (tmp_path / "k3.yaml").write_text(yaml.safe_dump({**raw, "num_classes": 3}))
        ref = build_classifier(load_classifier_config(tmp_path / "k3.yaml"), n_channels=2)
        assert new.output_shape == (None, 3) and new.layers[-1].name == "head"
        assert [layer.name for layer in new.layers] == [layer.name for layer in ref.layers]
        weighted = [layer for layer in new.layers if layer.weights]
        assert [layer.trainable for layer in weighted] == [False] * (len(weighted) - 2) + [True, True]

    def test_new_vocabulary_only_allows_a_new_head(self, spade_base):
        ds = _prepare(spade_base, ["normal", "outer_race", "unbalance"], target_classes=["normal", "outer_race", "unbalance"])
        assert ds.new_vocabulary and ds.classes == ["normal", "outer_race", "unbalance"]
        for strategy in (S.FULL_FINETUNE, S.DISCRIMINATIVE_LR, S.FEATURE_EXTRACTOR, S.ADAPTER):
            with pytest.raises(ValueError, match="new output head"):
                wf.finetune(spade_base, ds, {"strategy": strategy, "epochs": 1})

    def test_new_head_is_gated_on_an_absolute_floor_and_reloads(self, spade_base, tmp_path):
        ds = _prepare(spade_base, ["normal", "outer_race", "unbalance"], target_classes=["normal", "outer_race", "unbalance"])
        res = wf.finetune(spade_base, ds, {"strategy": S.NEW_HEAD, "epochs": 2, "unfreeze_last": 1, "learning_rate": 1e-2})
        checks = {c["name"]: c["status"] for c in res.report["checks"]}
        assert checks["macro_f1_vs_base"] == "NOT_APPLICABLE" and checks["min_macro_f1"] == "INCOMPLETE"  # no floor given: blocks
        assert res.test_evaluation["base_comparable"] is False and res.base_unchanged
        decision = wf.gate_decision(ds, res, {"approved": True, "approved_by": "reviewer"})
        items = {i.name: i.status for i in decision.items}
        assert items["independent_test_evaluated"] == "PASS" and items["validation:min_macro_f1"] == "INCOMPLETE" and not decision.approved
        run_dir = wf.save_candidate(res, spade_base, decision, run_dir=tmp_path / "cand")
        assert yaml.safe_load((run_dir / "config.yaml").read_text())["num_classes"] == 3
        model, config, metrics = registry.load_run_model(run_dir, "classifier")
        X = ds.windows["test"]["X"][:3]
        assert np.allclose(model.predict(X, verbose=0), res.pipeline.model.predict(X, verbose=0), atol=1e-5)
        assert metrics["classes"] == ["normal", "outer_race", "unbalance"]

    def test_a_floor_makes_the_check_decidable(self, spade_base):
        ds = _prepare(spade_base, ["normal", "outer_race", "unbalance"], target_classes=["normal", "outer_race", "unbalance"])
        res = wf.finetune(spade_base, ds, {"strategy": S.NEW_HEAD, "epochs": 1}, thresholds={"max_macro_f1_drop": 0.02, "min_macro_f1": 0.0})
        assert {c["name"]: c["status"] for c in res.report["checks"]}["min_macro_f1"] == "PASS"


class TestFullTransfer:
    def test_full_finetune_trains_every_layer_on_the_base_vocabulary(self, spade_base):
        ds = _prepare(spade_base, ["normal", "outer_race"], target_classes=["normal", "outer_race"])
        assert not ds.new_vocabulary
        before = spade_base["model"].layers[1].get_weights()[0].copy()
        res = wf.finetune(spade_base, ds, {"strategy": S.FULL_FINETUNE, "epochs": 1, "learning_rate": 1e-2})
        assert all(layer.trainable for layer in res.pipeline.model.layers)
        assert not np.allclose(res.pipeline.model.layers[1].get_weights()[0], before)  # first conv moved
        assert np.allclose(spade_base["model"].layers[1].get_weights()[0], before)  # base untouched
        assert res.lineage["transfer_mode"] == "full" and "macro_f1_vs_base" in {c["name"] for c in res.report["checks"]}
        assert "base_comparable" not in res.test_evaluation


def _keras_file(tmp_path, name="third.keras", lam=False, act="softmax", n_out=2):
    inp = keras.Input((WIN, 2))
    x = keras.layers.Lambda(lambda t: t * 2.0)(inp) if lam else inp
    x = keras.layers.Conv1D(4, 5)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(n_out, activation=act)(x)
    path = tmp_path / name
    keras.Model(inp, out).save(path)
    return path


class TestExternal:
    def test_supported_formats_are_listed(self):
        fmts = {f["Format"]: f["Status"] for f in ext.SUPPORTED_FORMATS}
        assert fmts["Keras v3 model"] == "Supported" and fmts["ONNX / PyTorch"] == "Not supported" and fmts["TensorFlow Lite"] == "Not supported"

    def test_import_and_feature_transfer_from_a_third_party_model(self, tmp_path):
        p = _keras_file(tmp_path)
        run_dir = ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r", "acc_x"], ["normal", "outer_race"], root=tmp_path / "ext")
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        assert cfg["model_format"] == "keras" and cfg["input_window"] == WIN and (run_dir / "model.keras").exists()
        assert json.loads((run_dir / "metrics.json").read_text())["origin"] == "EXTERNAL"
        base = ModelLoader(registry.DEFAULT_REGISTRY_PATH, "dc_motor", "classifier").load_by_generation(run_dir)
        ds = _prepare(base, ["normal", "outer_race", "inner_race"], target_classes=["normal", "outer_race", "inner_race"])
        res = wf.finetune(base, ds, {"strategy": S.NEW_HEAD, "epochs": 1}, thresholds={"min_macro_f1": 0.0})
        assert res.lineage["base_weights_sha256"] and res.base_unchanged
        cand = wf.save_candidate(res, base, wf.gate_decision(ds, res, None), run_dir=tmp_path / "cand")
        model, config, metrics = registry.load_run_model(cand, "classifier")
        assert isinstance(config, registry.KerasRunConfig) and config.num_classes == 3 and model.output_shape == (None, 3)

    def test_an_imported_model_can_be_promoted_and_loaded_from_the_registry(self, tmp_path):
        p = _keras_file(tmp_path)
        run_dir = ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r", "acc_x"], ["normal", "outer_race"], root=tmp_path / "ext")
        reg = tmp_path / "registry.yaml"
        entry = registry.promote(run_dir, registry_path=reg, promoted_by="test")
        assert entry.block == "classifier"
        model, config, _ = registry.load_promoted_model("dc_motor", "pc", "classifier", registry_path=reg)
        assert config.input_window == WIN

    @pytest.mark.parametrize("kwargs, message", [
        ({"lam": True}, "Lambda"),
        ({"act": "linear"}, "softmax"),
    ])
    def test_unsafe_or_unsupported_models_are_refused(self, tmp_path, kwargs, message):
        p = _keras_file(tmp_path, **kwargs)
        with pytest.raises(ext.ExternalModelError, match=message):
            ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r", "acc_x"], ["a", "b"], root=tmp_path / "ext")

    def test_declarations_must_match_the_model(self, tmp_path):
        p = _keras_file(tmp_path)
        with pytest.raises(ext.ExternalModelError, match="channel"):
            ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r"], ["a", "b"], root=tmp_path / "ext")
        with pytest.raises(ext.ExternalModelError, match="output unit"):
            ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r", "acc_x"], ["a", "b", "c"], root=tmp_path / "ext")

    def test_weights_only_and_other_formats_are_refused(self, tmp_path):
        inp = keras.Input((WIN, 2))
        m = keras.Model(inp, keras.layers.Dense(2, activation="softmax")(keras.layers.GlobalAveragePooling1D()(inp)))
        w = tmp_path / "w.weights.h5"
        m.save_weights(w)
        with pytest.raises(ext.ExternalModelError, match="no model architecture"):
            ext.import_model(w.read_bytes(), "w.h5", "dc_motor", ["current_r", "acc_x"], ["a", "b"], root=tmp_path / "ext")
        with pytest.raises(ext.ExternalModelError, match="unsupported file type"):
            ext.import_model(b"onnx", "model.onnx", "dc_motor", ["current_r", "acc_x"], ["a", "b"], root=tmp_path / "ext")


def test_model_registry_lists_imported_and_gui_trained_runs(tmp_path, monkeypatch):
    from driveflow.viz.adapters import registry_adapter as ra

    root = tmp_path / "classifiers"
    p = _keras_file(tmp_path)
    imported = ext.import_model(p.read_bytes(), p.name, "dc_motor", ["current_r", "acc_x"], ["normal", "outer_race"], root=root / "external")
    custom = root / "custom" / "mine" / "2026-01-01_run01"
    custom.mkdir(parents=True)
    for f in ("config.yaml", "model.weights.h5"):
        (custom / f).write_bytes((imported / f).read_bytes())
    (custom / "metrics.json").write_text(json.dumps({"kind": "classifier", "domain": "dc_motor", "tier": "pc", "channels": ["current_r", "acc_x"],
                                                    "classes": ["normal", "outer_race"], "trained_from": "ai_diagnostics.custom_training"}))
    monkeypatch.setattr(ra, "CONFIG_ROOTS", (root,))
    rows = {r["run"]: r for r in ra.list_runs(tmp_path / "registry.yaml")}
    assert rows[f"external/{imported.name}"]["origin"] == "imported third-party model" and rows[f"external/{imported.name}"]["block"] == "classifier"
    assert rows["mine/2026-01-01_run01"]["origin"] == "trained in AI Diagnostics"
