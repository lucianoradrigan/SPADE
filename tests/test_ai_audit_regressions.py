"""Regression tests for the AI audit findings (README_FOR_CLAUDE.md Sec. 2 / Sec. 6, T-AI-01..T-AI-09).

Each class pins one finding that was REPRODUCED on the unmodified code (docs/audit/
repro_h0_before_fix_output.txt) so it cannot come back. Everything here is in-memory / tmp_path:
no test touches the real configs/registry.yaml.
"""

import json
import shutil
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest

from driveflow.ai.promotion import PromotionGate, is_transfer_learning_run
from driveflow.ai.provenance import weights_fingerprint
from driveflow.ai.registry import PromotionRejected, _load_index, promote, registry_history, resolve, rollback
from driveflow.ai.transfer.data_merger import DataMerger, InsufficientDataError
from driveflow.ai.transfer.dataset_validation import validate_training_dataset
from driveflow.ai.transfer.pipeline import FineTuneStrategy, TransferLearningPipeline
from driveflow.ai.transfer.validator import CheckStatus, ModelValidator
from driveflow.ai.transfer import workflow as wf
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig
from driveflow.models.common.acquisitions import check_acquisitions
from driveflow.models.common.splits import InsufficientGroupsError, grouped_train_val_test_split
from driveflow.models.common.windowing import build_classification_windows, build_direct_forecast_windows
from driveflow.monitoring.agents.agent_gateway import Alert
from driveflow.monitoring.agents.agent_server import ServerAgent
from driveflow.monitoring.motor_status import derive_motor_view
from driveflow.viz.dpc_upload_validation import validate_dc_motor_upload

FS = 10_000.0


def _acq(name, n, label="normal", fs=FS, t0=0.0):
    t = t0 + np.arange(1, n + 1) / fs
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    return pd.DataFrame({
        "timestamp_s": t,
        "current_r": 10 * np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.5, n),
        "acc_x": np.cos(2 * np.pi * 120 * t) + rng.normal(0, 0.1, n),
        "label": label,
        "source_file": name,
    })


def _tiny_classifier(n_channels=2, window=16, n_classes=2, seed=0):
    keras.utils.set_random_seed(seed)
    cfg = ClassifierConfig(domain="dc_motor", tier="pc", input_window=window, num_classes=n_classes, blocks=(ConvBlockConfig(filters=4, kernel_size=3),))
    return build_classifier(cfg, n_channels=n_channels)


# --------------------------------------------------------------------------------------- T-AI-01
class TestAI01TemporalMixing:
    def test_merged_groups_keep_origin_monotonic_time_and_never_mix_sources(self):
        sim = pd.concat([_acq("run_a", 400), _acq("run_b", 400), _acq("run_c", 400)], ignore_index=True)
        ext = pd.concat([_acq("run_a", 400), _acq("ext_d", 400)], ignore_index=True)  # colliding name on purpose
        merger = DataMerger(sim, ext, mix_ratio=0.6, window_samples=50, expected_fs_hz=FS)
        merged = merger.merge(seed=0)
        assert merged.groupby("source_file")["_data_source"].nunique().max() == 1, "a sim and an ext recording were merged into one group"
        prefix = merged["_data_source"].map({"simulated": "sim::", "external": "ext::"})
        assert all(sf.startswith(p) for sf, p in zip(merged["source_file"], prefix)), "acquisition ID lost its origin prefix"
        for _, g in merged.groupby("source_file"):
            dt = np.diff(g["timestamp_s"].to_numpy())
            assert np.all(dt > 0)
            np.testing.assert_allclose(dt, 1 / FS, rtol=1e-6)

    def test_windows_never_cross_groups_or_gaps(self):
        a = _acq("a", 300)
        a = a.drop(index=range(100, 110))  # a 10-sample hole inside the recording
        df = pd.concat([a, _acq("b", 300, "outer_race")], ignore_index=True)
        X, y, groups = build_classification_windows(df, ["normal", "outer_race"], ["current_r"], 50)
        # Recording "a" now has segments of 100 and 190 rows -> 2 + 3 windows, never one across the hole.
        assert (groups == "a").sum() == 5
        assert (groups == "b").sum() == 6
        X2, Y2, g2 = build_direct_forecast_windows(df, ["current_r"], 40, 20)
        assert (g2 == "a").sum() == 1 + 3  # 100 -> 1 segment of 60, 190 -> 3

    def test_different_sampling_rate_is_rejected_not_resampled(self):
        ext = pd.concat([_acq("ok", 200), _acq("slow", 200, fs=5_000.0)], ignore_index=True)
        merger = DataMerger(_acq("s", 400), ext, mix_ratio=0.5, window_samples=50, expected_fs_hz=FS)
        merger.merge()
        excluded = {e["group"]: e["reason"] for e in merger.excluded_groups}
        assert "ext::slow" in excluded and "sampling rate" in excluded["ext::slow"]

    def test_out_of_order_rows_are_excluded_not_sorted(self):
        shuffled = _acq("shuffled", 200).sample(frac=1.0, random_state=0)
        checks = check_acquisitions(shuffled)
        assert checks[0].ok is False and "not strictly increasing" in checks[0].reason

    @pytest.mark.parametrize("ratio,expected_sources", [(1.0, {"simulated"}), (0.0, {"external"})])
    def test_extreme_ratios(self, ratio, expected_sources):
        merged = DataMerger(_acq("s", 400), _acq("e", 400), mix_ratio=ratio, window_samples=50).merge()
        assert set(merged["_data_source"]) == expected_sources

    def test_insufficient_external_dataset_fails_explicitly(self):
        tiny_ext = _acq("e", 10)  # shorter than one window
        with pytest.raises(InsufficientDataError):
            DataMerger(_acq("s", 400), tiny_ext, mix_ratio=0.5, window_samples=50).merge()

    def test_mix_ratio_is_measured_in_windows_and_reported(self):
        sim = pd.concat([_acq(f"s{i}", 500) for i in range(10)], ignore_index=True)
        ext = pd.concat([_acq(f"e{i}", 500) for i in range(10)], ignore_index=True)
        merger = DataMerger(sim, ext, mix_ratio=0.7, window_samples=50)
        merger.merge(seed=1)
        stats = merger.get_statistics()
        assert stats["mix_unit"] == "windows"
        assert abs(stats["realized_mix_ratio"] - 0.7) <= 0.1


# --------------------------------------------------------------------------------------- T-AI-02
class TestAI02MetadataPreserved:
    def test_training_validator_keeps_label_source_and_time(self):
        df = pd.concat([_acq("r1", 100, "outer_race"), _acq("r2", 100)], ignore_index=True)
        report = validate_training_dataset(df, "dc_motor", "classification", ["current_r", "acc_x"], ["normal", "outer_race"], expected_fs_hz=FS)
        assert report.ok
        for col in ("label", "source_file", "timestamp_s"):
            assert col in report.df.columns
        assert set(report.df["source_file"]) == {"ext::r1", "ext::r2"}
        assert report.sha256

    def test_file_without_labels_is_inference_only(self):
        df = _acq("r1", 100).drop(columns=["label"])
        report = validate_training_dataset(df, "dc_motor", "classification", ["current_r"], ["normal"])
        assert not report.ok
        assert report.usable_for == ["inference"]
        assert any("inference" in text for level, text in report.messages if level == "error")

    def test_unknown_labels_block_training(self):
        report = validate_training_dataset(_acq("r1", 100, "cage"), "dc_motor", "classification", ["current_r"], ["normal", "outer_race"])
        assert not report.ok and report.unknown_labels == ["cage"]

    def test_missing_base_channel_blocks_training(self):
        report = validate_training_dataset(_acq("r1", 100), "dc_motor", "classification", ["current_r", "torque_nm"], ["normal"])
        assert not report.ok and report.missing_channels == ["torque_nm"]

    def test_per_channel_plausibility_bounds(self):
        df = _acq("r1", 100)
        df.loc[10, "current_r"] = 5_000.0  # > 2x the simulated motor's 210 A limit, < generic 1e4
        report = validate_training_dataset(df, "dc_motor", "classification", ["current_r"], ["normal"])
        assert report.n_rows_out == 99
        assert any("current_r" in text for _, text in report.messages)

    def test_inference_validator_still_drops_metadata_by_design(self):
        clean, _, _ = validate_dc_motor_upload(_acq("r1", 20))
        assert "label" not in clean.columns  # documented inference-only contract


# --------------------------------------------------------------------------------------- T-AI-03
class TestAI03IndependentSplits:
    def test_three_way_group_split_is_disjoint(self):
        groups = np.repeat([f"g{i}" for i in range(10)], 4)
        m = grouped_train_val_test_split(groups, val_frac=0.2, test_frac=0.2, seed=0)
        assert m.is_disjoint()
        assert set(m.groups_val).isdisjoint(m.groups_test)
        assert not np.array_equal(np.sort(m.val_idx), np.sort(m.test_idx))
        assert min(len(m.train_idx), len(m.val_idx), len(m.test_idx)) > 0

    def test_too_few_groups_raises_instead_of_leaking(self):
        with pytest.raises(InsufficientGroupsError):
            grouped_train_val_test_split(np.array(["a", "a", "b", "b"]))

    def test_regression_dataset_has_distinct_val_and_test(self):
        df = pd.concat([_acq(f"s{i}", 400) for i in range(6)], ignore_index=True)
        ds = wf.prepare_dataset("vsc_dpc", "regressor", {"channels": ["current_r"]}, 40, 20, df, seed=0)
        assert ds.split["disjoint"]
        assert set(ds.split["group_hashes"]["val"]).isdisjoint(ds.split["group_hashes"]["test"])


# --------------------------------------------------------------------------------------- T-AI-04
class TestAI04BaseModelImmutable:
    def test_base_weights_hash_unchanged_after_repeated_finetuning(self):
        base = _tiny_classifier()
        fp = weights_fingerprint(base)
        rng = np.random.default_rng(0)
        data = {"X": rng.normal(size=(24, 16, 2)).astype("float32"), "y": rng.integers(0, 2, 24)}
        for strategy in FineTuneStrategy:
            pipe = TransferLearningPipeline.from_loaded_model(base, strategy=strategy, train_config={"learning_rate": 1e-2, "seed": 0})
            if strategy is FineTuneStrategy.DISCRIMINATIVE_LR:
                pipe.apply_discriminative_lr(1e-2, {})
            pipe.train(data, data, epochs=2, batch_size=8)
            assert pipe.model is not base
            assert pipe.verify_base_unchanged()
            assert weights_fingerprint(pipe.model) != fp
        assert weights_fingerprint(base) == fp

    def test_clone_false_keeps_legacy_in_place_behaviour_and_is_detected(self):
        base = _tiny_classifier()
        rng = np.random.default_rng(0)
        data = {"X": rng.normal(size=(16, 16, 2)).astype("float32"), "y": rng.integers(0, 2, 16)}
        pipe = TransferLearningPipeline.from_loaded_model(base, strategy=FineTuneStrategy.ADAPTER, train_config={"learning_rate": 1e-2}, clone=False)
        pipe.train(data, data, epochs=2, batch_size=8)
        assert pipe.verify_base_unchanged() is False


# --------------------------------------------------------------------------------------- T-AI-05
class _ConstRegressor:
    def __init__(self, value=0.0):
        self.value = value

    def predict(self, X, verbose=0):
        return np.full((len(X), 4, 2), self.value, dtype="float32")


class TestAI05TaskSpecificValidation:
    def _data(self):
        rng = np.random.default_rng(0)
        return {"X": rng.normal(size=(20, 8, 2)).astype("float32"), "y": rng.normal(size=(20, 4, 2)).astype("float32")}

    def test_nan_regressor_is_rejected(self):
        report = ModelValidator({}, {"max_rmse_increase": 0.05}).evaluate(_ConstRegressor(np.nan), self._data(), "regression", baseline_model=_ConstRegressor(0.0))
        assert not report.passed
        assert {c.name: c.status for c in report.checks}["predictions_finite"] is CheckStatus.FAIL

    def test_constant_regressor_is_rejected(self):
        report = ModelValidator({}, {"max_rmse_increase": 0.05}).evaluate(_ConstRegressor(0.0), self._data(), "regression", baseline_model=_ConstRegressor(0.0), history={"loss": [1.0, 0.5]})
        assert not report.passed
        assert {c.name: c.status for c in report.checks}["non_constant_output"] is CheckStatus.FAIL

    def test_mode_collapse_is_not_applicable_for_regression_not_pass(self):
        report = ModelValidator({}, {}).evaluate(_ConstRegressor(0.0), self._data(), "regression")
        assert {c.name: c.status for c in report.checks}["mode_collapse"] is CheckStatus.NOT_APPLICABLE

    def test_missing_baseline_or_threshold_is_incomplete_and_blocks(self):
        report = ModelValidator({}, {}).evaluate(_ConstRegressor(0.0), self._data(), "regression")
        statuses = {c.name: c.status for c in report.checks}
        assert statuses["rmse_vs_base"] is CheckStatus.INCOMPLETE
        assert statuses["convergence"] is CheckStatus.INCOMPLETE
        assert not report.passed

    def test_divergent_history_fails_convergence(self):
        check = ModelValidator({}, {}).convergence_check({"loss": [1.0, float("nan")], "val_loss": [1.0, 2.0]})
        assert check.status is CheckStatus.FAIL

    def test_legacy_validate_no_longer_passes_nan_regressor(self):
        passed, results = ModelValidator({}, {}).validate(_ConstRegressor(np.nan), self._data())
        assert passed is False and "mode_collapse_ok" not in results


# --------------------------------------------------------------------------------------- T-AI-06
def _fake_tl_run(tmp_path: Path) -> Path:
    run = tmp_path / "configs" / "classifiers" / "pc_server_tl" / "2026-01-01_run01"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        "domain: dc_motor\ntier: pc\ninput_window: 16\nnum_classes: 2\nblocks:\n  - type: conv1d\n    filters: 4\n    kernel_size: 3\ndense_units: []\ndropout: 0.0\n"
    )
    _tiny_classifier().save_weights(run / "model.weights.h5")
    (run / "metrics.json").write_text(json.dumps({"domain": "dc_motor", "channels": ["a", "b"], "fine_tuned_from": "base"}))
    return run


class TestAI06PromotionGate:
    def test_backend_rejects_tl_run_without_gate_and_registry_is_untouched(self, tmp_path):
        reg = tmp_path / "registry.yaml"
        reg.write_text("entries:\n  dc_motor/pc/classifier: configs/classifiers/pc_server/2026-09-01_run01\n")
        before = reg.read_text()
        run = _fake_tl_run(tmp_path)
        assert is_transfer_learning_run(run)
        with pytest.raises(PromotionRejected):
            promote(run, registry_path=reg)
        assert reg.read_text() == before
        assert registry_history(reg) == []

    def test_forged_approved_status_without_evidence_is_rejected(self, tmp_path):
        reg = tmp_path / "registry.yaml"
        run = _fake_tl_run(tmp_path)
        (run / "gate_report.json").write_text(json.dumps({"status": "APPROVED", "approved_by": "someone", "items": []}))
        with pytest.raises(PromotionRejected):
            promote(run, registry_path=reg)
        assert _load_index(reg) == {}

    @pytest.mark.parametrize("pending", ["human_approval", "validation"])
    def test_gate_blocks_on_any_pending_or_failed_item(self, pending):
        evidence = dict(
            dataset_summary={"ok": True, "sha256": "x" * 64, "n_acquisitions": 9},
            split_manifest={"disjoint": True, "n_groups": {"train": 3, "val": 3, "test": 3}},
            test_evaluation={"n_evaluations": 1, "used_for_selection": False, "same_test_for_base": True},
            validation_report={"checks": [{"name": "macro_f1_vs_base", "status": "PASS", "detail": "", "required": True}]},
            base_unchanged=True,
            lineage={"base_run_dir": "b", "base_weights_sha256": "h", "dataset_sha256": "d", "strategy": "adapter", "seed": 0, "split": {"x": 1}},
            human_approval={"approved": True, "approved_by": "reviewer"},
        )
        assert PromotionGate().evaluate(**evidence).approved
        if pending == "human_approval":
            evidence["human_approval"] = None
        else:
            evidence["validation_report"] = {"checks": [{"name": "rmse_vs_base", "status": "INCOMPLETE", "detail": "", "required": True}]}
        decision = PromotionGate().evaluate(**evidence)
        assert not decision.approved and decision.blocking

    def test_override_requires_reason_and_author_and_is_audited(self, tmp_path):
        reg = tmp_path / "registry.yaml"
        run = _fake_tl_run(tmp_path)
        with pytest.raises(PromotionRejected):
            promote(run, registry_path=reg, override={"reason": "", "by": "x"})
        promote(run, registry_path=reg, override={"reason": "bench experiment", "by": "researcher"})
        event = registry_history(reg)[-1]
        assert event["action"] == "promote_override" and event["override_reason"] == "bench experiment"

    def test_from_scratch_runs_still_promote_and_rollback_restores_previous(self, tmp_path):
        reg = tmp_path / "registry.yaml"
        src = Path(__file__).resolve().parents[1] / "configs" / "classifiers" / "pc_server" / "2026-09-01_run01"
        run1, run2 = tmp_path / "r1", tmp_path / "r2"
        for r in (run1, run2):
            r.mkdir()
            for f in ("config.yaml", "metrics.json", "model.weights.h5"):
                shutil.copyfile(src / f, r / f)
        promote(run1, registry_path=reg)
        promote(run2, registry_path=reg)
        assert resolve("dc_motor", "pc", "classifier", registry_path=reg) == run2.resolve()
        rollback("dc_motor", "pc", "classifier", registry_path=reg, by="test")
        assert resolve("dc_motor", "pc", "classifier", registry_path=reg) == run1.resolve()
        assert [e["action"] for e in registry_history(reg)] == ["promote", "promote", "rollback"]


# --------------------------------------------------------------------------------------- T-AI-07
class _UniformClassifier:
    def predict(self, X, verbose=0):
        return np.full((len(X), 3), 1 / 3, dtype="float32")


class TestClassifierMustBeatMajorityBaseline:
    def test_useless_model_fails_even_if_equal_to_a_useless_base(self):
        y = np.repeat([0, 1, 2], 4)
        data = {"X": np.zeros((12, 8, 1), dtype="float32"), "y": y}
        report = ModelValidator({}, {"max_macro_f1_drop": 0.02}).evaluate(_UniformClassifier(), data, "classification", baseline_model=_UniformClassifier(), history={"loss": [1.0, 0.9]})
        statuses = {c.name: c.status for c in report.checks}
        assert statuses["macro_f1_vs_base"] is CheckStatus.PASS  # equal to the base...
        assert statuses["beats_majority_baseline"] is CheckStatus.FAIL  # ...but no better than chance
        assert not report.passed


class TestAI07FairComparison:
    def test_base_and_candidate_are_scored_on_the_same_test_windows_once(self, tmp_path):
        df = pd.concat([_acq(f"s{i}", 200, ["normal", "outer_race"][i % 2]) for i in range(8)], ignore_index=True)
        base_model = _tiny_classifier(n_channels=2, window=16)
        base_dir = tmp_path / "pc_server" / "run01"
        base_dir.mkdir(parents=True)
        base_model.save_weights(base_dir / "model.weights.h5")
        base = {"model": base_model, "metrics": {"channels": ["current_r", "acc_x"], "classes": ["normal", "outer_race"]}, "run_dir": base_dir}
        ds = wf.prepare_dataset("dc_motor", "classifier", base["metrics"], 16, None, df, seed=0, val_frac=0.25, test_frac=0.25)
        res = wf.finetune(base, ds, {"strategy": FineTuneStrategy.ADAPTER, "epochs": 2, "batch_size": 8}, seed=0)
        assert res.test_evaluation == {"n_evaluations": 1, "used_for_selection": False, "same_test_for_base": True, "n_test_windows": len(ds.windows["test"]["X"])}
        assert res.report["metrics"]["n"] == res.report["baseline_metrics"]["n"] == len(ds.windows["test"]["X"])
        assert res.base_unchanged


# --------------------------------------------------------------------------------------- T-AI-08 / 09
class TestAI08ConcurrentAlerts:
    def test_resolving_one_of_two_alerts_keeps_domain_unhealthy(self):
        sa = ServerAgent()
        sa.record_alert("dc_motor", Alert("rule_A", "high", "alert", 1.0))
        sa.record_alert("dc_motor", Alert("rule_B", "medium", "alert", 2.0))
        sa.record_alert("dc_motor", Alert("rule_A", "high", "alert", 3.0, resolved=True))
        assert sa.domain_status("dc_motor") == "alert"
        assert sa.domain_health("dc_motor") == "WARNING"
        sa.record_alert("dc_motor", Alert("rule_B", "medium", "alert", 4.0, resolved=True))
        assert sa.domain_health("dc_motor") == "HEALTHY"

    def test_out_of_order_and_repeated_events(self):
        sa = ServerAgent()
        sa.record_alert("d", Alert("A", "medium", "alert", 1.0, resolved=True))  # resolve before fire
        sa.record_alert("d", Alert("A", "medium", "alert", 2.0))
        sa.record_alert("d", Alert("A", "medium", "alert", 3.0))  # repeated
        assert len(sa.active_alerts("d")) == 1
        sa.record_alert("d", Alert("A", "high", "alert", 4.0))  # escalation
        assert sa.domain_health("d") == "CRITICAL"

    def test_same_rule_from_two_sources_is_tracked_separately(self):
        sa = ServerAgent()
        sa.record_alert("d", Alert("A", "high", "alert", 1.0), source_id="motor_1")
        sa.record_alert("d", Alert("A", "high", "alert", 1.0), source_id="motor_2")
        sa.record_alert("d", Alert("A", "high", "alert", 2.0, resolved=True), source_id="motor_1")
        assert sa.domain_health("d") == "CRITICAL"

    def test_acknowledge_does_not_resolve(self):
        sa = ServerAgent()
        sa.record_alert("d", Alert("A", "high", "alert", 1.0))
        assert sa.acknowledge("d", "A", by="operator")
        assert sa.domain_health("d") == "CRITICAL"


class TestAI09NoTelemetryNeverHealthy:
    def test_never_reported_domain_is_unknown(self):
        assert ServerAgent().domain_health("dc_motor") == "UNKNOWN"

    def test_connected_source_without_heartbeat_goes_offline(self):
        sa = ServerAgent()
        sa.record_heartbeat("dc_motor", timestamp=100.0, timeout_s=5.0)
        assert sa.domain_health("dc_motor", now=103.0) == "HEALTHY"
        assert sa.domain_health("dc_motor", now=106.0) == "OFFLINE"

    def test_motor_view_is_unknown_without_monitoring_or_in_demo(self):
        assert derive_motor_view(None, "dc_motor", motor_id="m", source_mode="simulated", rpm=1500).status == "unknown"
        assert derive_motor_view(ServerAgent(), "dc_motor", motor_id="m", source_mode="demo", rpm=1500).status == "unknown"

    def test_motor_view_follows_all_active_alerts(self):
        sa = ServerAgent()
        sa.record_evaluation("dc_motor", 1.0)
        assert derive_motor_view(sa, "dc_motor", motor_id="m", source_mode="simulated").status == "healthy"
        sa.record_alert("dc_motor", Alert("A", "high", "alert", 2.0))
        view = derive_motor_view(sa, "dc_motor", motor_id="m", source_mode="simulated", rpm=0.0)
        assert view.status == "critical" and view.animate is False


# --------------------------------------------------------------------------------------- AI-07
class TestAI07VibrationGeneralizationReport:
    def test_report_reproduces_the_stored_leave_bearing_out_verdict(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "experiments" / "report_vibration_generalization.py"
        spec = importlib.util.spec_from_file_location("report_vibration_generalization", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        r = mod.report()
        assert r["n_folds"] == 6
        assert r["n_model_beats_null"] == 0  # the yaml's own note: worse than flat PSD in 6/6
        lo, hi = r["bootstrap_95ci_mean_ratio"]
        assert lo <= r["mean_ratio_model_over_null"] <= hi
