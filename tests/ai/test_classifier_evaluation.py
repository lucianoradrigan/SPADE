"""Classifier evaluation -- docs/AI_TECHNICAL_AUDIT.md Sec. 2-3.

Unit tests of the metrics (ai/evaluation.py) on hand-computed cases, the train/inference
preprocessing contract (AI-N01), and short fresh-run checks of the registered classifiers.
The full generalization study (speed, severity, noise, plant drift, unseen fault types) is
experiments/report_classifier_evaluation.py -> docs/audit/ai/classifier_evaluation.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driveflow.ai.calibration import UNCERTAIN, abstention_threshold, apply_temperature, decide, fit_temperature
from driveflow.ai.evaluation import classification_metrics, expected_calibration_error, ood_confidence_summary
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.export_parquet import records_to_dataframe
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import load_classifier_config
from driveflow.ai.registry import resolve
from driveflow.models.common import RPM_LEVEL_SCALING, build_classification_windows, normalize_window_like_training
from driveflow.viz.adapters.ai_adapter import normalize_window

pytestmark = pytest.mark.ai

ROOT = Path(__file__).resolve().parents[2]
#: Promoted 2026-10-18 (AI-N06/AI-N07): trained on the speed x severity x noise x plant-drift grid
#: (experiments/generate_diagnosis_grid_dataset.py), rpm level kept, temperature-calibrated.
PROMOTED = ROOT / "configs" / "classifiers" / "pc_server" / "2026-10-18_run01"
#: Before 2026-10-18: production model whose inner_race recall was 0 (AI-N06).
OLD_PROMOTED = ROOT / "configs" / "classifiers" / "pc_server" / "2026-09-01_run01"
#: Single-operating-point candidate of 2026-10-13 (AI-N07 evidence).
SINGLE_SPEED = ROOT / "configs" / "classifiers" / "pc_server" / "2026-10-13_run01"
CLASSES = ["inner_race", "normal", "outer_race"]


class TestMetrics:
    def test_hand_computed_case(self):
        # true: 2x inner, 2x normal, 2x outer; pred: inner, outer | normal, inner | outer, outer
        y = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 2, 1, 0, 2, 2])
        probs = np.eye(3)[pred] * 0.7 + 0.1
        m = classification_metrics(y, probs, CLASSES)
        assert m["accuracy"] == pytest.approx(4 / 6)
        assert m["balanced_accuracy"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
        assert m["per_class"]["outer_race"]["precision"] == pytest.approx(2 / 3)
        assert m["per_class"]["inner_race"]["fnr"] == pytest.approx(0.5)
        assert m["per_class"]["inner_race"]["fpr"] == pytest.approx(1 / 4)
        assert m["confusion_matrix_rows_true"] == [[1, 0, 1], [1, 1, 0], [0, 0, 2]]
        # healthy windows: 1 of 2 reported as a fault; faulty windows: none reported as normal
        assert m["fault_detection"] == {"false_alarm_rate": 0.5, "missed_detection_rate": 0.0}

    def test_ece_is_zero_for_a_calibrated_model_and_large_for_an_overconfident_one(self):
        rng = np.random.default_rng(0)
        n = 20000
        conf = rng.uniform(0.4, 1.0, n)
        y_pred = rng.integers(0, 3, n)
        correct = rng.uniform(size=n) < conf  # correct exactly as often as it claims
        y_true = np.where(correct, y_pred, (y_pred + 1) % 3)
        probs = np.full((n, 3), 0.0)
        probs[np.arange(n), y_pred] = conf
        probs[np.arange(n), (y_pred + 1) % 3] = 1 - conf
        assert expected_calibration_error(probs, y_true) < 0.02
        over = np.eye(3)[y_pred] * 0.999 + 0.0005  # always ~100 % sure
        assert expected_calibration_error(over, y_true) > 0.15

    def test_rejects_nonfinite_or_misshaped_probabilities(self):
        with pytest.raises(ValueError):
            classification_metrics([0, 1], np.array([[np.nan, 0.5, 0.5], [0.2, 0.3, 0.5]]), CLASSES)
        with pytest.raises(ValueError):
            classification_metrics([0, 1], np.ones((2, 2)) / 2, CLASSES)

    def test_ood_summary(self):
        s = ood_confidence_summary(np.array([[0.05, 0.05, 0.9], [0.1, 0.1, 0.8]]), CLASSES)
        assert s["predicted_share"]["outer_race"] == 1.0 and s["mean_confidence"] == pytest.approx(0.85)


@pytest.fixture(scope="module")
def healthy_steady_state():
    df = records_to_dataframe(run_scenario(Scenario(scenario_id="norm_contract", duration_s=0.7, seed=5)))
    return df[df["timestamp_s"] >= 0.5].reset_index(drop=True)


class TestPreprocessingContract:
    """AI-N01: the GUI used a different normalisation from training."""

    def test_gui_window_equals_training_window(self, healthy_steady_state):
        channels = ["acc_x", "acc_y", "acc_z", "current_r", "rpm", "torque_nm"]
        X, _y, _g = build_classification_windows(healthy_steady_state, ["normal"], channels, 512)
        raw = healthy_steady_state[channels].to_numpy(np.float32)
        for i in range(len(X)):
            gui, _, _ = normalize_window(raw[i * 512 : (i + 1) * 512])
            np.testing.assert_allclose(gui, X[i], atol=1e-5)

    def test_the_contract_is_not_vacuous(self, healthy_steady_state):
        """The healthy steady state really has dead channels (current/torque constant), i.e. the
        case where the two old rules disagreed."""
        w = healthy_steady_state[["current_r", "torque_nm"]].to_numpy(np.float32)[:512]
        wn, _, sigma = normalize_window_like_training(w)
        assert np.all(wn == 0) and np.all(sigma == 1)

    def test_forecast_denormalisation_of_a_dead_channel_returns_its_level(self):
        w = np.full((64, 1), 45.5, dtype=np.float32)
        wn, mu, sigma = normalize_window_like_training(w)
        assert np.all(wn * sigma + mu == pytest.approx(45.5))


def _load(run_dir):
    m = json.loads((run_dir / "metrics.json").read_text())
    model = build_classifier(load_classifier_config(run_dir / "config.yaml"), n_channels=len(m["channels"]))
    model.load_weights(run_dir / "model.weights.h5")
    return model, m


@pytest.fixture(scope="module")
def fresh_steady_state():
    """One fresh run per class (seed 900, never used in training), steady state t in [0.5, 1.2) s."""
    frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"fresh_{ft}", fault_type=ft, duration_s=1.2, seed=900))) for ft in (None, "outer_race", "inner_race")]
    df = pd.concat(frames, ignore_index=True)
    return df[df["timestamp_s"] >= 0.5]


def _recall(run_dir, df, cls):
    model, m = _load(run_dir)
    window = load_classifier_config(run_dir / "config.yaml").input_window
    X, y, _ = build_classification_windows(df, m["classes"], m["channels"], window, level_scaling=m.get("level_scaling"))
    probs = model.predict(X, verbose=0)
    assert np.all(np.isfinite(probs)) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    k = m["classes"].index(cls)
    yk = np.array([m["classes"].index(c) for c in y])
    return float((probs.argmax(axis=1)[yk == k] == k).mean())


def test_registry_resolves_the_grid_trained_classifier():
    # Since 2026-10-20 production is the 5-class, 2048-sample-window model (A2, AI_TECHNICAL_AUDIT
    # Sec. 8.5); PROMOTED (the 3-class grid model) stays under test as the previous production run.
    assert resolve("dc_motor", "pc", "classifier") == ROOT / "configs" / "classifiers" / "pc_server_5class_w2048" / "2026-10-20_run01"


def test_promoted_classifier_recognizes_every_class(fresh_steady_state):
    """AI-N06 (fixed): the previous production model never predicted inner_race."""
    for cls in CLASSES:
        assert _recall(PROMOTED, fresh_steady_state, cls) >= 0.9, cls


def test_previous_production_model_still_shows_the_ai_n06_defect(fresh_steady_state):
    """Keeps the evidence reproducible: the replaced model's inner_race recall is 0."""
    assert _recall(OLD_PROMOTED, fresh_steady_state, "inner_race") <= 0.1


def _fresh(seed: int, **kw) -> pd.DataFrame:
    frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"fresh_{ft}", fault_type=ft, duration_s=1.2, seed=seed, **kw))) for ft in (None, "outer_race", "inner_race")]
    df = pd.concat(frames, ignore_index=True)
    return df[df["timestamp_s"] >= 0.5]


@pytest.mark.slow
@pytest.mark.parametrize(
    "kw",
    [
        {"omega_ref_rad_s": 115.0},  # between the grid's 100 and 125 rad/s
        {"omega_ref_rad_s": 250.0},  # beyond its 220 rad/s maximum
        {"electrical_severity": 4.8, "mechanical_severity_scale": 0.6},  # between its x0.5 and x0.75
    ],
    ids=["speed_115", "speed_250", "severity_x0.6"],
)
def test_promoted_classifier_generalizes_to_held_out_conditions(kw):
    """AI-N07 (fixed): held-out operating conditions, never in the training grid."""
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    kw = dict(kw)
    scale = kw.pop("mechanical_severity_scale", None)
    frames = []
    for ft in (None, "outer_race", "inner_race"):
        extra = dict(kw)
        if scale is not None and ft is not None:
            extra["mechanical_severity"] = CALIBRATED_MECHANICAL_SEVERITY[ft] * scale
        if ft is None:
            extra.pop("electrical_severity", None)
        frames.append(records_to_dataframe(run_scenario(Scenario(scenario_id=f"held_{ft}", fault_type=ft, duration_s=1.2, seed=902, **extra))))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["timestamp_s"] >= 0.5]
    for cls in CLASSES:
        assert _recall(PROMOTED, df, cls) >= 0.9, cls


@pytest.mark.slow
def test_single_speed_model_confuses_inner_race_at_100_rad_s():
    """The AI-N07 root cause, kept as evidence: trained at one speed with the speed level z-scored
    away, at 100 rad/s the inner-race rate (78.5 Hz) sits where the outer-race one was at 150 rad/s
    (73.3 Hz)."""
    df = records_to_dataframe(run_scenario(Scenario(scenario_id="fresh_speed", fault_type="inner_race", duration_s=1.2, seed=901, omega_ref_rad_s=100.0)))
    assert _recall(SINGLE_SPEED, df[df["timestamp_s"] >= 0.5], "inner_race") <= 0.1


class TestLevelScalingContract:
    """AI-N07: a run's level_scaling is part of its preprocessing; the GUI must apply it too."""

    def test_gui_window_equals_training_window_with_level_scaling(self, healthy_steady_state):
        channels = ["acc_x", "acc_y", "acc_z", "current_r", "rpm", "torque_nm"]
        X, _y, _g = build_classification_windows(healthy_steady_state, ["normal"], channels, 512, level_scaling=RPM_LEVEL_SCALING)
        raw = healthy_steady_state[channels].to_numpy(np.float32)
        metrics = {"channels": channels, "level_scaling": RPM_LEVEL_SCALING}
        for i in range(len(X)):
            gui, _, _ = normalize_window(raw[i * 512 : (i + 1) * 512], metrics)
            np.testing.assert_allclose(gui, X[i], atol=1e-5)

    def test_rpm_level_survives_and_other_channels_are_unchanged(self, healthy_steady_state):
        channels = ["acc_x", "rpm"]
        w = healthy_steady_state[channels].to_numpy(np.float32)[:512]
        old, _, _ = normalize_window_like_training(w)
        new, _, _ = normalize_window_like_training(w, channels, RPM_LEVEL_SCALING)
        np.testing.assert_array_equal(new[:, 0], old[:, 0])
        assert np.all(old[:, 1] == 0)  # steady speed: a dead channel, the level was lost
        center, scale = RPM_LEVEL_SCALING["rpm"]
        np.testing.assert_allclose(new[:, 1], (w[:, 1] - center) / scale, rtol=1e-6)

    def test_level_scaling_needs_channel_names(self):
        with pytest.raises(ValueError, match="channel names"):
            normalize_window_like_training(np.zeros((8, 1), np.float32), level_scaling=RPM_LEVEL_SCALING)


class TestCalibration:
    """AI-N12: temperature scaling + abstention, fitted on validation data."""

    def _overconfident(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 3, n)
        pred = np.where(rng.uniform(size=n) < 0.8, y, (y + 1) % 3)  # 80 % accurate
        probs = np.full((n, 3), 0.005)
        probs[np.arange(n), pred] = 0.99
        return probs, y

    def test_temperature_keeps_the_predicted_class(self):
        probs, _ = self._overconfident()
        np.testing.assert_array_equal(apply_temperature(probs, 3.0).argmax(1), probs.argmax(1))

    def test_fitted_temperature_softens_an_overconfident_model(self):
        probs, y = self._overconfident()
        t, before, after = fit_temperature(probs, y)
        assert t > 1.0 and after < before
        assert expected_calibration_error(apply_temperature(probs, t), y) < expected_calibration_error(probs, y)

    def test_abstention_threshold_keeps_the_target_coverage_of_correct_windows(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet([1, 1, 1], 5000)
        y = p.argmax(1)
        th = abstention_threshold(p, y, 0.95)
        assert np.mean(p.max(1) >= th) == pytest.approx(0.95, abs=0.01)

    def test_decide_abstains_below_the_threshold_and_ignores_uncalibrated_runs(self):
        metrics = {"calibration": {"temperature": 1.0, "abstain_below": 0.6}}
        assert decide(np.array([0.5, 0.3, 0.2]), CLASSES, metrics)[0] == UNCERTAIN
        assert decide(np.array([0.7, 0.2, 0.1]), CLASSES, metrics)[0] == "inner_race"
        assert decide(np.array([0.5, 0.3, 0.2]), CLASSES, {})[0] == "inner_race"

    def test_promoted_run_records_its_calibration(self):
        m = json.loads((PROMOTED / "metrics.json").read_text())
        cal = m["calibration"]
        assert cal["fitted_on"] == "val" and cal["temperature"] > 0 and 0 < cal["abstain_below"] < 1
        assert m["level_scaling"] == RPM_LEVEL_SCALING
