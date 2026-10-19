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

from driveflow.ai.evaluation import classification_metrics, expected_calibration_error, ood_confidence_summary
from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.export_parquet import records_to_dataframe
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import load_classifier_config
from driveflow.models.common import build_classification_windows, normalize_window_like_training
from driveflow.viz.adapters.ai_adapter import normalize_window

pytestmark = pytest.mark.ai

ROOT = Path(__file__).resolve().parents[2]
PROMOTED = ROOT / "configs" / "classifiers" / "pc_server" / "2026-09-01_run01"
CANDIDATE = ROOT / "configs" / "classifiers" / "pc_server" / "2026-10-13_run01"
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
    X, y, _ = build_classification_windows(df, m["classes"], m["channels"], 512)
    probs = model.predict(X, verbose=0)
    assert np.all(np.isfinite(probs)) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    k = m["classes"].index(cls)
    yk = np.array([m["classes"].index(c) for c in y])
    return float((probs.argmax(axis=1)[yk == k] == k).mean())


def test_candidate_recognizes_every_class_at_the_training_operating_point(fresh_steady_state):
    for cls in CLASSES:
        assert _recall(CANDIDATE, fresh_steady_state, cls) >= 0.9, cls


@pytest.mark.xfail(strict=True, reason="AUDIT_ANSWERS Q15 / AI_TECHNICAL_AUDIT AI-N06 (open): the PROMOTED classifier "
                   "never predicts inner_race (recall 0.00); promotion of the corrected candidate awaits the researcher.")
def test_promoted_classifier_recognizes_inner_race(fresh_steady_state):
    assert _recall(PROMOTED, fresh_steady_state, "inner_race") >= 0.9


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="AI-N07 (open): trained at one speed (150 rad/s) with per-window z-scoring "
                   "that removes the speed level; at 100 rad/s the inner-race frequency (78.5 Hz) sits where the "
                   "outer-race one was at 150 rad/s (73.3 Hz) and every inner_race window is called outer_race.")
def test_candidate_generalizes_to_another_speed():
    df = records_to_dataframe(run_scenario(Scenario(scenario_id="fresh_speed", fault_type="inner_race", duration_s=1.2, seed=901, omega_ref_rad_s=100.0)))
    assert _recall(CANDIDATE, df[df["timestamp_s"] >= 0.5], "inner_race") >= 0.9
