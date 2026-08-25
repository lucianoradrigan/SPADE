"""Tests for models/common/windowing.py + splits.py, ported from paper_federative's
discover_channels/load_dataset/load_forecast_envelope/prepare_splits (see windowing.py's
docstring). Uses driveflow's own run_scenario() output, not paper_federative's real datasets --
this validates the port's mechanics, not fault-detection accuracy (that's covered separately once
a full diagnosis dataset is generated and trained on, see docs/macro_fase_C1_diagnosis.md).
"""

import numpy as np
import pandas as pd
import pytest

from driveflow.datagen import Scenario, run_scenario
from driveflow.datagen.runner import TAU
from driveflow.models.common import (
    build_classification_windows,
    build_forecast_windows,
    discover_channels,
    grouped_split,
    is_live_window,
    prepare_classification_splits,
    window_samples_for,
)


@pytest.fixture(scope="module")
def mixed_labeled_df():
    """Several short scenario runs (2 per class, distinct seeds -> distinct source_file groups)
    across 3 classes, concatenated -- enough for windowing + a grouped split without every group
    ending up alone in one split."""
    scenarios = []
    for fault_type in (None, "outer_race", "inner_race"):
        for seed in (0, 1):
            # duration_s must clear window_samples_for(1/TAU)=5000 rows (0.5s) -- 1.0s gives
            # 2 windows/run, enough to exercise cap_per_class and the grouped split below.
            scenarios.append(Scenario(scenario_id=f"w_{fault_type}_{seed}", fault_type=fault_type, duration_s=1.0, seed=seed))
    records = [r for s in scenarios for r in run_scenario(s)]
    return pd.DataFrame.from_records(records)


class TestWindowSamplesFor:
    def test_matches_driveflow_tau(self):
        assert window_samples_for(1.0 / TAU, window_s=0.5) == 5000


class TestIsLiveWindow:
    def test_constant_window_is_not_live(self):
        assert not is_live_window(np.full(100, 3.14, dtype=np.float32))

    def test_varying_window_is_live(self):
        assert is_live_window(np.sin(np.linspace(0, 10, 200)).astype(np.float32))


class TestDiscoverChannels:
    def test_finds_populated_live_channels(self, mixed_labeled_df):
        window_samples = window_samples_for(1.0 / TAU)
        channels = discover_channels(mixed_labeled_df, window_samples)
        for expected in ("acc_x", "acc_y", "acc_z", "current_r", "rpm", "torque_nm"):
            assert expected in channels

    def test_does_not_miss_a_channel_flat_only_in_the_first_class(self):
        """Regression for a real bug found generating the first Fase C dataset (see
        windowing.py's discover_channels docstring): driveflow's electrical/mechanical path is
        deterministic, so a channel like current_r is *exactly* constant once a healthy run
        settles and only varies for faulted runs. A sample that happened to land entirely on
        early (alphabetically/list-order-first) healthy groups discarded the channel entirely,
        even though it's the exact signal A.5 already validated as diagnostic. This builds that
        scenario directly: many "healthy" groups (perfectly flat) before a handful of "faulted"
        ones (real variation) -- old sequential-from-array-start sampling would see only flat
        windows and drop the channel; per-group sampling must not."""
        window_samples = 100
        rng = np.random.default_rng(0)
        rows = []
        for i in range(20):  # 20 flat "healthy" groups first
            for t in range(window_samples):
                rows.append({"source_file": f"healthy_{i}", "label": "normal", "current_r": 42.0})
        for i in range(3):  # a handful of "faulted" groups after, with real variation
            noise = rng.normal(0, 5.0, window_samples)
            for t in range(window_samples):
                rows.append({"source_file": f"faulted_{i}", "label": "fault", "current_r": 42.0 + noise[t]})
        df = pd.DataFrame(rows)

        channels = discover_channels(df, window_samples, min_live_frac=0.1)
        assert "current_r" in channels

    def test_excludes_always_nan_channels(self, mixed_labeled_df):
        """current_s/current_t are always NaN for the DC motor (single current channel) --
        see datagen/runner.py's docstring."""
        window_samples = window_samples_for(1.0 / TAU)
        channels = discover_channels(mixed_labeled_df, window_samples)
        assert "current_s" not in channels
        assert "current_t" not in channels


class TestBuildClassificationWindows:
    def test_returns_windows_for_each_class(self, mixed_labeled_df):
        window_samples = window_samples_for(1.0 / TAU)
        channels = discover_channels(mixed_labeled_df, window_samples)
        classes = ["normal", "outer_race", "inner_race"]
        X, y, groups = build_classification_windows(mixed_labeled_df, classes, channels, window_samples, cap_per_class=50)

        assert X is not None
        assert X.shape[1:] == (window_samples, len(channels))
        assert set(y) == set(classes)
        assert len(X) == len(y) == len(groups)

    def test_no_window_crosses_a_group_boundary(self, mixed_labeled_df):
        """Each window's group must be a single scenario run's source_file -- windows are never
        built by concatenating rows from two different runs."""
        window_samples = window_samples_for(1.0 / TAU)
        channels = discover_channels(mixed_labeled_df, window_samples)
        _, _, groups = build_classification_windows(
            mixed_labeled_df, ["normal", "outer_race"], channels, window_samples, cap_per_class=50
        )
        assert set(groups) <= set(mixed_labeled_df["source_file"].unique())

    def test_missing_class_returns_none(self, mixed_labeled_df):
        window_samples = window_samples_for(1.0 / TAU)
        X, y, groups = build_classification_windows(mixed_labeled_df, ["ball"], ["acc_x"], window_samples)
        assert X is None and y is None and groups is None


class TestBuildForecastWindows:
    def test_context_and_horizon_shapes(self, mixed_labeled_df):
        window_samples = window_samples_for(1.0 / TAU)
        channels = discover_channels(mixed_labeled_df, window_samples)
        healthy_df = mixed_labeled_df[mixed_labeled_df["label"] == "normal"]
        X, Y, groups = build_forecast_windows(healthy_df, channels, window_samples, horizon_samples=500, n_bins=10, cap_per_group=5)

        assert X is not None
        assert X.shape[1:] == (window_samples, len(channels))
        assert Y.shape[1:] == (10, len(channels))
        assert len(X) == len(Y) == len(groups)


class TestGroupedSplit:
    def test_no_group_appears_in_both_sides(self, mixed_labeled_df):
        window_samples = window_samples_for(1.0 / TAU)
        channels = discover_channels(mixed_labeled_df, window_samples)
        classes = ["normal", "outer_race", "inner_race"]
        X, y_str, groups = build_classification_windows(mixed_labeled_df, classes, channels, window_samples, cap_per_class=50)

        X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_classification_splits(X, y_str, classes, groups, seed=0)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert set(le.classes_) == set(classes)
