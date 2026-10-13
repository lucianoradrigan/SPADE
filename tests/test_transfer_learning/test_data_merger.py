"""Tests for DataMerger (Phase 2) -- not in the original spec's Phase 1 test list, added because
data_merger.py got a real implementation too, not just loader/pipeline/validator.

AUDIT AI-01 changed the mixing unit from individual rows to whole acquisitions (see
data_merger.py's module docstring), so the fixtures below are now lists of 10-row recordings with
a source_file and a timestamp, and the expected counts are counts of whole recordings. The
previous row-level expectations (exactly 70/30 rows) described the bug being fixed.
"""

import numpy as np
import pandas as pd
import pytest

from driveflow.ai.transfer.data_merger import DataMerger


def _recordings(prefix, n_recordings, rows_per=10, labels=("normal", "outer_race")):
    parts = []
    for i in range(n_recordings):
        t = np.arange(1, rows_per + 1) * 1e-4
        parts.append(pd.DataFrame({"timestamp_s": t, "acc_x": np.arange(rows_per), "current_r": np.arange(rows_per), "label": labels[i % len(labels)], "source_file": f"{prefix}{i}"}))
    return pd.concat(parts, ignore_index=True)


def _sim_df(n=100):
    return _recordings("sim_run_", n // 10)


def _ext_df(n=20):
    return _recordings("ext_run_", max(1, n // 10))


class TestValidateSchema:
    def test_true_when_a_known_channel_is_present(self):
        merger = DataMerger(external_data=_ext_df())
        assert merger.validate_schema("dc_motor") is True

    def test_false_when_no_known_channel_is_present(self):
        merger = DataMerger(external_data=pd.DataFrame({"unrelated_col": [1, 2, 3]}))
        assert merger.validate_schema("dc_motor") is False

    def test_false_when_no_external_data_given(self):
        merger = DataMerger(simulated_data=_sim_df())
        assert merger.validate_schema("dc_motor") is False

    def test_unknown_domain_raises(self):
        merger = DataMerger(external_data=_ext_df())
        with pytest.raises(ValueError):
            merger.validate_schema("not_a_real_domain")


class TestMerge:
    def test_respects_mix_ratio_approximately(self):
        merger = DataMerger(simulated_data=_sim_df(100), external_data=_ext_df(100), mix_ratio=0.7)
        merged = merger.merge()
        counts = merged["_data_source"].value_counts()
        # Whole 10-row recordings: 7 simulated + 3 external at the largest feasible total (10 + 4.3).
        assert counts["simulated"] % 10 == 0 and counts["external"] % 10 == 0
        assert abs(counts["simulated"] / len(merged) - 0.7) <= 0.1

    def test_bounded_by_available_external_rows(self):
        merger = DataMerger(simulated_data=_sim_df(1000), external_data=_ext_df(5), mix_ratio=0.5)
        merged = merger.merge()
        assert (merged["_data_source"] == "external").sum() == 10  # the single external recording, whole
        assert (merged["_data_source"] == "simulated").sum() == 10

    def test_only_simulated_data_given(self):
        merger = DataMerger(simulated_data=_sim_df(10))
        merged = merger.merge()
        assert len(merged) == 10

    def test_neither_given_raises(self):
        merger = DataMerger()
        with pytest.raises(ValueError):
            merger.merge()


class TestStratifyByFault:
    def test_balances_classes_to_the_minority_count(self):
        df = pd.concat([_recordings("n", 8, labels=("normal",)), _recordings("o", 2, labels=("outer_race",))], ignore_index=True)
        merger = DataMerger(simulated_data=df)
        merger.merge()
        stratified = merger.stratify_by_fault()
        counts = stratified["label"].value_counts()
        assert counts["normal"] == 20
        assert counts["outer_race"] == 20

    def test_requires_merge_first(self):
        merger = DataMerger(simulated_data=_sim_df())
        with pytest.raises(ValueError, match="merge"):
            merger.stratify_by_fault()


class TestGetStatistics:
    def test_reports_row_count_and_breakdowns(self):
        merger = DataMerger(simulated_data=_sim_df(50), external_data=_ext_df(50), mix_ratio=0.5)
        merger.merge()
        stats = merger.get_statistics()
        assert stats["n_rows"] == 100
        assert stats["by_source"] == {"simulated": 50, "external": 50}
        assert stats["realized_mix_ratio"] == 0.5 and stats["mix_unit"] == "rows"
        assert "by_label" in stats
