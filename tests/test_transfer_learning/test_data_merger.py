"""Tests for DataMerger (Phase 2) -- not in the original spec's Phase 1 test list, added because
data_merger.py got a real implementation too, not just loader/pipeline/validator."""

import pandas as pd
import pytest

from driveflow.ai.transfer.data_merger import DataMerger


def _sim_df(n=100):
    return pd.DataFrame({"acc_x": range(n), "current_r": range(n), "label": (["normal", "outer_race"] * n)[:n]})


def _ext_df(n=20):
    return pd.DataFrame({"acc_x": range(n), "current_r": range(n), "label": (["normal", "outer_race"] * n)[:n]})


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
        assert len(merged) == 100
        counts = merged["_data_source"].value_counts()
        assert counts["simulated"] == 70
        assert counts["external"] == 30

    def test_bounded_by_available_external_rows(self):
        merger = DataMerger(simulated_data=_sim_df(1000), external_data=_ext_df(5), mix_ratio=0.5)
        merged = merger.merge()
        assert (merged["_data_source"] == "external").sum() == 5

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
        df = pd.DataFrame({"label": ["normal"] * 80 + ["outer_race"] * 20})
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
        assert stats["n_rows"] == 50
        assert stats["by_source"]["simulated"] == 25
        assert "by_label" in stats
