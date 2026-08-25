"""Regression test for the mandatory Fase C entry-condition safeguard (INSTRUCTIONS.md Sec. 5,
punto 3): the training-set construction pipeline must exclude every "vsc_dpc_v1" record. Written
and passing BEFORE any paper_federative architecture is ported, per Sec. 10's explicit ordering.
"""

import numpy as np
import pandas as pd
import pytest

from driveflow.datagen import Scenario, export_parquet, run_scenario
from driveflow.models.common import DIAGNOSIS_PLANT_CONFIG_IDS, filter_diagnosis_domain, load_diagnosis_dataset


def _mixed_dataframe():
    dc_records = run_scenario(Scenario(scenario_id="dc", duration_s=0.005))
    vsc_records = run_scenario(Scenario(scenario_id="vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.005))
    return pd.DataFrame.from_records(dc_records + vsc_records)


class TestDiagnosisPlantConfigIds:
    def test_vsc_domain_is_not_included(self):
        """The one assertion this whole safeguard exists for."""
        assert "vsc_dpc_v1" not in DIAGNOSIS_PLANT_CONFIG_IDS

    def test_dc_motor_domain_is_included(self):
        assert "dc_perm_ex_v1" in DIAGNOSIS_PLANT_CONFIG_IDS


class TestFilterDiagnosisDomain:
    def test_excludes_all_vsc_rows(self):
        df = _mixed_dataframe()
        filtered = filter_diagnosis_domain(df)
        assert "vsc_dpc_v1" not in set(filtered["plant_config_id"])

    def test_keeps_all_dc_motor_rows(self):
        df = _mixed_dataframe()
        n_dc = int((df["plant_config_id"] == "dc_perm_ex_v1").sum())
        filtered = filter_diagnosis_domain(df)
        assert len(filtered) == n_dc
        assert set(filtered["plant_config_id"]) == {"dc_perm_ex_v1"}

    def test_filtered_rows_have_no_nan_in_diagnosis_columns(self):
        """The concrete failure mode this safeguard prevents: a vsc_dpc_v1 row has NaN in every
        column a bearing-fault/vibration model reads (see models/common/dataset.py's docstring).
        After filtering, none of those columns should contain NaN from a leaked VSC row."""
        df = _mixed_dataframe()
        filtered = filter_diagnosis_domain(df)
        for col in ("acc_x", "acc_y", "acc_z", "current_r", "rpm", "torque_nm"):
            assert filtered[col].notna().all()

    def test_all_vsc_dataset_filters_to_empty(self):
        vsc_records = run_scenario(Scenario(scenario_id="vsc_only", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.005))
        df = pd.DataFrame.from_records(vsc_records)
        filtered = filter_diagnosis_domain(df)
        assert len(filtered) == 0


class TestLoadDiagnosisDataset:
    def test_reads_parquet_and_excludes_vsc(self, tmp_path):
        dc_records = run_scenario(Scenario(scenario_id="dc", duration_s=0.005))
        vsc_records = run_scenario(Scenario(scenario_id="vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.005))
        out_path = tmp_path / "mixed.parquet"
        export_parquet([dc_records, vsc_records], out_path)

        loaded = load_diagnosis_dataset(out_path)
        assert "vsc_dpc_v1" not in set(loaded["plant_config_id"])
        assert len(loaded) == len(dc_records)
