"""Domain-filtering safeguard for Fase C (Macro-fase C, INSTRUCTIONS.md Sec. 5, punto 3):
mandatory entry condition, enforced here and regression-tested in
tests/test_diagnosis_dataset_filter.py, BEFORE any paper_federative architecture gets ported or
trained.

Why this exists: driveflow's exported Parquet (datagen/export_parquet.py) can hold records from
two physically unrelated domains in the same file/schema (see datagen/runner.py's module
docstring) -- "dc_perm_ex_v1" (the DC motor + bearing + Module B vibration/MCSA path Fase A
built) and "vsc_dpc_v1" (the Voltage Source Converter Fase B's DPC controller runs against, with
no motor, no bearing, no vibration at all -- see docs/patch5_alcance_macrofase_B.md). Fase C's
classifiers/regressor (sensor_dscnn-equivalent, gateway_resnet_se-equivalent,
envelope_forecaster-equivalent -- see the paper_federative exploration this module's callers
should reference) are bearing-fault/vibration models. A "vsc_dpc_v1" row has NaN in every column
those models read (acc_x/y/z, current_r, rpm, torque_nm, bpfo_hz...) -- silently including it
would not crash training, it would just inject rows of NaN/garbage into the training set. This
module is the one place that decides which plant_config_id values are valid input to Fase C, so
that decision is enforced once, not re-implemented (and potentially forgotten) at every call site.
"""

import pandas as pd

#: plant_config_id values that carry the physical signals (vibration, MCSA) Fase C's models
#: consume. Currently just the DC motor domain from Fase A -- "vsc_dpc_v1" (Fase B) is
#: deliberately NOT here, see module docstring.
DIAGNOSIS_PLANT_CONFIG_IDS = frozenset({"dc_perm_ex_v1"})


def load_diagnosis_dataset(parquet_path) -> pd.DataFrame:
    """Reads an exported Parquet dataset and returns only the rows from domains Fase C's models
    are meant to train on -- silently dropping any other plant_config_id (currently just
    "vsc_dpc_v1"), never raising on their presence: a mixed-domain export is the expected,
    supported shape of the dataset (see datagen/runner.py), not an error condition. What would be
    a bug is a caller reading the raw, unfiltered Parquet directly instead of going through here.
    """
    df = pd.read_parquet(parquet_path)
    return filter_diagnosis_domain(df)


def filter_diagnosis_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Same filter as load_diagnosis_dataset, operating on an already-loaded DataFrame (e.g. one
    built in-memory via datagen.export_parquet.records_to_dataframe without writing to disk)."""
    return df[df["plant_config_id"].isin(DIAGNOSIS_PLANT_CONFIG_IDS)].reset_index(drop=True)
