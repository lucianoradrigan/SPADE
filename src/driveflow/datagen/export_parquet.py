"""Exports datagen/runner.py records into the extended Parquet schema, per
docs/propuesta_consolidacion.pdf Sec. 4.5 and its extension in docs/addendum_vibracion_v1.md
Sec. 6 (acc_x/y/z populated in v1, plus vibration_source; audio stays NaN, out of scope).

The v_ref_real/v_ref_imag/vc_real/vc_imag/i_f_real/i_f_imag columns are Macro-fase B's addition
(docs/macro_fase_B2_dpc_deployment.md) for "vsc_dpc_v1" scenario records -- NaN for
"dc_perm_ex_v1" records, same as current_s/current_t/rpm/torque_nm/voltage_v/acc_x.../bpfo_hz... are NaN
for "vsc_dpc_v1" records (see runner.py's module docstring: these are two unrelated plants
sharing one wide schema, not one plant with optional fields).
"""

from pathlib import Path

import pandas as pd

#: Column order matches the addendum-extended schema: paper_federative's original columns first,
#: then the simulation-metadata columns the proposal adds (Sec. 4.5), then the per-fault
#: characteristic frequencies (NaN for healthy runs).
SCHEMA_COLUMNS = [
    "timestamp_s",
    "current_r",
    "current_s",
    "current_t",
    "acc_x",
    "acc_y",
    "acc_z",
    "vibration_source",
    "audio",
    "rpm",
    "torque_nm",
    "voltage_v",
    "label",
    "electrical_severity_nm",
    "mechanical_severity",
    "dataset_id",
    "source_file",
    "fs_hz",
    "controller_type",
    "plant_config_id",
    "bpfo_hz",
    "bpfi_hz",
    "bsf_hz",
    "ftf_hz",
    "seed",
    "v_ref_real",
    "v_ref_imag",
    "vc_real",
    "vc_imag",
    "i_f_real",
    "i_f_imag",
]


def records_to_dataframe(records: list) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    missing = set(SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"records are missing schema columns: {sorted(missing)}")
    return df[SCHEMA_COLUMNS]


def export_parquet(scenario_records, out_path) -> pd.DataFrame:
    """
    Args:
        scenario_records: Either a flat list[dict] (one scenario's records) or a list of such
            lists (several scenarios, as returned by runner.run_scenarios) -- concatenated in
            either case.
        out_path: Destination .parquet path (parent directories created if needed).

    Returns:
        The exported DataFrame (schema-ordered), for immediate inspection/notebook use without
        re-reading the file.
    """
    if scenario_records and isinstance(scenario_records[0], list):
        flat_records = [record for run in scenario_records for record in run]
    else:
        flat_records = list(scenario_records)

    df = records_to_dataframe(flat_records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
