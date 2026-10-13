"""Training-dataset validation for transfer learning (AUDIT AI-02, AI-12; AUDIT_ANSWERS.md Q03/Q04/Q07).

Why a separate validator: driveflow.viz.dpc_upload_validation's channel validators answer "can a
model be RUN on this file?" -- they keep only the numeric channel block and drop everything else.
Training needs a different contract: the rows must stay attached to their label, their recording
(`source_file`) and their time axis (`timestamp_s`), or the dataset cannot be windowed without
mixing recordings, cannot be split by acquisition, and has no supervision at all. Handing the
inference-validated frame to DataMerger (as the TL tab used to) silently stripped all three.

Contract enforced here (fail-closed -- nothing is filled in silently):
- Required metadata per task: classification needs `label`, `source_file`, `timestamp_s`;
  regression needs `source_file`, `timestamp_s`. Missing -> not trainable (the message says the
  file may still be usable for inference).
- Every channel the BASE model was trained on must be present (its input layer has fixed width).
- Labels must belong to the base model's class vocabulary (its output layer has fixed width).
- Timestamps must be strictly increasing within each acquisition (never re-sorted) and the
  sampling rate must match the simulation's (no resampling/anti-alias filter exists in this
  project -- a different rate is rejected, not converted).
- Rows with non-numeric / non-finite / implausible channel values are dropped and reported; the
  hole they leave is then seen as a discontinuity by windowing, so no window bridges it.
- Acquisition IDs are prefixed "ext::" so they can never collide with simulated runs.

Contains no Streamlit import (unit-testable, same reasoning as dpc_upload_validation.py).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driveflow.models.common.acquisitions import GROUP_COL, TIME_COL, check_acquisitions

#: Same generic "clearly wrong data" sentinel as viz.dpc_upload_validation.DEFAULT_MAGNITUDE_BOUND
#: (redeclared, not imported: library code must not depend on the viz layer).
DEFAULT_MAGNITUDE_BOUND = 1e4

#: Columns that identify / describe a row rather than measure something. Preserved verbatim
#: (outside the numeric channel block) whenever present.
METADATA_COLUMNS = (
    "label", GROUP_COL, TIME_COL, "dataset_id", "scenario_id", "plant_config_id", "fs_hz", "seed",
    "controller_type", "machine_id", "bearing_id", "sensor_id", "units", "origin",
)
REQUIRED_METADATA = {"classification": ("label", GROUP_COL, TIME_COL), "regression": (GROUP_COL, TIME_COL)}
#: Hard cap on rows accepted from one upload (AI-12): keeps a malformed / hostile file from
#: exhausting the server's memory. Generous relative to any dataset this project has used.
MAX_TRAINING_ROWS = 2_000_000

#: Per-channel plausibility bounds (AI-12) -- |value| above these is treated as a unit error /
#: sentinel, not physics. dc_motor bounds are 2x the simulated motor's OWN limits
#: (DcPermanentlyExcitedMotor().limits: i=210 A, omega=400 rad/s -> 3820 rpm, torque=38 Nm), so
#: they scale with the plant actually being modelled; the 2x margin is a proposed default that
#: REQUIRES RESEARCHER CONFIRMATION for real machines. Vibration and vsc_dpc channels have no
#: physical limit model in this repo, so they keep the generic sentinel bound (documented as such).
CHANNEL_BOUNDS = {
    "dc_motor": {
        "current_r": 420.0, "current_s": 420.0, "current_t": 420.0,
        "rpm": 7640.0, "torque_nm": 76.0,
        "acc_x": DEFAULT_MAGNITUDE_BOUND, "acc_y": DEFAULT_MAGNITUDE_BOUND, "acc_z": DEFAULT_MAGNITUDE_BOUND,
    },
    "vsc_dpc": {c: DEFAULT_MAGNITUDE_BOUND for c in ("vc_real", "vc_imag", "i_f_real", "i_f_imag")},
}


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (values + column names + order), for lineage."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


@dataclass
class TrainingDatasetReport:
    ok: bool
    task: str
    domain: str
    df: pd.DataFrame | None
    messages: list = field(default_factory=list)  # [(level, text)], level in error/warning/info
    usable_for: list = field(default_factory=list)  # subset of ["training", "inference"]
    channels: list = field(default_factory=list)
    missing_channels: list = field(default_factory=list)
    unknown_labels: list = field(default_factory=list)
    acquisitions: list = field(default_factory=list)  # AcquisitionCheck.to_dict() per group
    n_rows_in: int = 0
    n_rows_out: int = 0
    sha256: str | None = None

    def summary(self) -> dict:
        return {
            "ok": self.ok, "task": self.task, "domain": self.domain, "usable_for": self.usable_for,
            "n_rows_in": self.n_rows_in, "n_rows_out": self.n_rows_out, "channels": self.channels,
            "missing_channels": self.missing_channels, "unknown_labels": self.unknown_labels,
            "n_acquisitions": len(self.acquisitions),
            "n_acquisitions_rejected": sum(1 for a in self.acquisitions if not a["ok"]),
            "sha256": self.sha256,
        }


def validate_training_dataset(
    df_raw: pd.DataFrame,
    domain: str,
    task: str,
    required_channels: list,
    known_classes: list | None = None,
    expected_fs_hz: float | None = None,
    min_rows_per_acquisition: int = 1,
) -> TrainingDatasetReport:
    """See module docstring. `task` is "classification" or "regression"; `required_channels` /
    `known_classes` come from the base model's own metrics.json."""
    if task not in REQUIRED_METADATA:
        raise ValueError(f"task must be one of {sorted(REQUIRED_METADATA)}, got {task!r}")
    if domain not in CHANNEL_BOUNDS:
        raise ValueError(f"unknown domain {domain!r}")
    report = TrainingDatasetReport(ok=False, task=task, domain=domain, df=None, n_rows_in=int(len(df_raw)))
    msgs = report.messages

    if len(df_raw) > MAX_TRAINING_ROWS:
        msgs.append(("error", f"{len(df_raw)} rows exceeds the {MAX_TRAINING_ROWS:,}-row upload limit -- split the file."))
        return report

    # Positional index: a concatenated upload can carry duplicate index labels, which would
    # misalign the channel block and the metadata block when they are re-joined below.
    df = df_raw.reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    report.missing_channels = [c for c in required_channels if c not in df.columns]
    report.channels = [c for c in required_channels if c in df.columns]
    if report.channels:
        report.usable_for.append("inference")
    missing_meta = [c for c in REQUIRED_METADATA[task] if c not in df.columns]
    if missing_meta:
        hint = " It may still be usable for inference (AI Diagnostics upload)." if report.channels else ""
        msgs.append(("error", f"Not trainable for {task}: missing required column(s) {', '.join(missing_meta)} -- they are not filled in automatically.{hint}"))
    if report.missing_channels:
        msgs.append(("error", f"Missing channel(s) the base model was trained on: {', '.join(report.missing_channels)}."))
    if missing_meta or report.missing_channels:
        return report

    # Numeric channel block; metadata kept alongside, untouched.
    keep_meta = [c for c in METADATA_COLUMNS if c in df.columns]
    channels = list(report.channels)
    num = df[channels].apply(pd.to_numeric, errors="coerce")
    bad_numeric = ~np.isfinite(num.to_numpy(dtype=float)).all(axis=1)
    bounds = CHANNEL_BOUNDS[domain]
    out_of_range = np.zeros(len(df), dtype=bool)
    range_detail = []
    for c in channels:
        bound = bounds.get(c, DEFAULT_MAGNITUDE_BOUND)
        hit = (num[c].abs() > bound).to_numpy() & ~bad_numeric
        if hit.any():
            range_detail.append(f"{c} (|x| > {bound:g}: {int(hit.sum())} row(s))")
        out_of_range |= hit
    t = pd.to_numeric(df[TIME_COL], errors="coerce")
    bad_time = ~np.isfinite(t.to_numpy(dtype=float))

    drop = bad_numeric | out_of_range | bad_time
    if bad_numeric.any():
        msgs.append(("warning", f"Dropped {int(bad_numeric.sum())} row(s) with non-numeric/missing channel values -- the gap is treated as a discontinuity (no window crosses it)."))
    if out_of_range.any():
        msgs.append(("warning", "Dropped rows outside per-channel plausibility bounds: " + "; ".join(range_detail) + "."))
    if bad_time.any():
        msgs.append(("warning", f"Dropped {int(bad_time.sum())} row(s) with a non-numeric timestamp."))

    clean = pd.concat([num, df[keep_meta]], axis=1).loc[~drop].copy()
    clean[TIME_COL] = pd.to_numeric(clean[TIME_COL])
    clean[GROUP_COL] = "ext::" + clean[GROUP_COL].astype(str).str.removeprefix("ext::")
    if "label" in clean.columns:
        clean["label"] = clean["label"].astype(str).str.strip()

    if task == "classification" and known_classes is not None:
        report.unknown_labels = sorted(set(clean["label"]) - set(map(str, known_classes)))
        if report.unknown_labels:
            msgs.append((
                "error",
                f"Label(s) not in the base model's class vocabulary {sorted(map(str, known_classes))}: {report.unknown_labels}. "
                "The output layer has a fixed size; relabel or train a new model instead of fine-tuning.",
            ))
            return report
        if clean.groupby(GROUP_COL)["label"].nunique().max() > 1:
            msgs.append(("error", "At least one acquisition (source_file) carries more than one label -- split it into single-condition recordings first."))
            return report

    checks = check_acquisitions(clean, GROUP_COL, TIME_COL, expected_fs_hz=expected_fs_hz, min_rows=min_rows_per_acquisition)
    report.acquisitions = [c.to_dict() for c in checks]
    rejected = [c for c in checks if not c.ok]
    for c in rejected:
        msgs.append(("warning", f"Acquisition {c.group!r} rejected: {c.reason}."))
    gapped = [c for c in checks if c.ok and c.n_segments > 1]
    if gapped:
        msgs.append(("info", f"{len(gapped)} acquisition(s) contain gaps and will be windowed per contiguous segment."))
    clean = clean[clean[GROUP_COL].isin([c.group for c in checks if c.ok])].reset_index(drop=True)

    report.n_rows_out = int(len(clean))
    if clean.empty:
        msgs.append(("error", "No valid acquisition left after validation -- nothing to train on."))
        return report
    report.df = clean
    report.sha256 = dataframe_sha256(clean)
    report.ok = True
    report.usable_for.insert(0, "training")
    msgs.append(("info", f"{clean[GROUP_COL].nunique()} acquisition(s), {len(clean)} row(s) accepted for {task} training."))
    return report
