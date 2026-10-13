"""Acquisition (temporal group) integrity helpers -- the unit everything time-series in the AI
layer must respect (AUDIT AI-01/AI-03, AUDIT_ANSWERS.md Q01/Q02/Q05).

An *acquisition* is one contiguous recording: every row sharing one `source_file` value (one
simulated Scenario run, or one externally uploaded recording). Windowing slides over consecutive
rows of ONE acquisition and assumes those rows are consecutive samples in time; mixing, balancing
and splitting must therefore move whole acquisitions (or whole contiguous segments of one), never
individual rows -- a row-level sample of a time series is no longer a time series, and sorting it
back afterwards cannot recover the samples that were dropped in between.

These helpers only *check* and *describe*; they never reorder rows or resample. A group that fails
a check is reported (with the reason) so the caller can exclude it explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

GROUP_COL = "source_file"
TIME_COL = "timestamp_s"
#: Relative tolerance on the sampling interval: a step whose dt deviates from the group's median
#: dt by more than this fraction is treated as a discontinuity (gap, duplicate or jump). 1% is a
#: numerical-jitter tolerance for float timestamps, not a physical threshold.
DT_REL_TOL = 0.01


@dataclass(frozen=True)
class AcquisitionCheck:
    group: str
    n_rows: int
    fs_hz: float | None
    monotonic: bool
    n_segments: int
    ok: bool
    reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def contiguous_segments(t: np.ndarray, rel_tol: float = DT_REL_TOL) -> list:
    """Splits a strictly increasing timestamp vector into maximal runs of uniform sampling.

    Returns a list of (start, stop) index pairs (stop exclusive). A break is inserted wherever a
    step deviates from the median step by more than `rel_tol` (relative). For a clean recording the
    result is a single segment [(0, len(t))]."""
    n = len(t)
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]
    dt = np.diff(np.asarray(t, dtype=float))
    ref = float(np.median(dt))
    if not np.isfinite(ref) or ref <= 0:
        return [(i, i + 1) for i in range(n)]
    breaks = np.flatnonzero(np.abs(dt - ref) > rel_tol * ref) + 1
    edges = [0, *breaks.tolist(), n]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def window_starts(n_rows: int, seg_len: int, t: np.ndarray | None = None, rel_tol: float = DT_REL_TOL) -> list:
    """Start indices of non-overlapping windows of `seg_len` rows that each lie entirely inside one
    contiguous segment. Without timestamps (t is None) the whole group is one segment -- identical
    to the historical `range(0, n_rows // seg_len * seg_len, seg_len)` behaviour, which keeps
    previously trained models' windowing numerically unchanged on clean data."""
    if seg_len <= 0:
        raise ValueError("seg_len must be positive")
    segments = [(0, n_rows)] if t is None else contiguous_segments(t, rel_tol)
    starts = []
    for a, b in segments:
        starts.extend(range(a, b - seg_len + 1, seg_len))
    return starts


def check_acquisitions(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    time_col: str = TIME_COL,
    expected_fs_hz: float | None = None,
    min_rows: int = 1,
    rel_tol: float = DT_REL_TOL,
) -> list:
    """One AcquisitionCheck per group. A group is rejected (ok=False) if its timestamps are not
    strictly increasing (row order is NOT repaired -- see module docstring), if its sampling rate
    differs from `expected_fs_hz` by more than `rel_tol` (no resampling is implemented anywhere in
    this project, so a different rate cannot be made compatible silently), or if it has fewer than
    `min_rows` rows (cannot hold one window)."""
    if group_col not in df.columns:
        raise ValueError(f"missing acquisition column {group_col!r} -- rows cannot be assigned to a recording")
    if time_col not in df.columns:
        raise ValueError(f"missing time column {time_col!r} -- temporal continuity cannot be verified")
    checks = []
    for group, sub in df.groupby(group_col, sort=False):
        t = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
        n = len(t)
        if n < 2 or not np.all(np.isfinite(t)):
            finite = bool(np.all(np.isfinite(t)))
            ok = finite and n >= min_rows
            reason = None if ok else ("non-finite timestamps" if not finite else f"only {n} rows, fewer than one window ({min_rows})")
            checks.append(AcquisitionCheck(str(group), n, None, finite, 1 if n else 0, ok, reason))
            continue
        dt = np.diff(t)
        monotonic = bool(np.all(dt > 0))
        fs = float(1.0 / np.median(dt)) if monotonic else None
        segments = contiguous_segments(t, rel_tol) if monotonic else []
        reason = None
        if not monotonic:
            reason = "timestamps are not strictly increasing (rows out of order or duplicated) -- not repaired"
        elif expected_fs_hz is not None and abs(fs - expected_fs_hz) > rel_tol * expected_fs_hz:
            reason = f"sampling rate {fs:.1f} Hz != expected {expected_fs_hz:.1f} Hz (no resampling implemented)"
        elif n < min_rows:
            reason = f"only {n} rows, fewer than one window ({min_rows})"
        checks.append(AcquisitionCheck(str(group), n, fs, monotonic, len(segments), reason is None, reason))
    return checks
