"""Trained envelope of a classifier (AI-N40 correction, decision D1, 2026-11-16).

A dc_motor classifier is only evaluated inside the speed range it was trained on. Each promoted classifier DECLARES that
range in its audit_findings.json: {"trained_envelope": {"speed_rad_s": [min, max], "source": "..."}}. For any other run
(custom-trained, candidates, earlier runs) the range is deduced from the grid_speed column of the dataset named in its
run_metadata.json, read-only: the run's weights, metrics and metadata are never rewritten. Without a declaration and
without that dataset (a fresh clone has no data/) the envelope is unknown, and the GUI says so instead of guessing.

The rule is deterministic and lives here, in Python: inside -> the prediction is shown as before; outside -> no
prediction at all (no class, no bars, no confidence), only OUTSIDE_MESSAGE.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from driveflow.ai.audit_findings import FINDINGS_FILE
from driveflow.ai.registry import REPO_ROOT

OUTSIDE_MESSAGE = "Outside the trained range ({lo:.0f}–{hi:.0f} rad/s): no diagnosis"
#: tolerance on the boundary: a setpoint of 220.0 rad/s against a trained maximum of 220.0 is inside
EPS_RAD_S = 1e-6


def declared_envelope(run_dir) -> dict | None:
    path = Path(run_dir) / FINDINGS_FILE
    try:
        env = json.loads(path.read_text(encoding="utf-8")).get("trained_envelope")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if isinstance(env, dict) and isinstance(env.get("speed_rad_s"), list) and len(env["speed_rad_s"]) == 2:
        return {"speed_rad_s": [float(env["speed_rad_s"][0]), float(env["speed_rad_s"][1])], "source": env.get("source", ""), "declared": True}
    return None


@lru_cache(maxsize=64)
def _dataset_speed_range(dataset: str) -> tuple | None:
    import pyarrow.parquet as pq

    path = Path(dataset)
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        col = pq.read_table(path, columns=["grid_speed"]).column("grid_speed").to_pylist()
    except Exception:  # noqa: BLE001 -- no dataset (fresh clone) or no grid_speed column: unknown, never guessed
        return None
    vals = [float(v) for v in col if v is not None and float(v) > 0]
    return (min(vals), max(vals)) if vals else None


def deduced_envelope(run_dir) -> dict | None:
    try:
        meta = json.loads((Path(run_dir) / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ds = meta.get("dataset_path")
    rng = _dataset_speed_range(str(ds)) if ds else None
    if rng is None:
        return None
    return {"speed_rad_s": [rng[0], rng[1]], "source": f"deduced from the grid_speed column of {ds} (run_metadata.json)", "declared": False}


def trained_envelope(run_dir) -> dict | None:
    return declared_envelope(run_dir) or deduced_envelope(run_dir)


def check(run_dir, speed_rad_s: float | None) -> dict:
    """{"status": inside | outside | unknown_envelope | unknown_speed, "envelope": [lo, hi] | None, "speed_rad_s", "message"}"""
    env = trained_envelope(run_dir)
    out = {"envelope": env["speed_rad_s"] if env else None, "source": env["source"] if env else None, "speed_rad_s": speed_rad_s}
    if env is None:
        return {**out, "status": "unknown_envelope", "message": "Trained speed range unknown for this run: the prediction is shown without that check"}
    lo, hi = env["speed_rad_s"]
    if speed_rad_s is None:
        return {**out, "status": "unknown_speed", "message": f"Operating speed unknown: the trained range ({lo:.0f}–{hi:.0f} rad/s) could not be checked"}
    if lo - EPS_RAD_S <= float(speed_rad_s) <= hi + EPS_RAD_S:
        return {**out, "status": "inside", "message": ""}
    return {**out, "status": "outside", "message": OUTSIDE_MESSAGE.format(lo=lo, hi=hi)}


def speed_of(df, operating_point: dict | None = None) -> float | None:
    """The simulated condition's speed: the speed setpoint when known, else the mean rpm of the last 20 % of the run."""
    if operating_point and operating_point.get("omega_ref_rad_s") is not None:
        return float(operating_point["omega_ref_rad_s"])
    if df is not None and "rpm" in getattr(df, "columns", []) and df["rpm"].notna().any():
        import math

        tail = df["rpm"].iloc[int(len(df) * 0.8):]
        return float(tail.mean()) * 2 * math.pi / 60
    return None
