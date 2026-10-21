"""PLECS batch route: plant profiles, sweep design and one-condition runs -- expansion task E2
(spec Sec. 3.14.2, "Generate simulation datasets").

- A ``plecs_rpc_batch`` PlantProfile records the circuit's SHA-256, its name, where it lives on the
  backend's computer, the hand-written interface manifest and the RPC endpoint. If the file's hash
  changes, the profile is stale (runs/profiles.is_stale) and a sweep refuses to start.
- ``design_conditions`` builds a grid or a seeded Latin hypercube over ModelVars within the
  manifest's declared ranges.
- ``run_condition`` runs ONE condition and returns a table whose every row carries that condition's
  own ModelVars (regression for the Fund. notebook, which wrote combination 0's parameters on every
  row of Performance.csv).

This route never produces an interactive environment: a completed sweep is a dataset
("A completed PLECS sweep does not provide an interactive Gymnasium plant.").
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from driveflow.integrations.plecs import manifest as mf
from driveflow.integrations.plecs.rpc_client import DEFAULT_URL, PlecsRpcClient, check_local_url
from driveflow.runs.schemas import EnvironmentCapability, PlantProfile, utc_now
from driveflow.runs.store import sha256_file, workspace_root

KIND = "plecs_rpc_batch"
ENGINE = "plecs_rpc"
ORIGIN = "SIMULATED · PLECS"
PLANT_PREFIX = "plecs:"
METHODS = ("grid", "lhs")
MAX_CONDITIONS = 2000
_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")
NOT_GYM = "A completed PLECS sweep does not provide an interactive Gymnasium plant."


def data_root() -> Path:
    """Where sweep datasets are written: $SPADE_PLECS_DATA, else <repo>/data/plecs (git-ignored)."""
    env = os.environ.get("SPADE_PLECS_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[4] / "data" / "plecs"


def check_profile_id(profile_id: str) -> str:
    if not _PROFILE_ID.match(profile_id or ""):
        raise ValueError(f"profile id must be 2-63 characters of a-z, 0-9, '_' or '-', starting with a letter or digit; got {profile_id!r}")
    return profile_id


def store_uploaded_circuit(data: bytes, filename: str, root: str | os.PathLike | None = None) -> Path:
    """Copies an uploaded .plecs into <workspace>/plecs/circuits/<sha12>/<name> and returns the path.
    A circuit that references other files by relative path must be selected in place instead."""
    if not filename.lower().endswith(".plecs"):
        raise ValueError(f"expected a .plecs file, got {filename!r}")
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    target = workspace_root(root) / "plecs" / "circuits" / digest[:12] / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    return target


def create_profile(profile_id: str, circuit_path: str | os.PathLike, manifest: mf.InterfaceManifest, display_name: str | None = None,
                   rpc_url: str = DEFAULT_URL, version: int = 1) -> PlantProfile:
    """A new ``plecs_rpc_batch`` profile (status 'imported'). Batch capability stays False until a
    sweep has actually run (``mark_batch_shown``)."""
    check_profile_id(profile_id)
    path = Path(circuit_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".plecs":
        raise ValueError(f"circuit must be an existing .plecs file, got {path.name!r}")
    return PlantProfile(
        profile_id=profile_id, kind=KIND, engine=ENGINE, display_name=display_name or path.stem, origin=ORIGIN, file_hash=sha256_file(path),
        signals=tuple(manifest.output_names),
        parameters={"circuit_path": str(path), "circuit_name": path.name, "model_name": path.stem, "rpc_url": check_local_url(rpc_url),
                    "manifest": manifest.to_dict(), "created_utc": utc_now()},
        capabilities=EnvironmentCapability(diagnostic_messages=("Batch simulation not shown yet: run a sweep.", "Interactive stepping: not available over RPC (spec Sec. 3.15).")),
        status="imported", version=version)


def profile_manifest(profile: PlantProfile) -> mf.InterfaceManifest:
    return mf.from_dict(profile.parameters["manifest"])


def mark_batch_shown(profile: PlantProfile, run_id: str, n_ok: int, n_total: int) -> PlantProfile:
    """Next version with can_simulate_batch=True, citing the sweep run that showed it."""
    msg = f"Batch simulation shown by sweep run {run_id} ({n_ok}/{n_total} conditions completed)."
    caps = EnvironmentCapability(can_simulate_batch=True, validated_at=utc_now(),
                                 diagnostic_messages=(msg, "Interactive stepping: not available over RPC (spec Sec. 3.15)."))
    params = {**profile.parameters, "last_sweep_run": run_id}
    return dataclasses.replace(profile, capabilities=caps, parameters=params, version=profile.version + 1)


def design_conditions(manifest: mf.InterfaceManifest, ranges: dict, method: str = "grid", n_samples: int | None = None, seed: int = 0,
                      fixed: dict | None = None) -> list[dict]:
    """ModelVars dicts, one per condition.

    ranges: {name: (low, high, levels)} for grid, {name: (low, high)} for lhs; every swept range must
    lie inside the manifest's declared range. fixed: ModelVars held constant (else their default)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not ranges:
        raise ValueError("choose at least one ModelVar to sweep")
    fixed = manifest.check_values(dict(fixed or {}))
    base = {v.name: v.default for v in manifest.model_vars}
    base.update(fixed)
    names = list(ranges)
    bounds = []
    for name in names:
        lo, hi = float(ranges[name][0]), float(ranges[name][1])
        manifest.check_values({name: lo})
        manifest.check_values({name: hi})
        if not lo <= hi:
            raise ValueError(f"{name}: low {lo:g} is above high {hi:g}")
        bounds.append((lo, hi))
    if method == "grid":
        levels = []
        for name, (lo, hi) in zip(names, bounds):
            n = int(ranges[name][2]) if len(ranges[name]) > 2 else 3
            if n < 1:
                raise ValueError(f"{name}: levels must be >= 1")
            levels.append(np.linspace(lo, hi, n) if n > 1 else np.array([lo]))
        points = list(itertools.product(*levels))
    else:
        from scipy.stats import qmc

        if not n_samples or n_samples < 1:
            raise ValueError("lhs needs n_samples >= 1")
        unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n_samples)
        lo_arr, hi_arr = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        points = lo_arr + unit * (hi_arr - lo_arr)
    if len(points) > MAX_CONDITIONS:
        raise ValueError(f"{len(points)} conditions exceed the budget of {MAX_CONDITIONS}")
    return [{**base, **{n: float(x) for n, x in zip(names, p)}} for p in points]


def solver_options(time_span_s: float, n_output_points: int | None = None) -> dict:
    if not (np.isfinite(time_span_s) and time_span_s > 0):
        raise ValueError(f"simulated time must be > 0 s, got {time_span_s!r}")
    opts = {"TimeSpan": float(time_span_s)}
    if n_output_points:
        if n_output_points < 2:
            raise ValueError("output points must be >= 2")
        opts["OutputTimes"] = [float(x) for x in np.linspace(0.0, time_span_s, int(n_output_points))]
    return opts


def run_condition(client: PlecsRpcClient, model_name: str, manifest: mf.InterfaceManifest, model_vars: dict, time_span_s: float,
                  time_limit_s: float, n_output_points: int | None = None) -> tuple[pd.DataFrame, float]:
    """(table, wall seconds) of one condition: timestamp_s, the outputs, and this condition's own
    ModelVars on every row. Raises PlecsRpcError / ManifestError."""
    values = manifest.check_values(model_vars)
    res = client.simulate(model_name, manifest.output_names, values, solver_options(time_span_s, n_output_points), time_limit_s)
    df = pd.DataFrame({"timestamp_s": res.time, **res.outputs})
    for k, v in values.items():
        df[k] = v
    return df, res.wall_s
