"""Surrogate plants fitted to PLECS sweep datasets -- task A1 (round PLECS/OM, route a).

Route a: PLECS RPC sweeps (task E2) → dataset under data/plecs/<profile>/<run_id>/ → surrogate
(the E5 families of envs/surrogate_env.py) → ``SPADE/<Profile>Surrogate-v0``.

Inputs: a sweep run of a ``plecs_rpc_batch`` profile whose manifest declares
- ``excitation``: the duty profile the circuit applied (constant ModelVars only give step responses)
  and the outport that recorded the duty applied, which the dataset keeps as a column;
- ``surrogate``: template ``buck``, i.e. which outputs are i_L and v_out and which ModelVars are V_in
  and R.

Sampling: the surrogate steps at the native buck's control step (TAU = 20 µs), so the sweep must
return samples exactly every TAU (``n_output_points = TimeSpan / TAU + 1``). A dataset with another
spacing is refused, not resampled.

Split: by OPERATING POINT (V_in, R), never by window. All excitation seeds of a point stay in one
split. Points are assigned with a seeded permutation: validation and test each get ≥ 1 point
(~20 % each), the rest train. Families are fitted on train; the default family is chosen on
validation; test is reported only.

Provenance: the dataset is SIMULATED · PLECS; the fitted surrogate and its environment are
SIMULATED · SURROGATE and record the source profile, the circuit's SHA-256 and the sweep run_id.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from driveflow.envs import surrogate_env as se
from driveflow.integrations.plecs import excitation as ex
from driveflow.integrations.plecs import manifest as mf
from driveflow.sim.buck_system import TAU

ORIGIN = "SIMULATED · SURROGATE"
SOURCE_ORIGIN = "SIMULATED · PLECS"
VAL_FRAC, TEST_FRAC = 0.2, 0.2


class SurrogateSourceError(ValueError):
    pass


def surrogate_root() -> Path:
    """Fitted PLECS surrogates: $SPADE_SURROGATE_DATA, else <repo>/data/surrogates/plecs (git-ignored)."""
    env = os.environ.get("SPADE_SURROGATE_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "data" / "surrogates" / "plecs"


def env_id_for(profile_id: str) -> str:
    camel = "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]+", profile_id) if part)
    return f"SPADE/{camel}Surrogate-v0"


# ----------------------------------------------------------------------------- dataset
def load_dataset(dataset_dir) -> tuple[dict, list[dict]]:
    """(manifest.json of the sweep, records) for every COMPLETED condition. Each record:
    {condition_index, v_in_v, load_resistance_ohm, model_vars, duty (n,), x (n+1, 2) [i_L, v_out],
    consistency}. Raises SurrogateSourceError with what to change."""
    d = Path(dataset_dir)
    meta = json.loads((d / "manifest.json").read_text())
    man = mf.from_dict(meta["interface"])
    if man.excitation is None or man.surrogate is None:
        raise SurrogateSourceError("this sweep's manifest has no 'excitation' and 'surrogate' blocks: it cannot feed a surrogate "
                                   "(constant ModelVars only give step responses; see docs/guides/PLECS_RESEARCHER_KIT.md)")
    conds = pd.read_csv(d / "conditions.csv")
    ok = conds[conds["status"] == "ok"]
    if ok.empty:
        raise SurrogateSourceError("the sweep has no completed condition")
    sm, exc = man.surrogate, man.excitation
    records = []
    for _, row in ok.iterrows():
        df = pd.read_parquet(d / row["file"])
        t = df["timestamp_s"].to_numpy(float)
        steps = np.diff(t)
        if len(t) < 3 or not np.allclose(steps, TAU, rtol=1e-6, atol=1e-12):
            raise SurrogateSourceError(f"condition {int(row['condition_index'])}: samples must be exactly every {TAU:g} s (the surrogate's control step); "
                                       f"set n_output_points = TimeSpan / {TAU:g} + 1. Got spacing {float(np.median(steps)) if len(steps) else float('nan'):g} s")
        mv = {v.name: float(df[v.name].iloc[0]) for v in man.model_vars if v.name in df.columns}
        applied = df[exc["action_output"]].to_numpy(float)
        x = np.column_stack([df[sm["states"]["i_l"]].to_numpy(float), df[sm["states"]["v_out"]].to_numpy(float)])
        records.append({"condition_index": int(row["condition_index"]), "v_in_v": mv[sm["conditions"]["v_in_v"]],
                        "load_resistance_ohm": mv[sm["conditions"]["load_resistance_ohm"]], "model_vars": mv,
                        "duty": applied[:-1], "x": x, "consistency": ex.consistency(exc, mv, t, applied)})
    return meta, records


def assign_splits(records: list[dict], seed: int = 0) -> dict:
    """(V_in, R) -> split, by a seeded permutation of the distinct operating points."""
    points = sorted({(r["v_in_v"], r["load_resistance_ohm"]) for r in records})
    if len(points) < 3:
        raise SurrogateSourceError(f"{len(points)} operating point(s): at least 3 are needed (train, validation, test)")
    order = np.random.default_rng(seed).permutation(len(points))
    n_val = max(1, int(round(VAL_FRAC * len(points))))
    n_test = max(1, int(round(TEST_FRAC * len(points))))
    split = {}
    for rank, i in enumerate(order):
        split[points[i]] = "validation" if rank < n_val else ("test" if rank < n_val + n_test else "train")
    for r in records:
        r["split"] = split[(r["v_in_v"], r["load_resistance_ohm"])]
    return split


def _rel(a, b):
    return se._rel_rmse(a, b)


def fit(dataset_dir, seed: int = 0, epochs: int = 30, out_root: Path | None = None) -> tuple[Path, dict]:
    """Fits both E5 families on the train points, picks the default on validation, scores every
    condition (open-loop rollout of the surrogate under the duty the circuit applied, against the
    circuit's own trajectory) and writes <root>/<profile>/<date>_runNN/surrogate.json."""
    meta, records = load_dataset(dataset_dir)
    split = assign_splits(records, seed)
    train = [r for r in records if r["split"] == "train"]
    fits = {"zoh": se.fit_zoh(train), "mlp": se.fit_mlp(train, seed=seed, epochs=epochs)}
    tr = se.transitions(train)
    domain = {"v_in_v": [float(tr["vin"].min()), float(tr["vin"].max())], "load_resistance_ohm": [float(tr["r"].min()), float(tr["r"].max())],
              "duty": [float(tr["d"].min()), float(tr["d"].max())], "i_l_a": [float(tr["x"][:, 0].min()), float(tr["x"][:, 0].max())],
              "v_out_v": [float(tr["x"][:, 1].min()), float(tr["x"][:, 1].max())]}
    surrogate = {"source": f"plecs:{meta['profile_id']}", "origin": ORIGIN, "source_origin": SOURCE_ORIGIN, "domain": domain,
                 "families": {f: {"fit": fits[f]} for f in se.FAMILIES}, "seed": seed,
                 "provenance": {"plecs_profile": meta["profile_id"], "profile_version": meta.get("profile_version"), "circuit_name": meta.get("circuit_name"),
                                "circuit_sha256": meta["circuit_sha256"], "sweep_run_id": meta["run_id"], "dataset_dir": str(Path(dataset_dir)),
                                "excitation": mf.from_dict(meta["interface"]).excitation},
                 "conditions": {f"{v:g}V_{r:g}ohm": s for (v, r), s in split.items()}}
    rows = []
    for rec in records:
        for fam in se.FAMILIES:
            plant = se.make_plant(surrogate, fam, rec["v_in_v"], rec["load_resistance_ohm"])
            rel = _rel(se.rollout(plant, rec["duty"]), rec["x"])
            rows.append({"condition_index": rec["condition_index"], "v_in_v": rec["v_in_v"], "load_resistance_ohm": rec["load_resistance_ohm"],
                         "prbs_seed": rec["model_vars"].get("prbs_seed"), "split": rec["split"], "family": fam,
                         "rollout_rel_rmse_i_l": rel[0], "rollout_rel_rmse_v_out": rel[1], "rollout_finite": rel[0] is not None,
                         "action_consistency_max_abs": rec["consistency"]["max_abs_error"]})

    def med(fam, sp, key):
        vals = [r[key] for r in rows if r["family"] == fam and r["split"] == sp and r[key] is not None]
        return float(np.median(vals)) if vals else None

    for fam in se.FAMILIES:
        surrogate["families"][fam]["validation"] = {k: med(fam, "validation", k) for k in ("rollout_rel_rmse_v_out", "rollout_rel_rmse_i_l")}
        surrogate["families"][fam]["test"] = {k: med(fam, "test", k) for k in ("rollout_rel_rmse_v_out", "rollout_rel_rmse_i_l")}
    score = {f: surrogate["families"][f]["validation"]["rollout_rel_rmse_v_out"] for f in se.FAMILIES}
    surrogate["default_family"] = min(score, key=lambda f: np.inf if score[f] is None else score[f])
    surrogate["default_rule"] = "lower median relative v_out rollout RMSE on the validation operating points"
    surrogate["rows"] = rows
    root = Path(out_root) if out_root else surrogate_root()
    base = root / meta["profile_id"]
    n = 1
    while (base / f"{dt.date.today().isoformat()}_run{n:02d}").exists():
        n += 1
    out = base / f"{dt.date.today().isoformat()}_run{n:02d}"
    out.mkdir(parents=True)
    (out / "surrogate.json").write_text(json.dumps(surrogate, indent=1, default=float))
    return out, surrogate


# ----------------------------------------------------------------------------- environments
def _index_path(root: Path | None = None) -> Path:
    return (Path(root) if root else surrogate_root()) / "published.json"


def publish(profile_id: str, surrogate_dir, root: Path | None = None) -> str:
    """Registers SPADE/<Profile>Surrogate-v0 on the fitted surrogate and records it so later
    processes register it too (envs.registry.register_all). Returns the env ID."""
    from driveflow.envs import registry

    env_id = env_id_for(profile_id)
    idx = _index_path(root)
    data = json.loads(idx.read_text()) if idx.exists() else {}
    data[env_id] = {"profile_id": profile_id, "surrogate_dir": str(Path(surrogate_dir).resolve())}
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(data, indent=2))
    registry.register_dynamic(env_id, profile_id, str(Path(surrogate_dir).resolve()))
    return env_id


def published(root: Path | None = None) -> dict:
    idx = _index_path(root)
    if not idx.exists():
        return {}
    try:
        return {k: v for k, v in json.loads(idx.read_text()).items() if (Path(v["surrogate_dir"]) / "surrogate.json").exists()}
    except (OSError, json.JSONDecodeError, KeyError):
        return {}
