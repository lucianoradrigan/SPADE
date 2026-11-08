"""AI-N28 repair round of the ESP32 int8 classifier. Pre-registration (approved, committed before this
script): docs/audit/assistant/esp32_repair_preregistration.md.

    python experiments/report_esp32_repair.py prepare --work <dir>                     # R1-R3, stage-1 exports V0-V5
    python experiments/report_esp32_repair.py export --work <dir> --keras K1=<run> K2=<run>   # stage-2 exports V6-V17
    python experiments/report_esp32_repair.py select --work <dir>                      # validation of every variant, the fixed rule
    python experiments/report_esp32_repair.py test --work <dir>                        # the ONE test of the selected variant
Variants are run-like directories under <dir>/variants/V<k> (metrics.json, config.yaml, run_metadata.json and
weights copied from their Keras run, plus the variant's model.tflite); nothing in configs/classifiers is written.
Report: docs/audit/assistant/esp32_repair.{md,json}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/home/seven/post_doc/plataform_proposed/SPADE-dev/data") if not (REPO / "data" / "diagnosis_grid_dataset_dcm.parquet").exists() else REPO / "data"
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import report_ai_n27_regen as RG  # noqa: E402

PREREG = "docs/audit/assistant/esp32_repair_preregistration.md"
OUT = REPO / "docs" / "audit" / "assistant" / "esp32_repair"
PROMOTED = "configs/classifiers/esp32_dscnn/2026-10-18_run01"
K0 = "configs/classifiers/esp32_dscnn/2026-11-04_run01"
GRID = DATA / "diagnosis_grid_dataset_dcm.parquet"
E2A_OLD = DATA / "ai_n27_heldout_3class_dcm.parquet"
E2A_NEW = DATA / "esp32_repair_heldout_3class_dcm.parquet"
EVAL_CACHE = DATA / "eval_cache_5class_dcm.parquet"
N_CAL = 1000
CAL_SPLIT = (334, 333, 333)  # inner_race / normal / outer_race
CALIBRATIONS = ("C0", "C1", "C2")
QUANT = (("Qc", True), ("Qt", False))


def _variants(keras: dict) -> list:
    """[(Vk, keras key, calibration, quant label, per_channel)] in the pre-registered numbering."""
    out, k = [], 0
    for kk in ("K0", "K1", "K2"):
        if kk not in keras:
            k += 6
            continue
        for cal in CALIBRATIONS:
            for ql, pc in QUANT:
                out.append((f"V{k}", kk, cal, ql, pc))
                k += 1
    return out


def _hash(g) -> str:
    return hashlib.sha256(str(g).encode()).hexdigest()[:12]


def _keras(run: str):
    from driveflow.models.classifiers.builder import build_classifier
    from driveflow.models.classifiers.schemas import load_classifier_config

    d = REPO / run if not Path(run).is_absolute() else Path(run)
    m = json.loads((d / "metrics.json").read_text())
    model = build_classifier(load_classifier_config(d / "config.yaml"), n_channels=len(m["channels"]))
    model.load_weights(d / "model.weights.h5")
    return model, m


def _grid_windows(info: dict):
    df = pd.read_parquet(GRID)
    X, y, g = RG._windows(df, sorted(df["label"].unique()), info, info["cap"])
    return df, X, y, g


def _calibration(cal: str, run: str, info: dict, grid) -> np.ndarray:
    df, X, y, g = grid
    if cal == "C0":  # the current recipe, exactly (experiments/export_tflite._representative_windows)
        from experiments.export_tflite import _representative_windows
        from driveflow.models.classifiers.schemas import load_classifier_config

        d = REPO / run if not Path(run).is_absolute() else Path(run)
        return _representative_windows("dc_motor", "classifier", GRID, load_classifier_config(d / "config.yaml"), info["metrics"], N_CAL)
    train = set(info["metrics"]["split"]["group_hashes"]["train"])
    keep = np.array([_hash(x) in train for x in g])
    if cal == "C2":
        noise = df.groupby("source_file")[["grid_electrical_noise_pct", "grid_mechanical_noise_pct"]].first()
        free = set(noise.index[(noise["grid_electrical_noise_pct"] == 0) & (noise["grid_mechanical_noise_pct"] == 0)])
        keep &= np.isin(g, list(free))
    rng = np.random.default_rng(0)
    parts, counts = [], {}
    for cls, n in zip(info["metrics"]["classes"], CAL_SPLIT):
        idx = np.flatnonzero(keep & (y == cls))
        take = idx if len(idx) <= n else rng.choice(idx, size=n, replace=False)
        counts[cls] = int(len(take))
        parts.append(take)
    sel = np.concatenate(parts)
    return X[rng.permutation(sel)], counts


def _export(work: Path, vk: str, run: str, cal_X: np.ndarray, per_channel: bool) -> Path:
    from driveflow.ai.tflite_export import export_int8

    src = REPO / run if not Path(run).is_absolute() else Path(run)
    d = work / "variants" / vk
    d.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.json", "config.yaml", "run_metadata.json", "model.weights.h5"):
        shutil.copy2(src / f, d / f)
    model, _m = _keras(run)
    export_int8(model, cal_X, d / "model.tflite", n_calibration_samples=len(cal_X), per_channel=per_channel)
    return d


def _scales_per_weight_tensor(tflite: Path) -> int:
    """Max number of quantization scales over the weight (constant, int8) tensors of a model."""
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(tflite))
    n = 1
    for t in it.get_tensor_details():
        q = t.get("quantization_parameters") or {}
        if t["dtype"] == np.int8 and len(t["shape"]) >= 2 and q.get("scales") is not None:
            n = max(n, len(q["scales"]))
    return n


def _load_state(work: Path) -> dict:
    p = work / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"keras": {"K0": K0}, "variants": {}, "log": []}


def _save_state(work: Path, st: dict) -> None:
    (work / "state.json").write_text(json.dumps(st, indent=2, default=str) + "\n")


def _val_windows(info, grid):
    _df, X, y, g = grid
    val = set(info["metrics"]["split"]["group_hashes"]["val"])
    m = np.array([_hash(x) in val for x in g])
    return X[m], y[m], g[m]


def cmd_prepare(work: Path) -> None:
    import tensorflow as tf

    st = _load_state(work)
    info = RG._model_info(K0)
    grid = _grid_windows(info)
    # R3: validation windows
    Xv, yv, gv = _val_windows(info, grid)
    r3 = {"n_val_windows": int(len(Xv)), "expected": info["metrics"]["split"]["n_windows"]["val"],
          "groups_match": sorted({_hash(x) for x in gv}) == sorted(info["metrics"]["split"]["group_hashes"]["val"])}
    r3["pass"] = r3["n_val_windows"] == r3["expected"] and r3["groups_match"]
    st["R3"] = r3
    if not r3["pass"]:
        _save_state(work, st)
        raise SystemExit(f"R3 failed: {r3}")
    # R2: the per-tensor switch exists and gives one scale per weight tensor
    r2 = {"tensorflow": tf.__version__}
    cal0 = _calibration("C0", K0, info, grid)
    d_c = _export(work, "V0", K0, cal0, True)
    d_t = _export(work, "V1", K0, cal0, False)
    r2["scales_per_channel_export"] = _scales_per_weight_tensor(d_c / "model.tflite")
    r2["scales_per_tensor_export"] = _scales_per_weight_tensor(d_t / "model.tflite")
    r2["pass"] = r2["scales_per_tensor_export"] == 1 and r2["scales_per_channel_export"] > 1
    st["R2"] = r2
    # R1: V0 reproduces the committed candidate's int8 outputs on every validation window
    sys.path.insert(0, str(REPO / "experiments"))
    from report_tflite_parity import _tflite_predict

    a, _ = _tflite_predict(REPO / K0 / "model.tflite", Xv)
    b, _ = _tflite_predict(d_c / "model.tflite", Xv)
    r1 = {"identical_outputs": bool(np.array_equal(a, b)), "max_abs_diff": float(np.max(np.abs(a - b))),
          "sha256_committed": hashlib.sha256((REPO / K0 / "model.tflite").read_bytes()).hexdigest(),
          "sha256_reexport": hashlib.sha256((d_c / "model.tflite").read_bytes()).hexdigest()}
    r1["pass"] = r1["identical_outputs"]
    st["R1"] = r1
    st["variants"]["V0"] = {"keras": "K0", "calibration": "C0", "quant": "Qc", "dir": str(d_c)}
    st["variants"]["V1"] = {"keras": "K0", "calibration": "C0", "quant": "Qt", "dir": str(d_t)}
    _save_state(work, st)
    if not (r1["pass"] and r2["pass"]):
        raise SystemExit(f"precondition failed: R1 {r1['pass']}, R2 {r2['pass']} (see {work / 'state.json'})")
    _export_all(work, st, {"K0": K0}, grid_cache={"K0": grid})
    print(json.dumps({k: st[k] for k in ("R1", "R2", "R3")}, indent=1))


def _export_all(work: Path, st: dict, keras: dict, grid_cache: dict | None = None) -> None:
    grid_cache = grid_cache or {}
    for vk, kk, cal, ql, pc in _variants(keras):
        if vk in st["variants"]:
            continue
        run = keras[kk]
        info = RG._model_info(run)
        grid = grid_cache.get(kk) or _grid_windows(info)
        grid_cache[kk] = grid
        res = _calibration(cal, run, info, grid)
        cal_X, counts = (res, None) if cal == "C0" else res
        d = _export(work, vk, run, cal_X, pc)
        st["variants"][vk] = {"keras": kk, "calibration": cal, "quant": ql, "dir": str(d), "calibration_counts": counts}
        _save_state(work, st)
        print("exported", vk, kk, cal, ql, flush=True)


def cmd_export(work: Path, keras_args: list) -> None:
    st = _load_state(work)
    for x in keras_args:
        k, run = x.split("=", 1)
        st["keras"][k] = run
    # P3 of the AI-N27 round: the same split as the candidate
    base = RG._model_info(K0)["metrics"]["split"]["group_hashes"]
    for k, run in st["keras"].items():
        gh = RG._model_info(run)["metrics"]["split"]["group_hashes"]
        if gh != base:
            raise SystemExit(f"{k} ({run}) has another split: stop (pre-registration section 3)")
    _save_state(work, st)
    _export_all(work, st, st["keras"])


def _score_dir(d: str, X, y_str, classes) -> dict:
    keep = np.isin(y_str, classes)
    y = np.array([classes.index(c) for c in y_str[keep]])
    run = str(Path(d))
    P = RG._predict(run, "tflite", X[keep])
    thr = json.loads((Path(d) / "metrics.json").read_text())["calibration"]["abstain_below"]
    return RG._score(P, y, classes, thr)


def cmd_select(work: Path) -> None:
    st = _load_state(work)
    info = RG._model_info(K0)
    grid = _grid_windows(info)
    Xv, yv, _gv = _val_windows(info, grid)
    classes = info["metrics"]["classes"]
    rows = {}
    for vk, v in sorted(st["variants"].items(), key=lambda kv: int(kv[0][1:])):
        s = _score_dir(v["dir"], Xv, yv, classes)
        s["min_recall"] = min(r for r in s["recall"].values() if r is not None)
        rows[vk] = {**v, "validation": s}
        print(vk, v["keras"], v["calibration"], v["quant"], f"min recall {s['min_recall']:.3f} macro-F1 {s['macro_f1']:.3f}", flush=True)
    ranked = sorted(rows, key=lambda k: (-rows[k]["validation"]["min_recall"], -rows[k]["validation"]["macro_f1"], int(k[1:])))
    st["validation"] = rows
    st["ranking"] = ranked
    st["selected"] = ranked[0]
    st["selected_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(work, st)
    print("selected", ranked[0])


def cmd_test(work: Path) -> None:
    st = _load_state(work)
    if "tested" in st:
        raise SystemExit("the selected variant was already tested once (pre-registration section 5)")
    if not E2A_NEW.exists():
        raise SystemExit(f"E2a' missing: {E2A_NEW}")
    from report_classifier_evaluation import _SHARES_NOMINAL, CONDITIONS

    vk = st["selected"]
    v = st["variants"][vk]
    sel_dir = Path(v["dir"])
    prom = RG._model_info(PROMOTED)
    cand = {"run": str(sel_dir), "metrics": json.loads((sel_dir / "metrics.json").read_text()), "window": prom["window"], "cap": prom["cap"]}
    classes = prom["metrics"]["classes"]
    t = {"promoted": PROMOTED, "selected": vk, "variant": v, "sets": {}}
    grid_df = pd.read_parquet(GRID)
    X, y, g = RG._windows(grid_df, sorted(grid_df["label"].unique()), prom, prom["cap"])
    test = set(prom["metrics"]["split"]["group_hashes"]["test"])
    m = np.array([_hash(x) in test for x in g])
    t["sets"]["E1"] = RG._compare(prom, cand, "tflite", X[m], y[m], g[m], classes)
    for name, path in (("E2a", E2A_NEW), ("E2a_old_descriptive", E2A_OLD)):
        hd = pd.read_parquet(path)
        Xh, yh, gh = RG._windows(hd, classes, prom, 100_000)
        t["sets"][name] = RG._compare(prom, cand, "tflite", Xh, yh, gh, classes)
    ev = pd.read_parquet(EVAL_CACHE)
    parts = []
    for cond, (_o, _s, win) in CONDITIONS.items():
        df = ev[ev["condition"] == _SHARES_NOMINAL.get(cond, cond)]
        df = df[df["timestamp_s"] >= RG.SETTLE_S] if win == "steady" else df[df["timestamp_s"] < RG.SETTLE_S]
        Xc, yc, gc = RG._windows(df, classes, prom, 10_000)
        parts.append((Xc, yc, np.array([f"{cond}/{x}" for x in gc])))
    t["sets"]["E3"] = RG._compare(prom, cand, "tflite", np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), classes)
    t["checks"] = RG._checks(t, classes)
    t["verdict"] = "MEETS THE CRITERION" if all(c["pass"] for c in t["checks"].values()) else "DOES NOT MEET THE CRITERION"
    st["tested"] = t
    st["tested_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(work, st)
    res = {"command": "python experiments/report_esp32_repair.py prepare|export|select|test --work <dir>", "preregistration": PREREG, "host": platform.platform(),
           "python": platform.python_version(), "rule": RG.RULE, **{k: st.get(k) for k in ("R1", "R2", "R3", "keras", "ranking", "selected", "selected_at", "tested_at")},
           "validation": st["validation"], "test": t}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(t["verdict"])
    print(json.dumps(t["checks"], indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prepare", "export", "select", "test"))
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--keras", nargs="*", default=[])
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    {"prepare": lambda: cmd_prepare(a.work), "export": lambda: cmd_export(a.work, a.keras), "select": lambda: cmd_select(a.work), "test": lambda: cmd_test(a.work)}[a.step]()


if __name__ == "__main__":
    main()
