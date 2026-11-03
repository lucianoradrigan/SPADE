"""AI-N27 option 2: evaluation of the four candidates trained on the dcm x1000 grids against the
promoted dc_motor classifiers, per tier. Pre-registered in
docs/audit/assistant/ai_n27_regen_preregistration.md (§3-§6; committed before any data existed).

Sets (each model's own window / channels / level scaling / calibration; windows from the training
builder; the tier's known classes only):
  E1   test split (same acquisitions for promoted and candidate, precondition P3) of the dcm grid;
  E2a  new held-out runs in the affected regime (data/ai_n27_heldout_<grid>_dcm.parquet), all windows;
  E2b  descriptive: the affected (S21-1) test-split acquisitions of the 5-class grid, dcm version;
  E3   regenerated generalization cache (data/eval_cache_5class_dcm.parquet), the 13 conditions pooled
       (per-condition and OOD descriptive).
Decision rule per tier (§6): D1 gate on E1 (candidate macro-F1 >= 0.88, recall >= 0.70 per class); on
each of E1, E2a and E3: delta macro-F1 >= -1 point, delta recall >= -5 points per class, |delta
abstention| <= 5 points. Deployed artifacts decide (PC Keras, RPi5 float16 TFLite, ESP32 int8 TFLite).

Usage: python experiments/report_ai_n27_regen.py --candidates PC3=<run> ESP32=<run> PC5=<run> RPI5=<run>
Writes docs/audit/assistant/ai_n27_regen.{md,json}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
OUT = REPO / "docs" / "audit" / "assistant" / "ai_n27_regen"
PREREG = "docs/audit/assistant/ai_n27_regen_preregistration.md"
DATA = {"3class": {"grid": "data/diagnosis_grid_dataset_dcm.parquet", "stored": "data/diagnosis_grid_dataset.parquet", "heldout": "data/ai_n27_heldout_3class_dcm.parquet"},
        "5class": {"grid": "data/diagnosis_grid_dataset_5class_dcm.parquet", "stored": "data/diagnosis_grid_dataset_5class.parquet", "heldout": "data/ai_n27_heldout_5class_dcm.parquet"}}
EVAL_CACHE = "data/eval_cache_5class_dcm.parquet"
#: tier -> (promoted run, candidate key, grid, deployed artifact)
TIERS = {"PC": ("configs/classifiers/pc_server_5class_w2048/2026-10-20_run01", "PC5", "5class", "keras"),
         "RPi5": ("configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01", "RPI5", "5class", "tflite"),
         "ESP32": ("configs/classifiers/esp32_dscnn/2026-10-18_run01", "ESP32", "3class", "tflite")}
RULE = {"gate_min_macro_f1": 0.88, "gate_min_recall": 0.70, "min_delta_macro_f1_points": -1.0, "min_delta_recall_points": -5.0, "max_abs_delta_abstention_points": 5.0}
DECIDING_SETS = ("E1", "E2a", "E3")
SETTLE_S = 0.5
N_BOOT, BOOT_SEED = 1000, 0


def _hash(g) -> str:
    return hashlib.sha256(str(g).encode()).hexdigest()[:12]


def _model_info(run: str) -> dict:
    import yaml

    d = REPO / run
    m = json.loads((d / "metrics.json").read_text())
    meta = json.loads((d / "run_metadata.json").read_text())
    return {"run": run, "metrics": m, "window": int(yaml.safe_load((d / "config.yaml").read_text())["input_window"]), "cap": int(meta.get("cap_per_class", 500)),
            "dataset": meta.get("dataset_path"), "dataset_sha256": meta.get("dataset_sha256")}


def _predict(run: str, kind: str, X: np.ndarray) -> np.ndarray:
    import report_s16_discretization_sensitivity as r2

    tag = f"{kind}@{run}"
    r2.MODELS[tag] = (run, kind)
    return r2._predict(tag, X)[0]  # calibrated probabilities


def _windows(df, classes, info, cap, regime=None):
    from driveflow.models.common import build_classification_windows

    if regime == "startup":
        df = df[df["timestamp_s"] < SETTLE_S]
    elif regime == "steady_state":
        df = df[df["timestamp_s"] >= SETTLE_S]
    m = info["metrics"]
    X, y, g = build_classification_windows(df, list(classes), m["channels"], info["window"], cap_per_class=cap, level_scaling=m.get("level_scaling"))
    if X is None:
        return np.zeros((0, info["window"], len(m["channels"])), np.float32), np.array([]), np.array([])
    return X, np.asarray(y), np.asarray(g)


def _macro_f1(y, pred, k):
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0))


def _score(P, y, classes, thr) -> dict:
    pred = P.argmax(1)
    k = len(classes)
    return {"n_windows": int(len(y)), "accuracy": float(np.mean(pred == y)), "macro_f1": _macro_f1(y, pred, k),
            "recall": {c: (float(np.mean(pred[y == i] == i)) if np.any(y == i) else None) for i, c in enumerate(classes)},
            "n_per_class": {c: int(np.sum(y == i)) for i, c in enumerate(classes)},
            "abstention": float(np.mean(P.max(1) < thr)),
            "confusion": [[int(np.sum((y == t) & (pred == q))) for q in range(k)] for t in range(k)]}


def _boot_delta(Pa, Pb, y, groups, k) -> list:
    rng = np.random.default_rng(BOOT_SEED)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    pa, pb = Pa.argmax(1), Pb.argmax(1)
    d = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        d.append(100.0 * (_macro_f1(y[sel], pb[sel], k) - _macro_f1(y[sel], pa[sel], k)))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def _compare(prom, cand, kind, X, y_str, groups, classes) -> dict:
    keep = np.isin(y_str, classes)
    X, y_str, groups = X[keep], y_str[keep], groups[keep]
    y = np.array([classes.index(c) for c in y_str])
    out = {"n_acquisitions": int(len(np.unique(groups)))}
    P = {}
    for role, info in (("promoted", prom), ("candidate", cand)):
        P[role] = _predict(info["run"], kind, X) if len(X) else np.zeros((0, len(classes)))
        out[role] = _score(P[role], y, classes, info["metrics"]["calibration"]["abstain_below"]) if len(X) else None
    if len(X):
        a, b = out["promoted"], out["candidate"]
        out["delta_points"] = {"macro_f1": 100.0 * (b["macro_f1"] - a["macro_f1"]), "accuracy": 100.0 * (b["accuracy"] - a["accuracy"]),
                               "abstention": 100.0 * (b["abstention"] - a["abstention"]),
                               "recall": {c: (None if a["recall"][c] is None else 100.0 * (b["recall"][c] - a["recall"][c])) for c in classes}}
        out["delta_macro_f1_ci95_points"] = _boot_delta(P["promoted"], P["candidate"], y, groups, len(classes)) if len(np.unique(groups)) > 1 else None
    return out


def _checks(tier_res: dict, classes: list) -> dict:
    c = {}
    e1c = tier_res["sets"]["E1"]["candidate"]
    c["gate_D1_macro_f1"] = {"value": e1c["macro_f1"], "pass": e1c["macro_f1"] >= RULE["gate_min_macro_f1"]}
    worst = min(v for v in e1c["recall"].values() if v is not None)
    c["gate_D1_min_recall"] = {"value": worst, "pass": worst >= RULE["gate_min_recall"]}
    for s in DECIDING_SETS:
        d = tier_res["sets"][s]["delta_points"]
        c[f"{s}_delta_macro_f1"] = {"value": d["macro_f1"], "pass": d["macro_f1"] >= RULE["min_delta_macro_f1_points"]}
        rec = {k: v for k, v in d["recall"].items() if v is not None}
        worst_cls = min(rec, key=rec.get)
        c[f"{s}_worst_delta_recall"] = {"class": worst_cls, "value": rec[worst_cls], "pass": rec[worst_cls] >= RULE["min_delta_recall_points"]}
        c[f"{s}_delta_abstention"] = {"value": d["abstention"], "pass": abs(d["abstention"]) <= RULE["max_abs_delta_abstention_points"]}
    return c


def main(argv=None) -> None:
    from report_s21_dataset_scan import load, scan_acquisitions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs=4, required=True, help="PC3=<run> ESP32=<run> PC5=<run> RPI5=<run>")
    args = ap.parse_args(argv)
    cand_runs = dict(x.split("=", 1) for x in args.candidates)
    t0 = time.time()
    frames = {g: {k: pd.read_parquet(REPO / p) for k, p in d.items() if k != "stored"} for g, d in DATA.items()}
    affected = {}
    for g, d in DATA.items():
        acq = scan_acquisitions(load(d["stored"]))
        affected[g] = set(acq.loc[acq["branch"], "source_file"])
    ev = pd.read_parquet(REPO / EVAL_CACHE)
    from report_classifier_evaluation import _SHARES_NOMINAL, CONDITIONS

    res = {"command": "python experiments/report_ai_n27_regen.py " + " ".join(argv if argv is not None else sys.argv[1:]), "preregistration": PREREG,
           "host": platform.platform(), "python": platform.python_version(), "rule": RULE, "candidates": cand_runs, "tiers": {}}
    for tier, (prom_run, ckey, grid, kind) in TIERS.items():
        prom, cand = _model_info(prom_run), _model_info(cand_runs[ckey])
        classes = prom["metrics"]["classes"]
        for k in ("classes", "channels", "level_scaling"):
            assert prom["metrics"].get(k) == cand["metrics"].get(k), f"{tier}: {k} differs between promoted and candidate"
        assert prom["window"] == cand["window"] and prom["cap"] == cand["cap"]
        t = {"promoted": prom_run, "candidate": cand["run"], "artifact": kind, "classes": classes,
             "candidate_dataset": cand["dataset"], "candidate_dataset_sha256": cand["dataset_sha256"], "sets": {}, "keras_descriptive": {}}
        grid_df = frames[grid]["grid"]
        # E1: the training windows (same cap and classes as training), test-split acquisitions only
        X, y, g = _windows(grid_df, sorted(grid_df["label"].unique()), prom, prom["cap"])
        test = set(prom["metrics"]["split"]["group_hashes"]["test"])
        assert test == set(cand["metrics"]["split"]["group_hashes"]["test"]), "P3 violated"
        m = np.array([_hash(x) in test for x in g])
        t["sets"]["E1"] = _compare(prom, cand, kind, X[m], y[m], g[m], classes)
        assert t["sets"]["E1"]["promoted"]["n_windows"] == cand["metrics"]["n_test"], "E1 windows differ from the candidate's recorded test set"
        e2b = m & np.isin(g, list(affected[grid]))
        t["sets"]["E2b"] = _compare(prom, cand, kind, X[e2b], y[e2b], g[e2b], classes) if e2b.any() else {"n_acquisitions": 0}
        if kind != "keras":
            t["keras_descriptive"]["E1"] = _compare(prom, cand, "keras", X[m], y[m], g[m], classes)
        t["candidate_recorded_test_macro_f1_keras"] = cand["metrics"]["test_macro_f1"]
        # E2a: held-out runs in the affected regime, all windows, plus regimes (descriptive)
        hd = frames[grid]["heldout"]
        X, y, g = _windows(hd, classes, prom, 100_000)
        t["sets"]["E2a"] = _compare(prom, cand, kind, X, y, g, classes)
        if kind != "keras":
            t["keras_descriptive"]["E2a"] = _compare(prom, cand, "keras", X, y, g, classes)
        for regime in ("startup", "steady_state"):
            Xr, yr, gr = _windows(hd, classes, prom, 100_000, regime)
            t["sets"][f"E2a_{regime}"] = _compare(prom, cand, kind, Xr, yr, gr, classes)
        # E3: 13 conditions pooled (each with its own time window), per condition descriptive
        parts = []
        for cond, (_o, _s, win) in CONDITIONS.items():
            df = ev[ev["condition"] == _SHARES_NOMINAL.get(cond, cond)]
            df = df[df["timestamp_s"] >= SETTLE_S] if win == "steady" else df[df["timestamp_s"] < SETTLE_S]
            Xc, yc, gc = _windows(df, classes, prom, 10_000)
            parts.append((cond, Xc, yc, np.array([f"{cond}/{x}" for x in gc])))
            t["sets"][f"E3_{cond}"] = _compare(prom, cand, kind, Xc, yc, gc, classes)
        X = np.concatenate([p[1] for p in parts])
        y = np.concatenate([p[2] for p in parts])
        g = np.concatenate([p[3] for p in parts])
        t["sets"]["E3"] = _compare(prom, cand, kind, X, y, g, classes)
        if kind != "keras":
            t["keras_descriptive"]["E3"] = _compare(prom, cand, "keras", X, y, g, classes)
        if tier == "ESP32":
            ood = ev[(ev["condition"] == "ood_unseen_fault_type") & (ev["timestamp_s"] >= SETTLE_S)]
            Xo, _yo, _go = _windows(ood, ["ball", "cage"], prom, 10_000)
            t["ood_ball_cage_abstention"] = {r: float(np.mean(_predict(i["run"], kind, Xo).max(1) < i["metrics"]["calibration"]["abstain_below"]))
                                             for r, i in (("promoted", prom), ("candidate", cand))}
        t["checks"] = _checks(t, classes)
        t["verdict"] = "MEETS THE CRITERION" if all(c["pass"] for c in t["checks"].values()) else "DOES NOT MEET THE CRITERION"
        res["tiers"][tier] = t
        print(tier, t["verdict"], flush=True)
    pc3 = _model_info(cand_runs["PC3"])["metrics"]
    res["teacher_pc3"] = {"run": cand_runs["PC3"], "test_accuracy": pc3["test_accuracy"], "test_macro_f1": pc3["test_macro_f1"], "n_test": pc3["n_test"]}
    res["elapsed_s"] = time.time() - t0
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")
    OUT.with_suffix(".md").write_text(_md(res))
    print(_md(res))


def _f(x, p=3):
    return "–" if x is None else f"{x:.{p}f}"


def _md(r: dict) -> str:
    L = ["# AI-N27 option 2 — candidates trained on the `dcm` ×1000 grids vs the promoted classifiers", "",
         f"Pre-registration: [`ai_n27_regen_preregistration.md`](ai_n27_regen_preregistration.md) (committed before any data existed). Command: `{r['command']}`. "
         f"Host: {r['host']}, Python {r['python']}.", "",
         "**Nothing is promoted.** Meeting the criterion is not a promotion; the decision is Luciano's.", "",
         "## Verdict per tier (pre-registered rule §6)", "",
         "| Tier | Artifact | Verdict | Failing checks |", "|---|---|---|---|"]
    for tier, t in r["tiers"].items():
        fails = [f"{k} ({_f(c['value'], 2)}{', ' + c['class'] if 'class' in c else ''})" for k, c in t["checks"].items() if not c["pass"]]
        L.append(f"| {tier} | {t['artifact']} | **{t['verdict']}** | {'; '.join(fails) or '—'} |")
    L += ["", "Δ = candidate − promoted, in points. Gate D1 on E1: candidate macro-F1 ≥ 0.88 and recall ≥ 0.70 per class. On E1, E2a, E3: Δ macro-F1 ≥ −1, "
          "Δ recall ≥ −5 per class, |Δ abstention| ≤ 5.", ""]
    for tier, t in r["tiers"].items():
        L += [f"## {tier}", "", f"Promoted `{t['promoted']}` vs candidate `{t['candidate']}` ({t['artifact']}; candidate dataset `{t['candidate_dataset']}`).", "",
              "| Set | Acq. | Windows | Macro-F1 prom. → cand. (Δ, 95 % CI) | Accuracy prom. → cand. | Abstention prom. → cand. | Worst Δ recall (class) |", "|---|---|---|---|---|---|---|"]
        for s, v in t["sets"].items():
            if not v.get("promoted"):
                L.append(f"| {s} | {v.get('n_acquisitions', 0)} | 0 | – | – | – | – |")
                continue
            a, b, d = v["promoted"], v["candidate"], v["delta_points"]
            rec = {k: x for k, x in d["recall"].items() if x is not None}
            wc = min(rec, key=rec.get) if rec else None
            ci = v.get("delta_macro_f1_ci95_points")
            L.append(f"| {'**' + s + '**' if s in DECIDING_SETS else s} | {v['n_acquisitions']} | {a['n_windows']} | {a['macro_f1']:.3f} → {b['macro_f1']:.3f} "
                     f"({d['macro_f1']:+.1f}{'' if not ci else f', [{ci[0]:+.1f}, {ci[1]:+.1f}]'}) | {a['accuracy']:.3f} → {b['accuracy']:.3f} | "
                     f"{a['abstention']:.3f} → {b['abstention']:.3f} | {'–' if wc is None else f'{rec[wc]:+.1f} ({wc})'} |")
        L += ["", "Per-class recall (decision sets):", "", "| Set | " + " | ".join(t["classes"]) + " |", "|---|" + "---|" * len(t["classes"])]
        for s in DECIDING_SETS + ("E2b",):
            v = t["sets"].get(s, {})
            if not v.get("promoted"):
                continue
            L.append(f"| {s} | " + " | ".join(f"{_f(v['promoted']['recall'][c])} → {_f(v['candidate']['recall'][c])}" for c in t["classes"]) + " |")
        if t.get("keras_descriptive"):
            L += ["", "Keras (descriptive, not deciding): " + "; ".join(f"{s} macro-F1 {v['promoted']['macro_f1']:.3f} → {v['candidate']['macro_f1']:.3f}" for s, v in t["keras_descriptive"].items()) + "."]
        if "ood_ball_cage_abstention" in t:
            o = t["ood_ball_cage_abstention"]
            L += ["", f"Unseen fault types (ball, cage; descriptive): abstention {o['promoted']:.3f} → {o['candidate']:.3f}."]
        L.append("")
    tp = r["teacher_pc3"]
    L += [f"Teacher (PC 3-class, no tier, no criterion): `{tp['run']}`, recorded test accuracy {tp['test_accuracy']:.3f}, macro-F1 {tp['test_macro_f1']:.3f} (n = {tp['n_test']}).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
