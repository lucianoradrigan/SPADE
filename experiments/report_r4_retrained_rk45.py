"""R4 (round PLECS/OpenModelica; user decision 2026-10-25 after R2/R3): the dc_motor classifiers
retrained on the RK45 datasets (candidates), compared with the promoted Euler-trained runs.

Same recipe as the promoted runs (train_model.py --epochs 30 --batch-size 32 --seed 42
--keep-rpm-level --cap-per-class 3000; RPi5 and ESP32 distilled with alpha 0.5 from the matching
new PC teacher; RPi5 float16 / ESP32 int8 TFLite as before). Nothing is promoted.

Evaluated on:
1. each run's own recorded test split (metrics.json);
2. the R2 simulations (60 runs per version, seeds 5000+, never used for training): NAT (Euler plant)
   and REF (RK45 plant), start-up (t < 0.5 s) and steady state, accuracy and macro-F1 (ESP32: its
   own 3 classes), abstention;
3. PC models only: the conditions of experiments/report_classifier_evaluation.py on its cached
   evaluation runs (data/eval_cache_5class.parquet, Euler plant): held-out speeds and severity,
   noise, plant drift, start-up.

Usage: python experiments/report_r4_retrained_rk45.py --r2-cache <r2 simulations parquet>
           --candidates PC=<run> RPi5=<run> ESP32=<run> [--out docs/audit/plecs_om]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import report_s16_discretization_sensitivity as r2  # noqa: E402

PROMOTED = {"PC": "configs/classifiers/pc_server_5class_w2048/2026-10-20_run01",
            "RPi5": "configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-10-20_run01",
            "ESP32": "configs/classifiers/esp32_dscnn/2026-10-18_run01"}
KIND = {"PC": "keras", "RPi5": "tflite", "ESP32": "tflite"}


def _r2_eval(data, cells, key, run) -> dict:
    from sklearn.metrics import f1_score

    import yaml

    tag = f"{key}@{run}"
    r2.MODELS[tag] = (run, KIND[key])
    metrics = json.loads((REPO / run / "metrics.json").read_text())
    window = int(yaml.safe_load((REPO / run / "config.yaml").read_text())["input_window"])
    classes, thr = metrics["classes"], metrics["calibration"]["abstain_below"]
    out = {}
    for seg in ("startup", "steady_state"):
        X, labels, _ = r2._windows_paired(data, cells, metrics, window, seg)
        known = np.isin(labels, classes)
        y = np.array([classes.index(l) if l in classes else -1 for l in labels])
        for v in r2.VERSIONS:
            p = r2._predict(tag, X[v])[0]
            pred = p.argmax(1)
            out[f"{seg}_{v}"] = {"accuracy": float(np.mean(pred[known] == y[known])),
                                 "macro_f1": float(f1_score(y[known], pred[known], labels=list(range(len(classes))), average="macro", zero_division=0)),
                                 "abstention": float(np.mean(p.max(1) < thr)), "n": int(known.sum())}
    return out


def _cond_eval(run) -> dict:
    import report_classifier_evaluation as ce

    data = pd.read_parquet(REPO / "data" / "eval_cache_5class.parquet")
    res = ce.evaluate_run((REPO / run).resolve(), data)
    return {c: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n": m["n"]} for c, m in res["conditions"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r2-cache", type=Path, required=True)
    ap.add_argument("--candidates", nargs=3, required=True, help="PC=<run> RPi5=<run> ESP32=<run>")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "audit" / "plecs_om")
    args = ap.parse_args(argv)
    cand = dict(x.split("=", 1) for x in args.candidates)
    data = pd.read_parquet(args.r2_cache)
    cells = r2.sens_cells([100.0, 150.0, 200.0], [1.0, 0.5], 2)
    res = {"command": "python experiments/report_r4_retrained_rk45.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "host": platform.platform(), "python": platform.python_version(), "models": {}}
    for key in ("PC", "RPi5", "ESP32"):
        rec = {}
        for role, run in (("promoted_euler", PROMOTED[key]), ("candidate_rk45", cand[key])):
            m = json.loads((REPO / run / "metrics.json").read_text())
            meta = json.loads((REPO / run / "run_metadata.json").read_text())
            rec[role] = {"run": run, "dataset": meta["dataset_path"], "dataset_sha256": meta["dataset_sha256"],
                         "distilled_from": m.get("distilled_from"), "classes": m["classes"],
                         "recorded_test": {k: m.get(k) for k in ("test_accuracy", "test_macro_f1", "n_test")},
                         "calibration": {k: (m.get("calibration") or {}).get(k) for k in ("temperature", "abstain_below")},
                         "r2_simulations": _r2_eval(data, cells, key, run)}
            if key == "PC":
                rec[role]["evaluation_conditions_euler_cache"] = _cond_eval(run)
        res["models"][key] = rec
    (args.out / "r4_retrained_rk45.json").write_text(json.dumps(res, indent=2) + "\n")
    md = _md(res)
    (args.out / "r4_retrained_rk45.md").write_text(md)
    print(md)


def _md(res) -> str:
    L = ["# R4 — Classifiers retrained on the RK45 datasets (candidates)", "", f"Command: `{res['command']}`. Host: {res['host']}, Python {res['python']}.", "",
         "Decision: user, 2026-10-25 (after R2 → S-18 and R3). Same recipe as the promoted runs (`train_model.py --epochs 30 --batch-size 32 --seed 42 "
         "--keep-rpm-level --cap-per-class 3000`; RPi5 and ESP32 distilled, alpha 0.5, from the new PC teacher of the same dataset; RPi5 float16 and ESP32 int8 "
         "TFLite exported as before). **Candidates only: nothing promoted, `configs/registry.yaml` unchanged.**", "",
         "## 1. Recorded test split (each run's own grouped test split, from its own dataset)", "",
         "| Tier | Run | Dataset | Test accuracy | Test macro-F1 | n_test | Temperature / abstain below |", "|---|---|---|---|---|---|---|"]
    for key, rec in res["models"].items():
        for role in ("promoted_euler", "candidate_rk45"):
            r = rec[role]
            t, c = r["recorded_test"], r["calibration"]
            L.append(f"| {key} | `{r['run'].split('classifiers/')[1]}` ({'promoted, Euler' if role == 'promoted_euler' else 'candidate, RK45'}) | `{r['dataset'].split('/')[-1]}` | "
                     f"{t['test_accuracy']:.3f} | {t['test_macro_f1']:.3f} | {t['n_test']} | {c['temperature']:.3f} / {c['abstain_below']:.3f} |")
    L += ["", "## 2. R2 simulations (unseen seeds; NAT = Euler plant, REF = RK45 plant)", "",
          "| Tier | Model | Start-up acc. NAT / REF | Start-up macro-F1 NAT / REF | Steady acc. NAT / REF | Steady macro-F1 NAT / REF | Steady abstention NAT / REF |", "|---|---|---|---|---|---|---|"]
    for key, rec in res["models"].items():
        for role in ("promoted_euler", "candidate_rk45"):
            s = rec[role]["r2_simulations"]
            g = lambda seg, k: f"{s[f'{seg}_NAT'][k]:.3f} / {s[f'{seg}_REF'][k]:.3f}"
            L.append(f"| {key} | {'promoted (Euler-trained)' if role == 'promoted_euler' else 'candidate (RK45-trained)'} | {g('startup', 'accuracy')} | "
                     f"{g('startup', 'macro_f1')} | {g('steady_state', 'accuracy')} | {g('steady_state', 'macro_f1')} | {g('steady_state', 'abstention')} |")
    L += ["", "ESP32: accuracy and macro-F1 over its own 3 classes.", "", "## 3. PC models on the classifier-evaluation conditions (cached runs, Euler plant)", "",
          "| Condition | Promoted accuracy / macro-F1 | Candidate accuracy / macro-F1 |", "|---|---|---|"]
    a, b = res["models"]["PC"]["promoted_euler"]["evaluation_conditions_euler_cache"], res["models"]["PC"]["candidate_rk45"]["evaluation_conditions_euler_cache"]
    for cond in a:
        L.append(f"| {cond} | {a[cond]['accuracy']:.3f} / {a[cond]['macro_f1']:.3f} | {b[cond]['accuracy']:.3f} / {b[cond]['macro_f1']:.3f} |")
    L += ["", "**Not done:** promotion, registry or default changes (they need the user's decision).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
