"""B4c confirmatory evaluation (docs/audit/b4/b4c_confirmatory_preregistration.md, approved 2026-11-12 before any E4c data).

    python experiments/report_b4c.py [--e4c data/b4c_e4c_5class.parquet]

Frozen B4 candidates (PC Keras, RPi5 float16) against the promoted run of each tier, on the same E4c windows
(steady state t >= 0.5 s, built as B4's E4). Criteria per tier, all must hold:
  1. macro-F1 >= 0.80 and recall >= 0.50 per class (argmax);
  2. no class's recall drops more than 5 pt against the promoted run;
  3. abstention one-sided: a rise is at most +5 pt; a fall requires selective risk (error among the answered windows)
     candidate <= promoted + 1 pt.
Descriptive: bootstrap 95 % CI (1000 resamples of acquisitions, seed 0) of delta macro-F1 and delta selective risk,
per-speed results, confusion matrices. Nothing is promoted (M-D5). Writes docs/audit/b4/b4c_results.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
import report_ai_n27_regen as RG  # noqa: E402

PLAN = yaml.safe_load((REPO / "configs" / "b4c_plan.yaml").read_text())
CRIT = PLAN["criteria"]
OUT = REPO / "docs" / "audit" / "b4" / "b4c_results"


def _selective_risk(P, y, thr):
    ans = P.max(1) >= thr
    return float(np.mean(P[ans].argmax(1) != y[ans])) if ans.any() else None


def _boot(Pa, Pb, y, groups, k, ta, tb):
    rng = np.random.default_rng(RG.BOOT_SEED)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    df1, dsr = [], []
    for _ in range(RG.N_BOOT):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        df1.append(100.0 * (RG._macro_f1(y[sel], Pb[sel].argmax(1), k) - RG._macro_f1(y[sel], Pa[sel].argmax(1), k)))
        a, b = _selective_risk(Pa[sel], y[sel], ta), _selective_risk(Pb[sel], y[sel], tb)
        if a is not None and b is not None:
            dsr.append(100.0 * (b - a))
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None  # noqa: E731
    return pc(df1), pc(dsr)


def _tier(tier: str, spec: dict, df: pd.DataFrame) -> dict:
    prom, cand = RG._model_info(spec["promoted"]), RG._model_info(spec["candidate"])
    classes = prom["metrics"]["classes"]
    assert classes == cand["metrics"]["classes"] and prom["window"] == cand["window"]
    assert prom["metrics"].get("level_scaling") == cand["metrics"].get("level_scaling")
    X, ys, g = RG._windows(df[df["timestamp_s"] >= RG.SETTLE_S], classes, prom, 100_000)
    keep = np.isin(ys, classes)
    X, ys, g = X[keep], ys[keep], g[keep]
    y = np.array([classes.index(c) for c in ys])
    speed = df.groupby("source_file")["grid_speed"].first() if "grid_speed" in df.columns else None
    res = {"promoted": spec["promoted"], "candidate": spec["candidate"], "artifact": spec["artifact"], "classes": classes,
           "n_windows": int(len(y)), "n_acquisitions": int(len(np.unique(g)))}
    P, S = {}, {}
    for role, info in (("promoted", prom), ("candidate", cand)):
        thr = info["metrics"]["calibration"]["abstain_below"]
        P[role] = RG._predict(info["run"], spec["artifact"], X)
        S[role] = RG._score(P[role], y, classes, thr)
        S[role]["abstain_below"] = thr
        S[role]["selective_risk"] = _selective_risk(P[role], y, thr)
        if speed is not None:
            sp = speed.reindex(g).to_numpy()
            S[role]["per_speed"] = {f"{s:g}": RG._score(P[role][sp == s], y[sp == s], classes, thr) for s in sorted(set(sp))}
    res["promoted_scores"], res["candidate_scores"] = S["promoted"], S["candidate"]
    a, b = S["promoted"], S["candidate"]
    d_abs = 100.0 * (b["abstention"] - a["abstention"])
    d_rec = {c: 100.0 * (b["recall"][c] - a["recall"][c]) for c in classes if a["recall"][c] is not None}
    worst = min(d_rec, key=d_rec.get)
    checks = {
        "1_low_speed_macro_f1": {"value": b["macro_f1"], "threshold": CRIT["low_speed"]["macro_f1_min"], "pass": b["macro_f1"] >= CRIT["low_speed"]["macro_f1_min"]},
        "1_low_speed_min_recall": {"value": min(v for v in b["recall"].values() if v is not None), "threshold": CRIT["low_speed"]["recall_min_per_class"]},
        "2_worst_delta_recall": {"class": worst, "value": d_rec[worst], "threshold": -CRIT["max_recall_drop_pts_per_class"], "pass": d_rec[worst] >= -CRIT["max_recall_drop_pts_per_class"]},
    }
    checks["1_low_speed_min_recall"]["pass"] = checks["1_low_speed_min_recall"]["value"] >= CRIT["low_speed"]["recall_min_per_class"]
    ab = CRIT["abstention_one_sided"]
    if d_abs > 0:
        checks["3_abstention"] = {"branch": "rise", "delta_abstention_points": d_abs, "threshold": ab["max_rise_pts"], "pass": d_abs <= ab["max_rise_pts"]}
    else:
        ok = a["selective_risk"] is not None and b["selective_risk"] is not None
        excess = 100.0 * (b["selective_risk"] - a["selective_risk"]) if ok else None
        checks["3_abstention"] = {"branch": "fall_or_equal", "delta_abstention_points": d_abs, "selective_risk_candidate": b["selective_risk"],
                                  "selective_risk_promoted": a["selective_risk"], "excess_points": excess, "threshold": ab["max_selective_risk_excess_pts_if_lower"],
                                  "pass": ok and excess <= ab["max_selective_risk_excess_pts_if_lower"]}
    res["delta_points"] = {"macro_f1": 100.0 * (b["macro_f1"] - a["macro_f1"]), "abstention": d_abs, "recall": d_rec,
                           "selective_risk": (100.0 * (b["selective_risk"] - a["selective_risk"])) if a["selective_risk"] is not None and b["selective_risk"] is not None else None}
    ci_f1, ci_sr = _boot(P["promoted"], P["candidate"], y, g, len(classes), a["abstain_below"], b["abstain_below"])
    res["ci95_points"] = {"delta_macro_f1": ci_f1, "delta_selective_risk": ci_sr}
    res["checks"] = checks
    res["verdict"] = "PASSES" if all(c["pass"] for c in checks.values()) else "DOES NOT PASS"
    return res


def _md(r: dict) -> str:
    f = lambda x, p=3: "—" if x is None else f"{x:.{p}f}"  # noqa: E731
    L = ["# B4c results: confirmatory evaluation of the B4 PC and RPi5 candidates", "",
         f"**Simulated** (`dc_perm_ex_v2`, `dcm_exact`, 10 kHz). Pre-registration: [`b4c_confirmatory_preregistration.md`](b4c_confirmatory_preregistration.md) "
         "(approved 2026-11-12, committed before any E4c data). Command: `" + r["command"] + "`. Raw: [`b4c_results.json`](b4c_results.json).", "",
         "**Nothing is promoted (M-D5).** A pass is a recommendation to Luciano.", "",
         "| Tier | Verdict | Macro-F1 cand / prom | Min recall cand | Worst Δ recall (pt) | Δ abstention (pt) | Selective risk cand / prom |", "|---|---|---|---|---|---|---|"]
    for t, x in r["tiers"].items():
        c, a, b = x["checks"], x["promoted_scores"], x["candidate_scores"]
        L.append(f"| {t} | **{x['verdict']}** | {f(b['macro_f1'])} / {f(a['macro_f1'])} | {f(c['1_low_speed_min_recall']['value'])} | "
                 f"{c['2_worst_delta_recall']['class']} {c['2_worst_delta_recall']['value']:+.1f} | {x['delta_points']['abstention']:+.1f} | "
                 f"{f(b['selective_risk'])} / {f(a['selective_risk'])} |")
    for t, x in r["tiers"].items():
        c = x["checks"]
        L += ["", f"## {t}: `{x['candidate']}` vs `{x['promoted']}` ({x['n_windows']} windows, {x['n_acquisitions']} acquisitions)", "",
              "| Criterion | Value | Threshold | Pass |", "|---|---|---|---|",
              f"| 1. macro-F1 ≥ 0.80 | {f(c['1_low_speed_macro_f1']['value'])} | 0.80 | {c['1_low_speed_macro_f1']['pass']} |",
              f"| 1. recall ≥ 0.50 in every class (min) | {f(c['1_low_speed_min_recall']['value'])} | 0.50 | {c['1_low_speed_min_recall']['pass']} |",
              f"| 2. worst Δ recall ≥ −5 pt | {c['2_worst_delta_recall']['class']} {c['2_worst_delta_recall']['value']:+.1f} | −5.0 | {c['2_worst_delta_recall']['pass']} |"]
        a3 = c["3_abstention"]
        if a3["branch"] == "rise":
            L.append(f"| 3. abstention rose: Δ ≤ +5 pt | {a3['delta_abstention_points']:+.1f} | +5.0 | {a3['pass']} |")
        else:
            L.append(f"| 3. abstention fell ({a3['delta_abstention_points']:+.1f} pt): selective risk excess ≤ 1 pt | "
                     f"{f(a3['selective_risk_candidate'])} vs {f(a3['selective_risk_promoted'])} → {('—' if a3['excess_points'] is None else format(a3['excess_points'], '+.1f'))} pt | +1.0 | {a3['pass']} |")
        ci = x["ci95_points"]
        L += ["", f"Descriptive: Δ macro-F1 {x['delta_points']['macro_f1']:+.1f} pt, 95 % CI {ci['delta_macro_f1'] and '[%+.1f, %+.1f]' % tuple(ci['delta_macro_f1'])}; "
              f"Δ selective risk 95 % CI {ci['delta_selective_risk'] and '[%+.1f, %+.1f]' % tuple(ci['delta_selective_risk'])} pt.", "",
              "| Speed (rad/s) | Macro-F1 cand / prom | Abstention cand / prom | Recall cand (per class) |", "|---|---|---|---|"]
        for s, sc in x["candidate_scores"].get("per_speed", {}).items():
            sp = x["promoted_scores"]["per_speed"][s]
            L.append(f"| {s} | {f(sc['macro_f1'])} / {f(sp['macro_f1'])} | {f(sc['abstention'])} / {f(sp['abstention'])} | "
                     + ", ".join(f"{k} {f(v, 2)}" for k, v in sc["recall"].items()) + " |")
        cls = x["classes"]
        L += ["", "Confusion (rows true, columns predicted; candidate (promoted)):", "", "| true \\ pred | " + " | ".join(cls) + " |", "|---|" + "---|" * len(cls)]
        for i, cc in enumerate(cls):
            L.append(f"| {cc} | " + " | ".join(f"{x['candidate_scores']['confusion'][i][j]} ({x['promoted_scores']['confusion'][i][j]})" for j in range(len(cls))) + " |")
    L += ["", f"CPU: {r['cpu']}", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e4c", default="data/b4c_e4c_5class.parquet")
    ap.add_argument("--cpu", default="{}", help="JSON with the measured CPU time of the generation (added to the report)")
    args = ap.parse_args(argv)
    t0, c0 = time.time(), time.process_time()
    df = pd.read_parquet(REPO / args.e4c)
    r = {"command": "python experiments/report_b4c.py" + (f" --e4c {args.e4c}" if args.e4c != "data/b4c_e4c_5class.parquet" else ""),
         "simulated": True, "preregistration": "docs/audit/b4/b4c_confirmatory_preregistration.md", "plan": PLAN, "host": platform.platform(),
         "e4c": {"file": args.e4c, "n_runs": int(df["source_file"].nunique()), "seeds": [int(df["seed"].min()), int(df["seed"].max())] if "seed" in df.columns else None},
         "tiers": {t: _tier(t, spec, df) for t, spec in PLAN["models"].items()}, "nothing_promoted": True}
    r["cpu"] = {**json.loads(args.cpu), "evaluation_wall_s": round(time.time() - t0, 1), "evaluation_process_cpu_s": round(time.process_time() - c0, 1)}
    OUT.with_suffix(".json").write_text(json.dumps(r, indent=2, default=float) + "\n")
    OUT.with_suffix(".md").write_text(_md(r))
    print(json.dumps({t: x["verdict"] for t, x in r["tiers"].items()}))


if __name__ == "__main__":
    main()
