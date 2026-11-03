"""AI-N27: does removing the S-21 converter artifact change the promoted dc_motor classifiers' outputs?

Pre-registered in docs/audit/assistant/ai_n27_preregistration.md §2 and amendment 1 (committed before
this ran). Runs only after the amended convergence criterion passed (s21_2_dcm_convergence_a1.json).

  set: the 124 grid acquisitions with the converter-branch signature (S21-1) + 30 unaffected controls
       (rng 2711), each rebuilt with experiments/generate_diagnosis_grid_dataset.build_jobs (3 seeds,
       2.5 s, --seed 7, the grid's faults) and the generator's Scenario;
  precondition: the rebuilt default plant equals the stored parquet exactly for 6 acquisitions;
  versions: GEM1000 (dc_converter_model "gem", 1000 substeps), DCM1000 ("dcm", 1000), STORED (the
       dataset); comparisons (a) GEM1000 vs DCM1000 -- decides -- and (b) STORED vs DCM1000;
  windows: each model's build_classification_windows, paired (same samples of the same run), by
       regime (start-up t < 0.5 s, steady state t >= 0.5 s) and all;
  rule: innocuous if agreement >= 0.98 and |delta accuracy| <= 1 point in every known class with
       affected windows, for every model (comparison (a), affected set, all windows).

    python experiments/report_ai_n27_impact.py [--workers 10] [--scratch DIR]
Writes docs/audit/assistant/ai_n27_impact.{md,json}. Regenerated runs go to --scratch (not committed).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "docs" / "audit" / "assistant" / "ai_n27_impact"
GRIDS = {"3class": ("data/diagnosis_grid_dataset.parquet", (None, "outer_race", "inner_race")),
         "5class": ("data/diagnosis_grid_dataset_5class.parquet", (None, "outer_race", "inner_race", "ball", "cage"))}
VERSIONS = {"GEM1000": {"dc_converter_model": "gem", "dc_plant_substeps": 1000}, "DCM1000": {"dc_converter_model": "dcm", "dc_plant_substeps": 1000}}
COMPARISONS = {"a_GEM1000_vs_DCM1000": ("GEM1000", "DCM1000"), "b_STORED_vs_DCM1000": ("STORED", "DCM1000")}
DECIDING = "a_GEM1000_vs_DCM1000"
CONTROL_PER_CLASS = {"3class": 5, "5class": 3}
CONTROL_SEED = 2711
SETTLE_S = 0.5
RULE = {"agreement_min": 0.98, "abs_delta_accuracy_max_points": 1.0}
CHECK_COLS = ("current_r", "rpm", "voltage_v", "torque_nm", "acc_x")
CLASSES5 = ("normal", "outer_race", "inner_race", "ball", "cage")


def sid_of(job: dict) -> str:
    return f"grid_{job['fault'] or 'normal'}_w{job['speed']:.0f}_s{job['severity_scale']:.2f}_seed{job['seed']}"


def simulate(args):
    """generate_diagnosis_grid_dataset._simulate, with the plant options of one version added."""
    import warnings

    warnings.simplefilter("ignore")
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY

    job, version, opts, out_path = args
    fault = job["fault"]
    kw = {}
    if fault is not None:
        kw = {"electrical_severity": 8.0 * job["severity_scale"], "mechanical_severity": CALIBRATED_MECHANICAL_SEVERITY[fault] * job["severity_scale"]}
    t0 = time.process_time()
    sc = Scenario(scenario_id=sid_of(job), fault_type=fault, seed=job["seed"], duration_s=job["duration_s"], omega_ref_rad_s=job["speed"],
                  electrical_noise_pct=job["electrical_noise_pct"], mechanical_noise_pct=job["mechanical_noise_pct"],
                  motor_parameter_overrides={"psi_e": job["psi_e"], "j_rotor": job["j_rotor"]}, **kw, **opts)
    df = records_to_dataframe(run_scenario(sc))
    df.to_parquet(out_path)
    return str(out_path), version, time.process_time() - t0


def select(scratch: Path) -> tuple[dict, list]:
    from generate_diagnosis_grid_dataset import build_jobs
    from report_s21_dataset_scan import load, scan_acquisitions

    sets, stored = {}, {}
    for g, (path, faults) in GRIDS.items():
        jobs = {sid_of(j): j for j in build_jobs(3, 2.5, 7, faults)}
        df = load(path)
        acq = scan_acquisitions(df)
        affected = list(acq.loc[acq["branch"], "source_file"])
        rng = np.random.default_rng(CONTROL_SEED)
        controls = []
        clean = acq[~acq["branch"]]
        for cls in sorted(clean["label"].unique()):
            pool = list(clean.loc[clean["label"] == cls, "source_file"])
            controls += list(rng.choice(pool, size=CONTROL_PER_CLASS[g], replace=False))
        missing = [s for s in affected + controls if s not in jobs]
        if missing:
            raise SystemExit(f"{g}: {len(missing)} acquisitions not in the generator's job list, e.g. {missing[:3]}")
        sets[g] = {"affected": affected, "control": controls, "jobs": jobs}
        full = pd.read_parquet(ROOT / path)
        stored[g] = {s: full[full["source_file"] == s].sort_values("timestamp_s").reset_index(drop=True) for s in affected + controls}
    return sets, stored


def reconstruction_check(sets, stored, scratch, workers) -> dict:
    tasks = []
    for g, s in sets.items():
        for sid in s["affected"][:3]:
            tasks.append((s["jobs"][sid], "DEFAULT", {}, scratch / f"check_{g}_{sid}.parquet"))
    with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
        done = pool.map(simulate, tasks)
    out = {}
    for (path, _v, _c), t in zip(done, tasks):
        g = path.split("check_")[1].split("_")[0]
        sid = t[0] and sid_of(t[0])
        a = pd.read_parquet(path).reset_index(drop=True)
        b = stored[g][sid]
        out[f"{g}/{sid}"] = all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) for c in CHECK_COLS)
    return out


def windows_for(df, metrics, window, regime):
    from driveflow.models.common import build_classification_windows

    d = df.sort_values("timestamp_s")
    if regime == "startup":
        d = d[d["timestamp_s"] < SETTLE_S]
    elif regime == "steady_state":
        d = d[d["timestamp_s"] >= SETTLE_S]
    X, y, _ = build_classification_windows(d, list(CLASSES5), metrics["channels"], window, cap_per_class=100_000, level_scaling=metrics.get("level_scaling"))
    return X, y


def evaluate(data: dict, sets: dict) -> dict:
    """data[(grid, sid, version)] -> DataFrame. Returns metrics per model / comparison / set / regime / class."""
    import yaml
    from sklearn.metrics import f1_score

    from report_s16_discretization_sensitivity import MODELS, _predict

    res = {}
    for key, (run, _kind) in MODELS.items():
        metrics = json.loads((ROOT / run / "metrics.json").read_text())
        window = int(yaml.safe_load((ROOT / run / "config.yaml").read_text())["input_window"])
        classes, thr = metrics["classes"], metrics["calibration"]["abstain_below"]
        res[key] = {"run": run, "classes": classes, "window": window, "abstain_below": thr}
        for set_name in ("affected", "control"):
            for regime in ("startup", "steady_state", "all"):
                X = {v: [] for v in ("STORED", "GEM1000", "DCM1000")}
                labels = []
                for g, s in sets.items():
                    for sid in s[set_name]:
                        per = {v: windows_for(data[(g, sid, v)], metrics, window, regime) for v in X}
                        n = {v: (0 if per[v][0] is None else len(per[v][0])) for v in X}
                        assert len(set(n.values())) == 1, f"windows not paired for {g}/{sid}: {n}"
                        if n["DCM1000"] == 0:
                            continue
                        for v in X:
                            X[v].append(per[v][0])
                        labels += list(per["DCM1000"][1])
                if not labels:
                    continue
                labels = np.array(labels)
                P = {v: _predict(key, np.concatenate(X[v]))[0] for v in X}
                pred = {v: P[v].argmax(1) for v in X}
                y = np.array([classes.index(l) if l in classes else -1 for l in labels])
                for comp, (va, vb) in COMPARISONS.items():
                    block = {}
                    for cls in ["ALL"] + sorted(set(labels)):
                        m = np.ones(len(labels), bool) if cls == "ALL" else labels == cls
                        known = m & (y >= 0)
                        r = {"n_windows": int(m.sum()), "n_known": int(known.sum()), "agreement": float(np.mean(pred[va][m] == pred[vb][m])),
                             "mean_abs_delta_confidence": float(np.mean(np.abs(P[va][m].max(1) - P[vb][m].max(1)))),
                             f"abstention_{va}": float(np.mean(P[va][m].max(1) < thr)), f"abstention_{vb}": float(np.mean(P[vb][m].max(1) < thr))}
                        if known.any():
                            for v in (va, vb):
                                r[f"accuracy_{v}"] = float(np.mean(pred[v][known] == y[known]))
                                r[f"macro_f1_{v}"] = float(f1_score(y[known], pred[v][known], labels=list(range(len(classes))), average="macro", zero_division=0))
                            r["delta_accuracy_points"] = 100.0 * (r[f"accuracy_{vb}"] - r[f"accuracy_{va}"])
                        block[cls] = r
                    res[key].setdefault(comp, {}).setdefault(set_name, {})[regime] = block
    return res


def verdict(res: dict, sets: dict) -> dict:
    out = {}
    for key, r in res.items():
        block = r[DECIDING]["affected"]["all"]
        checks = {}
        for cls, m in block.items():
            if cls == "ALL" or cls not in r["classes"]:
                continue
            ok = m["agreement"] >= RULE["agreement_min"] and abs(m.get("delta_accuracy_points", 0.0)) <= RULE["abs_delta_accuracy_max_points"]
            checks[cls] = {"agreement": m["agreement"], "delta_accuracy_points": m.get("delta_accuracy_points"), "pass": bool(ok)}
        out[key] = {"classes": checks, "innocuous": all(c["pass"] for c in checks.values())}
    out["overall"] = "innocuous" if all(v["innocuous"] for k, v in out.items() if k != "overall") else "affects"
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--scratch", type=Path, required=True)
    args = ap.parse_args(argv)
    conv = json.loads((OUT.parent / "s21_2_dcm_convergence_a1.json").read_text())
    if not conv["converged"]:
        raise SystemExit("the amended convergence criterion was not met: the impact study is not run (amendment 1, §2.2)")
    args.scratch.mkdir(parents=True, exist_ok=True)
    sets, stored = select(args.scratch)
    check = reconstruction_check(sets, stored, args.scratch, args.workers)
    print("reconstruction check:", check, flush=True)
    if not all(check.values()):
        OUT.with_suffix(".json").write_text(json.dumps({"stopped": "reconstruction check failed", "check": check}, indent=2) + "\n")
        raise SystemExit("reconstruction check failed: stopping (preregistration §2.2)")
    tasks = [(s["jobs"][sid], v, opts, args.scratch / f"{g}_{v}_{sid}.parquet") for g, s in sets.items() for sid in s["affected"] + s["control"] for v, opts in VERSIONS.items()]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        done = pool.map(simulate, tasks, chunksize=1)
    wall = time.time() - t0
    data = {}
    for g, s in sets.items():
        for sid in s["affected"] + s["control"]:
            data[(g, sid, "STORED")] = stored[g][sid]
            for v in VERSIONS:
                data[(g, sid, v)] = pd.read_parquet(args.scratch / f"{g}_{v}_{sid}.parquet")
    res = evaluate(data, sets)
    ver = verdict(res, sets)
    speed = {}
    for g, s in sets.items():
        for set_name in ("affected", "control"):
            for v in ("STORED", "GEM1000", "DCM1000"):
                errs = [float(((data[(g, sid, v)]["rpm"].to_numpy() * 2 * np.pi / 60)[len(data[(g, sid, v)]) // 2:].mean() - s["jobs"][sid]["speed"]) / s["jobs"][sid]["speed"])
                        for sid in s[set_name]]
                speed.setdefault(set_name, {}).setdefault(v, []).extend(errs)
    speed_summary = {k: {v: {"median_pct": 100 * float(np.median(e)), "max_pct": 100 * float(np.max(e))} for v, e in d.items()} for k, d in speed.items()}
    result = {"command": "python experiments/report_ai_n27_impact.py --workers 10 --scratch <scratch>", "simulated": True,
              "preregistration": ["docs/audit/assistant/ai_n27_preregistration.md", "docs/audit/assistant/ai_n27_preregistration_amendment1.md"],
              "n_runs": {g: {"affected": len(s["affected"]), "control": len(s["control"])} for g, s in sets.items()}, "reconstruction_check": check,
              "simulation_wall_s": wall, "cpu_s_total": float(sum(c for _p, _v, c in done)), "speed_error_second_half": speed_summary,
              "results": res, "verdict": ver, "rule": RULE}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    write_md(result)


def _fmt(m, va, vb):
    acc = f"{100 * m[f'accuracy_{va}']:.1f} → {100 * m[f'accuracy_{vb}']:.1f} ({m['delta_accuracy_points']:+.1f})" if f"accuracy_{va}" in m else "—"
    f1 = f"{m[f'macro_f1_{va}']:.3f} → {m[f'macro_f1_{vb}']:.3f}" if f"macro_f1_{va}" in m else "—"
    return (f"{m['n_windows']} | {m['agreement']:.3f} | {acc} | {f1} | {m['mean_abs_delta_confidence']:.3f} | "
            f"{100 * m[f'abstention_{va}']:.1f} → {100 * m[f'abstention_{vb}']:.1f}")


def write_md(r: dict) -> None:
    ver = r["verdict"]
    L = ["# AI-N27 — impact of the S-21 converter artifact on the promoted dc_motor classifiers", "",
         "**Simulated. Pre-registered** ([`ai_n27_preregistration.md`](ai_n27_preregistration.md) §2 and [`amendment 1`](ai_n27_preregistration_amendment1.md)), committed before this ran.", "",
         f"- **Command:** `{r['command']}`. Raw values: [`ai_n27_impact.json`](ai_n27_impact.json).",
         f"- **Runs:** {r['n_runs']}, each in GEM1000 and DCM1000 ({r['simulation_wall_s'] / 3600:.1f} h wall clock, {r['cpu_s_total'] / 3600:.1f} h CPU). Reconstruction check (default plant = stored data, exactly): "
         + ("**pass**" if all(r["reconstruction_check"].values()) else "**fail**") + f" ({len(r['reconstruction_check'])} acquisitions).",
         f"- **Verdict (comparison a, affected set, all windows; agreement ≥ {RULE['agreement_min']} and |Δ accuracy| ≤ {RULE['abs_delta_accuracy_max_points']:g} point in every known class): "
         f"{ver['overall'].upper()}**.", ""]
    L += ["| Model | Verdict | Per class (agreement; Δ accuracy points) |", "|---|---|---|"]
    for key in ("PC", "RPi5", "ESP32"):
        v = ver[key]
        L.append(f"| {key} | **{'innocuous' if v['innocuous'] else 'affects'}** | " + "; ".join(f"{c} {x['agreement']:.3f}, {x['delta_accuracy_points']:+.1f}" + ("" if x["pass"] else " ✗")
                                                                                          for c, x in v["classes"].items()) + " |")
    L += ["", "Mean speed over the second half vs the reference (affected set): " + ", ".join(
        f"{v} median {x['median_pct']:+.1f} %, max {x['max_pct']:+.1f} %" for v, x in r["speed_error_second_half"]["affected"].items()) + ".", ""]
    titles = {"a_GEM1000_vs_DCM1000": ("(a) GEM1000 vs DCM1000 — pre-registered, decides", "GEM1000", "DCM1000"),
              "b_STORED_vs_DCM1000": ("(b) STORED (training data) vs DCM1000 — practical question, same weight, does not decide", "STORED", "DCM1000")}
    for key in ("PC", "RPi5", "ESP32"):
        res = r["results"][key]
        L += [f"## {key} (`{res['run']}`)", ""]
        for comp, (title, va, vb) in titles.items():
            L += [f"### {title}", ""]
            for set_name in ("affected", "control"):
                L += [f"**{set_name.capitalize()} set**", "", f"| Regime | Class | Windows | Agreement | Accuracy % {va} → {vb} (Δ) | Macro-F1 | Mean |Δ conf.| | Abstention % |",
                      "|---|---|---|---|---|---|---|---|"]
                for regime in ("startup", "steady_state", "all"):
                    block = res.get(comp, {}).get(set_name, {}).get(regime, {})
                    for cls, m in block.items():
                        L.append(f"| {regime} | {cls} | {_fmt(m, va, vb)} |")
                L.append("")
    L += ["ESP32: accuracy and macro-F1 over its 3 classes; ball and cage windows enter only the agreement (AI-N13).", "",
          "## Not executed", "", "- Full grid regeneration, retraining, and any default change (S21-4)."]
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
