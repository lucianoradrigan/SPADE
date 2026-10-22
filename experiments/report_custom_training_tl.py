"""Evidence for AI Diagnostics → Train a model and the Transfer Learning modes (2026-10-21).

Three simulated experiments, all through the same library calls the GUI uses:

A. Custom training (driveflow.ai.custom_training): a 4-class dc_motor classifier (normal, outer
   race, inner race and a user-defined "unbalance" fault at order 1.0) over PI and MPC, three
   speeds and three severities. Reports the grouped-by-run test metrics and the measured
   simulation time per simulated second (the GUI's time estimate).
B. Full vs. feature transfer on the base vocabulary: the promoted 5-class model, adapted to a
   shifted condition (MPC controller, 0-60 % sensor noise, 120-220 rad/s) with
   FULL_FINETUNE, FEATURE_EXTRACTOR and NEW_HEAD; base and candidates on the same test windows.
C. A new class: NEW_HEAD from the 3-class model pc_server/2026-10-18_run01 to
   {normal, outer_race, inner_race, unbalance}; the base cannot be compared, so the absolute floor
   is what the gate checks.

Usage:
    python experiments/report_custom_training_tl.py            # ~15 min on a laptop CPU
Writes docs/audit/ai/custom_training_tl.{md,json}. Models go to a temporary directory (not configs/).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from driveflow.ai import custom_training as ct
from driveflow.ai.registry import DEFAULT_REGISTRY_PATH, REPO_ROOT
from driveflow.ai.transfer import workflow as wf
from driveflow.ai.transfer.loader import ModelLoader
from driveflow.ai.transfer.pipeline import FineTuneStrategy as S

OUT = REPO_ROOT / "docs" / "audit" / "ai"
CMD = "python experiments/report_custom_training_tl.py"
UNBALANCE = ct.dc_class("unbalance", "unbalance", 1.0, [], 0.03)


def _tl_data(classes: list, n_per_class: int, seed: int, controller: str, speeds: tuple, noise: float, duration: float, workers: int) -> pd.DataFrame:
    """A small grid through custom_training (same simulation path), labelled for TL."""
    req = ct.default_request("dc_motor")
    req["classes"] = classes
    req["operating"].update(controllers=[controller], setpoints=list(speeds), severity_scales=[1.0], noise_max_pct=noise, noise_free_share=0.0,
                            drift={}, duration_s=duration, skip_start_s=0.5, runs_per_cell=max(1, n_per_class // len(speeds)), seed=seed)
    req["model"].update(input_window=2048)
    df, outcomes = ct.simulate(ct.build_conditions(req), workers=workers)
    assert all(o["status"] == "ok" for o in outcomes), outcomes
    df = df[df["timestamp_s"] >= 0.5].reset_index(drop=True)
    return df


def _ckeys(report: dict) -> dict:
    return {c["name"]: c["status"] for c in report["checks"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="spade_ct_"))
    result = {"command": CMD, "workers": args.workers, "origin": "SIMULATED"}

    # ---------------------------------------------------------------- A. custom training
    req = ct.default_request("dc_motor")
    req["name"] = "evidence_4class"
    req["classes"].append(UNBALANCE)
    req["operating"].update(controllers=["PI", "MPC"], setpoints=[100.0, 150.0, 200.0], severity_scales=[0.5, 1.0, 2.0], runs_per_cell=1,
                            duration_s=1.5, skip_start_s=0.5, noise_max_pct=50.0, seed=3)
    req["model"].update(preset="pc_server", input_window=1024, epochs=15, cap_per_class=2000, seed=42)
    s = ct.summary(req)
    t0 = time.perf_counter()
    df, outcomes = ct.simulate(ct.build_conditions(req), workers=args.workers)
    t_sim = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_dir = ct.train(req, df, root=tmp / "models", dataset_root=tmp / "data")
    t_train = time.perf_counter() - t0
    m = json.loads((run_dir / "metrics.json").read_text())
    result["A_custom_training"] = {
        "request": {"classes": [c["label"] for c in req["classes"]], "controllers": ["PI", "MPC"], "setpoints_rad_s": [100, 150, 200], "severity_scales": [0.5, 1.0, 2.0],
                    "noise_max_pct": 50, "duration_s": 1.5, "skip_start_s": 0.5, "window": 1024, "preset": "pc_server", "epochs": 15},
        "n_runs": s["n_runs"], "runs_per_class": s["runs_per_class"], "n_failed_runs": sum(o["status"] != "ok" for o in outcomes),
        "simulation_wall_s": round(t_sim, 1), "simulated_s": s["simulated_s"], "wall_s_per_simulated_s_with_workers": round(t_sim / s["simulated_s"], 3),
        "estimate_single_process_s": s["estimated_single_process_s"], "training_wall_s": round(t_train, 1),
        "classes": m["classes"], "n_train": m["n_train"], "n_val": m["n_val"], "n_test": m["n_test"],
        "test_macro_f1": m["test_macro_f1"], "test_accuracy": m["test_accuracy"], "test_per_class_recall": m["test_per_class_recall"],
        "test_confusion_matrix": m["test_confusion_matrix"], "ood_fitted": "ood" in m,
    }
    print("A", json.dumps({k: result["A_custom_training"][k] for k in ("n_runs", "test_macro_f1", "simulation_wall_s", "training_wall_s")}))

    # ---------------------------------------------------------------- B. full vs feature transfer
    loader = ModelLoader(DEFAULT_REGISTRY_PATH, "dc_motor", "classifier")
    base, meta = loader.load_latest(), loader.get_metadata()
    classes3 = [ct.dc_class(ct.HEALTHY), ct.dc_class("outer_race", "outer_race"), ct.dc_class("inner_race", "inner_race")]
    df_b = _tl_data(classes3, 12, seed=11, controller="MPC", speeds=(120.0, 170.0, 220.0), noise=60.0, duration=1.5, workers=args.workers)
    ds = wf.prepare_dataset("dc_motor", "classifier", meta["metrics"], meta["config"]["input_window"], None, df_b, seed=0, val_frac=0.25, test_frac=0.25,
                            target_classes=["normal", "outer_race", "inner_race"])
    rows = []
    for strat, cfg in ((S.FULL_FINETUNE, {"learning_rate": 1e-4}), (S.FEATURE_EXTRACTOR, {"frozen_layers": 3, "learning_rate": 1e-4}), (S.NEW_HEAD, {"learning_rate": 1e-3})):
        t0 = time.perf_counter()
        res = wf.finetune(base, ds, {"strategy": strat, "epochs": 10, "batch_size": 16, **cfg}, seed=0)
        rep = res.report
        rows.append({"strategy": strat.value, "mode": res.lineage["transfer_mode"], "candidate_macro_f1": rep["metrics"]["macro_f1"],
                     "base_macro_f1_same_test": rep["baseline_metrics"].get("macro_f1"), "candidate_accuracy": rep["metrics"]["accuracy"],
                     "checks": _ckeys(rep), "base_unchanged": res.base_unchanged, "wall_s": round(time.perf_counter() - t0, 1)})
        print("B", rows[-1]["strategy"], rows[-1]["candidate_macro_f1"], rows[-1]["base_macro_f1_same_test"])
    result["B_full_vs_feature"] = {"base": str(Path(meta["run_dir"]).relative_to(REPO_ROOT)), "condition": "MPC, 120-220 rad/s, 0-60 % noise (all runs noisy), 1.5 s runs, first 0.5 s dropped",
                                   "n_acquisitions": ds.summary["n_acquisitions"], "n_windows": {k: len(v["X"]) for k, v in ds.windows.items()},
                                   "classes_present": ds.summary["classes_present"], "epochs": 10, "results": rows}

    # ---------------------------------------------------------------- C. new class with a new head
    base3_dir = REPO_ROOT / "configs" / "classifiers" / "pc_server" / "2026-10-18_run01"
    base3 = loader.load_by_generation(base3_dir)
    df_c = _tl_data(classes3 + [UNBALANCE], 12, seed=21, controller="PI", speeds=(100.0, 150.0, 200.0), noise=30.0, duration=1.5, workers=args.workers)
    target = ["normal", "outer_race", "inner_race", "unbalance"]
    ds_c = wf.prepare_dataset("dc_motor", "classifier", base3["metrics"], base3["config"].input_window, None, df_c, seed=0, val_frac=0.25, test_frac=0.25, target_classes=target)
    refused = None
    try:
        wf.finetune(base3, ds_c, {"strategy": S.FULL_FINETUNE, "epochs": 1})
    except ValueError as exc:
        refused = str(exc)
    res = wf.finetune(base3, ds_c, {"strategy": S.NEW_HEAD, "epochs": 15, "batch_size": 16, "unfreeze_last": 1, "learning_rate": 1e-3}, seed=0,
                      thresholds={"max_macro_f1_drop": 0.02})
    dec = wf.gate_decision(ds_c, res, None)
    result["C_new_class"] = {"base": str(base3_dir.relative_to(REPO_ROOT)), "base_classes": base3["metrics"]["classes"], "target_classes": ds_c.classes,
                             "new_vocabulary": ds_c.new_vocabulary, "full_finetune_refused_with": refused,
                             "n_windows": {k: len(v["X"]) for k, v in ds_c.windows.items()}, "candidate_macro_f1": res.report["metrics"]["macro_f1"],
                             "candidate_recall_per_class": dict(zip(ds_c.classes, res.report["metrics"]["recall_per_class"])),
                             "floors": res.report.get("floors"), "checks": _ckeys(res.report),
                             "gate_without_reviewer": {"status": dec.status, "blocking": [i.name for i in dec.blocking]}}
    print("C", result["C_new_class"]["candidate_macro_f1"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "custom_training_tl.json").write_text(json.dumps(result, indent=2, default=float))
    (OUT / "custom_training_tl.md").write_text(_markdown(result))
    print("wrote", OUT / "custom_training_tl.md")


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _markdown(r: dict) -> str:
    a, b, c = r["A_custom_training"], r["B_full_vs_feature"], r["C_new_class"]
    cm_rows = ["| true \\ predicted | " + " | ".join(a["classes"]) + " |", "|---" * (len(a["classes"]) + 1) + "|"]
    cm_rows += [f"| {k} | " + " | ".join(str(v) for v in row) + " |" for k, row in zip(a["classes"], a["test_confusion_matrix"])]
    lines = [
        "# Custom classifier training and transfer-learning modes: evidence (2026-10-21)",
        "",
        f"Every number below comes from one run of `{r['command']}` ({r['workers']} simulation processes), which also writes `custom_training_tl.json`. "
        "All data is SIMULATED. No model was promoted; the models were written to a temporary directory.",
        "",
        "## A. AI Diagnostics → Train a model",
        "",
        f"- **Request:** classes {', '.join(a['request']['classes'])} (unbalance = user-defined fault, order 1.0, mechanical severity 0.03). "
        f"Controllers PI and MPC; ω_ref ∈ {{100, 150, 200}} rad/s; severity × {{0.5, 1, 2}}; noise up to 50 % in two thirds of the runs; ψ_e and J drift. "
        f"Runs of 1.5 s with the first 0.5 s dropped; window 1024 samples; `pc_server` architecture; 15 epochs.",
        f"- **Data:** {a['n_runs']} runs ({', '.join(f'{k}: {v}' for k, v in a['runs_per_class'].items())}), {a['n_failed_runs']} failed.",
        f"- **Windows:** train {a['n_train']} / validation {a['n_val']} / test {a['n_test']}, split by run.",
        f"- **Test macro-F1 {_f(a['test_macro_f1'])}**, accuracy {_f(a['test_accuracy'])}. OOD detector fitted: {a['ood_fitted']}.",
        "- **Recall per class:** " + ", ".join(f"{k} {_f(v)}" for k, v in a["test_per_class_recall"].items()) + ".",
        f"- **Time:** simulation {a['simulation_wall_s']} s for {a['simulated_s']:.0f} simulated seconds with {r['workers']} processes "
        f"({a['wall_s_per_simulated_s_with_workers']} s per simulated second). The GUI's single-process estimate was {a['estimate_single_process_s']:.0f} s. "
        f"Training took {a['training_wall_s']} s.",
        "",
        "Test confusion matrix (windows):",
        "",
        "\n".join(cm_rows),
        "",
        "## B. Full vs. feature transfer on a shifted condition (same classes)",
        "",
        f"- **Base:** `{b['base']}`, the promoted 5-class model.",
        f"- **Condition:** {b['condition']}.",
        f"- **Data:** {b['n_acquisitions']} acquisitions. Windows: train {b['n_windows']['train']} / validation {b['n_windows']['val']} / test {b['n_windows']['test']}. "
        f"Classes present: {', '.join(b['classes_present'])}. {b['epochs']} epochs each.",
        "",
        "| Strategy | Mode | Base macro-F1 (same test) | Candidate macro-F1 | Candidate accuracy | Failing checks | Base unchanged |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in b["results"]:
        failing = [k for k, v in row["checks"].items() if v not in ("PASS", "NOT_APPLICABLE")]
        lines.append(f"| {row['strategy']} | {row['mode']} | {_f(row['base_macro_f1_same_test'])} | {_f(row['candidate_macro_f1'])} | {_f(row['candidate_accuracy'])} | "
                     f"{', '.join(failing) or 'none'} | {row['base_unchanged']} |")
    lines += [
        "",
        "The base model keeps all 5 classes. The test set holds only the 3 classes listed, so its macro-F1 is computed over the classes present.",
        "",
        "## C. A class the base model does not have (feature transfer, new output layer)",
        "",
        f"- **Base:** `{c['base']}`, classes {', '.join(c['base_classes'])}.",
        f"- **Target classes:** {', '.join(c['target_classes'])} (new vocabulary: {c['new_vocabulary']}).",
        f"- **Full fine-tuning was refused:** \"{c['full_finetune_refused_with']}\".",
        f"- **Windows:** train {c['n_windows']['train']} / validation {c['n_windows']['val']} / test {c['n_windows']['test']}.",
        f"- **NEW_HEAD candidate:** macro-F1 {_f(c['candidate_macro_f1'])}; recall " + ", ".join(f"{k} {_f(v)}" for k, v in c["candidate_recall_per_class"].items()) + ".",
        "- **Checks:** " + ", ".join(f"{k} {v}" for k, v in c["checks"].items()) + ".",
        f"- **Floors (D1, 2026-10-22):** macro-F1 ≥ {c['floors']['min_macro_f1']['value']:g}, recall per class ≥ {c['floors']['min_per_class_recall']['value']:g}.",
        f"- **Gate without a reviewer:** {c['gate_without_reviewer']['status']}, blocking {', '.join(c['gate_without_reviewer']['blocking'])}.",
        "",
        "## Limits",
        "",
        "- Simulated data only. The unbalance, ball and cage amplitudes are synthetic placeholders, not calibrated against measured machines.",
        "- One seed per experiment. The differences between strategies in B are single runs, not a statistical comparison.",
        "- The simulation-time estimate in the GUI is an order of magnitude measured on this laptop, not a guarantee for other machines.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
