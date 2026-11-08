"""dc_perm_ex_v2 step 1, §6 of the pre-registration (dc_perm_ex_v2_convergence_preregistration.md): dataset
equivalence of variant B (``dc_converter_model="dcm_exact"``), run only because B met §4.

10 runs of the 3-class dcm grid (data/diagnosis_grid_dataset_dcm.parquet), regenerated with B through the
generator's own jobs (build_jobs(3, 2.5, 7)) -- only the converter changes:
- 5 affected: the first FAULT acquisition, in build_jobs order, of each affected 3-class cell
  (80, x1.0), (80, x1.5), (80, x2.0), (100, x1.5), (100, x2.0);
- 5 healthy: the first healthy acquisition at 80, 125, 150, 200 and 220 rad/s.
Criterion (a): per run NRMS <= 1e-3 for current_r, rpm, torque_nm against the stored dcm x1000 run.
Criterion (b): top-1 agreement >= 0.99 of the promoted PC (Keras) and RPi5 (float16 TFLite) on all windows of
the 10 runs (known classes), stored x1000 vs B.

    python experiments/report_dc_perm_ex_v2_equivalence.py [--workers 10]
Writes docs/audit/assistant/dc_perm_ex_v2_equivalence.{md,json} (simulated).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" if (REPO / "data" / "diagnosis_grid_dataset_dcm.parquet").exists() else Path("/home/seven/post_doc/plataform_proposed/SPADE-dev/data")
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "docs" / "audit" / "assistant" / "dc_perm_ex_v2_equivalence"
AFFECTED_CELLS = ((80.0, 1.0), (80.0, 1.5), (80.0, 2.0), (100.0, 1.5), (100.0, 2.0))
HEALTHY_SPEEDS = (80.0, 125.0, 150.0, 200.0, 220.0)
CHANNELS = ("current_r", "rpm", "torque_nm")
NRMS_MAX = 1e-3
AGREEMENT_MIN = 0.99
MODELS = {"PC": ("configs/classifiers/pc_server_5class_w2048/2026-11-04_run01", "keras"),
          "RPi5": ("configs/classifiers/rpi5_resnet1d_se_5class_w2048/2026-11-04_run01", "tflite")}


def _select(jobs: list) -> list:
    out = []
    for speed, sev in AFFECTED_CELLS:
        out.append(next(j for j in jobs if j["fault"] is not None and j["speed"] == speed and j["severity_scale"] == sev))
    for speed in HEALTHY_SPEEDS:
        out.append(next(j for j in jobs if j["fault"] is None and j["speed"] == speed))
    return out


def _sid(job) -> str:
    return f"grid_{job['fault'] or 'normal'}_w{job['speed']:.0f}_s{job['severity_scale']:.2f}_seed{job['seed']}"


def _nrms(a, ref):
    rng = float(ref.max() - ref.min())
    d = float(np.sqrt(np.mean((a - ref) ** 2)))
    return d / rng if rng > 0 else d


def main(argv=None) -> None:
    from generate_diagnosis_grid_dataset import _simulate, build_jobs

    import report_ai_n27_regen as RG

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    sel = _select(build_jobs(3, 2.5, 7))
    jobs = [{**j, "dc_converter_model": "dcm_exact"} for j in sel]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        frames = pool.map(_simulate, jobs)
    stored = pd.read_parquet(DATA / "diagnosis_grid_dataset_dcm.parquet")
    runs, new_all, old_all = [], [], []
    for j, new in zip(sel, frames):
        sid = _sid(j)
        old = stored[stored["source_file"].astype(str) == sid].reset_index(drop=True)
        assert len(old) == len(new), f"{sid}: {len(old)} stored rows vs {len(new)} regenerated"
        assert np.array_equal(old["timestamp_s"].to_numpy(), new["timestamp_s"].to_numpy()), f"{sid}: timestamps differ"
        r = {"source": sid, "fault": j["fault"], "speed": j["speed"], "severity_scale": j["severity_scale"],
             "nrms": {c: _nrms(new[c].to_numpy(), old[c].to_numpy()) for c in CHANNELS}}
        r["pass"] = all(v <= NRMS_MAX for v in r["nrms"].values())
        runs.append(r)
        new = new.copy()
        new["source_file"] = old["source_file"].iloc[0]
        new_all.append(new)
        old_all.append(old)
    new_df, old_df = pd.concat(new_all, ignore_index=True), pd.concat(old_all, ignore_index=True)
    agreement = {}
    for tier, (run, kind) in MODELS.items():
        info = RG._model_info(run)
        classes = info["metrics"]["classes"]
        Xo, yo, go = RG._windows(old_df, classes, info, 100_000)
        Xn, yn, gn = RG._windows(new_df, classes, info, 100_000)
        assert len(Xo) == len(Xn) and list(go) == list(gn), f"{tier}: windows not paired"
        po, pn = RG._predict(run, kind, Xo).argmax(1), RG._predict(run, kind, Xn).argmax(1)
        agreement[tier] = {"run": run, "artifact": kind, "n_windows": int(len(Xo)), "agreement": float(np.mean(po == pn)),
                           "accuracy_stored": float(np.mean(po == np.array([classes.index(c) for c in yo]))),
                           "accuracy_B": float(np.mean(pn == np.array([classes.index(c) for c in yn])))}
        agreement[tier]["pass"] = agreement[tier]["agreement"] >= AGREEMENT_MIN
    res = {"command": "python experiments/report_dc_perm_ex_v2_equivalence.py" + (f" --workers {args.workers}" if args.workers != 10 else ""), "simulated": True,
           "preregistration": "docs/audit/assistant/dc_perm_ex_v2_convergence_preregistration.md §6", "runs": runs, "agreement": agreement,
           "criterion_a": all(r["pass"] for r in runs), "criterion_b": all(a["pass"] for a in agreement.values())}
    res["verdict"] = "MET" if res["criterion_a"] and res["criterion_b"] else "NOT MET"
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n")
    L = ["# dc_perm_ex_v2 step 1 — dataset equivalence of variant B (pre-registration §6)", "",
         f"**Simulated.** Command: `{res['command']}`. Raw: `dc_perm_ex_v2_equivalence.json`. **Verdict: {res['verdict']}.**", "",
         f"(a) Per run, NRMS(B vs stored dcm ×1000) ≤ {NRMS_MAX:g} for current_r, rpm, torque_nm: **{'met' if res['criterion_a'] else 'not met'}**.", "",
         "| Run | NRMS current_r | NRMS rpm | NRMS torque_nm | Pass |", "|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['source']} | {r['nrms']['current_r']:.1e} | {r['nrms']['rpm']:.1e} | {r['nrms']['torque_nm']:.1e} | {'yes' if r['pass'] else '**no**'} |")
    L += ["", f"(b) Top-1 agreement ≥ {AGREEMENT_MIN:g} of the promoted classifiers, stored ×1000 vs B: **{'met' if res['criterion_b'] else 'not met'}**.", "",
          "| Model | Artifact | Windows | Agreement | Accuracy stored / B (descriptive) |", "|---|---|---|---|---|"]
    for t, a in agreement.items():
        L.append(f"| {t} (`{a['run']}`) | {a['artifact']} | {a['n_windows']} | {a['agreement']:.4f} | {a['accuracy_stored']:.3f} / {a['accuracy_B']:.3f} |")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
