"""AUDIT AI-07: leave-bearing-out generalization report for Module B, from the numbers ALREADY
stored in configs/vibration_module_b.yaml (written by experiments/calibrate_module_b.py). Nothing
is re-fitted or re-estimated here; this only makes the comparison against the flat-PSD baseline
explicit, per fold, with a bootstrap interval over the folds.

Usage:  python experiments/report_vibration_generalization.py [--json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "vibration_module_b.yaml"


def report(config_path: Path = CONFIG, n_boot: int = 10_000, seed: int = 0) -> dict:
    data = yaml.safe_load(config_path.read_text())
    folds = data.get("cross_validation") or []
    rows = [{"holdout": f["holdout_code"], "fit_on": list(f["fit_codes"]), "model_rmse": f["model_rmse"], "null_rmse": f["null_rmse"], "ratio": f["model_rmse"] / f["null_rmse"], "model_beats_null": f["model_rmse"] < f["null_rmse"]} for f in folds]
    ratios = np.array([r["ratio"] for r in rows])
    rng = np.random.default_rng(seed)
    boot = rng.choice(ratios, size=(n_boot, len(ratios)), replace=True).mean(axis=1) if len(ratios) else np.array([np.nan])
    return {
        "metric": "RMSE of log10 PSD, held-out bearing (lower is better); null = flat PSD",
        "folds": rows,
        "n_folds": len(rows),
        "n_model_beats_null": int(sum(r["model_beats_null"] for r in rows)),
        "mean_ratio_model_over_null": float(ratios.mean()) if len(ratios) else None,
        "bootstrap_95ci_mean_ratio": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "caveat": "6 folds over 6 bearings of one dataset (KAt-DataCenter), one measured axis -- the interval reflects fold-to-fold spread only, not machine/load/speed variability.",
        "note": data.get("cross_validation_note"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = report()
    if args.json:
        print(json.dumps(r, indent=2))
        return
    print(r["metric"])
    for f in r["folds"]:
        print(f"  holdout {f['holdout']}: model {f['model_rmse']:.3f} vs null {f['null_rmse']:.3f}  ratio {f['ratio']:.2f}  {'BEATS' if f['model_beats_null'] else 'worse than'} null")
    print(f"model beats null in {r['n_model_beats_null']}/{r['n_folds']} folds; mean ratio {r['mean_ratio_model_over_null']:.2f} (bootstrap 95% CI {r['bootstrap_95ci_mean_ratio'][0]:.2f}-{r['bootstrap_95ci_mean_ratio'][1]:.2f})")
    print(r["caveat"])


if __name__ == "__main__":
    main()
