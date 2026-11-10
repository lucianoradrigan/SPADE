"""B4 §4 (pre-registration approved 2026-11-09): the four ESP32 int8 variants and their selection on VALIDATION.

    python experiments/b4_esp32_variants.py --k4a <run> --k4b <run> --work /mnt/datos/spade/b4_work

Variants (numbered before any result): V0 = (4a, per channel, the current export recipe), V1 = (4a, per tensor, noise-free
stratified calibration from the training split = the repair round's selected configuration, labelled C2/Qt there),
V2 = (4b, per channel), V3 = (4b, per tensor, noise-free stratified). Selection: the repair round's fixed rule
(esp32_repair_preregistration.md §4.1) on the validation split of the B4 3-class training data: highest minimum per-class
recall, then highest macro-F1, then the lowest variant number. Only the selected variant is tested (report_b4.py).
Variants are run-like directories under <work>/variants/V<k>; nothing in configs/classifiers is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import report_ai_n27_regen as RG  # noqa: E402
import report_esp32_repair as ER  # noqa: E402

ER.GRID = REPO / "data" / "b4_train_3class.parquet"  # B4: the 3-class v2 grid + the off-grid healthy runs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k4a", required=True)
    ap.add_argument("--k4b", required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    plan = [("V0", args.k4a, "C0", True), ("V1", args.k4a, "C2", False), ("V2", args.k4b, "C0", True), ("V3", args.k4b, "C2", False)]
    info4a = RG._model_info(args.k4a)
    grid = ER._grid_windows(info4a)
    out = {"grid": str(ER.GRID.relative_to(REPO)), "variants": {}}
    for vk, run, cal, per_channel in plan:
        info = RG._model_info(run)
        cal_X = ER._calibration(cal, run, info, grid)
        counts = None
        if isinstance(cal_X, tuple):
            cal_X, counts = cal_X
        d = ER._export(args.work, vk, run, cal_X, per_channel)
        out["variants"][vk] = {"keras": run, "calibration": "current recipe (per channel)" if cal == "C0" else "noise-free stratified, training split",
                               "per_channel": per_channel, "calibration_counts": counts, "dir": str(d)}
        print(vk, "exported", flush=True)
    Xv, yv, _ = ER._val_windows(info4a, grid)
    classes = info4a["metrics"]["classes"]
    for vk, v in out["variants"].items():
        s = ER._score_dir(v["dir"], Xv, yv, classes)
        s["min_recall"] = min(r for r in s["recall"].values() if r is not None)
        v["validation"] = s
        print(vk, f"min recall {s['min_recall']:.3f} macro-F1 {s['macro_f1']:.3f}", flush=True)
    ranked = sorted(out["variants"], key=lambda k: (-out["variants"][k]["validation"]["min_recall"], -out["variants"][k]["validation"]["macro_f1"], int(k[1:])))
    out["ranking"], out["selected"], out["n_val_windows"], out["wall_s"] = ranked, ranked[0], int(len(Xv)), round(time.time() - t0, 1)
    (args.work / "esp32_selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print("selected", ranked[0])


if __name__ == "__main__":
    main()
