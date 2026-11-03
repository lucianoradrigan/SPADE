"""S21-1 (decision of 2026-11-02): which diagnosis datasets, and which TRAINING windows of the promoted
classifiers, contain the one-quadrant converter effect of finding S-21. Reads data only; no plant code
is run or changed.

Per acquisition (``source_file``) of every DC diagnosis parquet in data/:
  (a) samples (= control periods, 10 kHz) with current_r < 0, and with current_r < -5 A (robust to
      the sensor noise some grids add after the run);
  (b) whether the mean speed over the second half is more than 2 % above the reference (grid_speed);
crossed with the S-19 list (docs/audit/plecs_om/r3_speed_tracking.json) and broken down by class,
speed and severity. For the promoted dc_motor classifiers (configs/registry.yaml), the training
windows are rebuilt exactly as models.common.windowing.build_classification_windows takes them
(cap_per_class from run_metadata.json) and matched to the split's train group hashes, giving the
fraction of TRAINING windows that lie in affected acquisitions and that themselves contain a
negative-current sample.

    python experiments/report_s21_dataset_scan.py
Writes docs/audit/assistant/s21_dataset_scan.{md,json}.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from driveflow.models.common.windowing import window_starts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "assistant" / "s21_dataset_scan"
DATASETS = ["data/diagnosis_grid_dataset.parquet", "data/diagnosis_grid_dataset_rk45.parquet", "data/diagnosis_grid_dataset_5class.parquet",
            "data/diagnosis_grid_dataset_5class_rk45.parquet", "data/diagnosis_dataset.parquet", "data/diagnosis_dataset_c1_5s.parquet"]
COLS = ["source_file", "label", "timestamp_s", "current_r", "voltage_v", "rpm", "grid_speed", "grid_severity_scale", "grid_electrical_noise_pct", "grid_psi_e", "grid_j_rotor", "plant_config_id"]
NEG_STRONG_A = -5.0
#: Noise-free signature of the converter's negative-current branch (S-21): the full supply applied
#: while the speed is ALREADY above its reference (the PI asks for less torque there, so full voltage
#: can only come from the branch). voltage_v and rpm carry no added noise; current_r does in the grids
#: with grid_electrical_noise_pct > 0, so a negative current alone is not evidence there.
FULL_SUPPLY_V = 64.9
OVER_FRAC = 0.02
RPM = 2 * math.pi / 60


def _hash(g) -> str:
    return hashlib.sha256(str(g).encode()).hexdigest()[:12]


def load(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(ROOT / path).schema_arrow.names
    df = pd.read_parquet(ROOT / path, columns=[c for c in COLS if c in names])
    if "plant_config_id" in df.columns:
        df = df[df["plant_config_id"].isin(["dc_perm_ex_v1"])]
    return df


def scan_acquisitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("source_file", sort=False):
        i = g["current_r"].to_numpy()
        w = g["rpm"].to_numpy() * RPM
        u = g["voltage_v"].to_numpy()
        ref = float(g["grid_speed"].iloc[0]) if "grid_speed" in g.columns and pd.notna(g["grid_speed"].iloc[0]) else math.nan
        half = w[len(w) // 2:]
        rel = (half.mean() - ref) / ref if ref and math.isfinite(ref) else math.nan
        rows.append({"source_file": src, "label": g["label"].iloc[0], "speed": ref, "severity": float(g["grid_severity_scale"].iloc[0]) if "grid_severity_scale" in g.columns else math.nan,
                     "noise_pct": float(g["grid_electrical_noise_pct"].iloc[0]) if "grid_electrical_noise_pct" in g.columns else math.nan,
                     "psi_e": float(g["grid_psi_e"].iloc[0]) if "grid_psi_e" in g.columns else math.nan,
                     "n": len(g), "n_branch": int(((u >= FULL_SUPPLY_V) & (w > ref)).sum()) if math.isfinite(ref) else 0, "n_neg": int((i < 0).sum()), "n_neg_strong": int((i < NEG_STRONG_A).sum()), "min_current_a": float(i.min()),
                     "mean_speed_second_half": float(half.mean()), "rel_speed_error": float(rel), "over_2pct": bool(rel > OVER_FRAC) if math.isfinite(rel) else False})
    a = pd.DataFrame(rows)
    a["neg_current"] = a["n_neg"] > 0
    a["neg_strong"] = a["n_neg_strong"] > 0
    a["branch"] = a["n_branch"] > 0
    return a


def training_windows(df: pd.DataFrame, window: int, cap: int, train_hashes: set, acq: pd.DataFrame) -> dict:
    """Rebuild the windows build_classification_windows takes (same group order, same caps), keep the
    train-split ones, and count those in affected acquisitions / containing a negative current."""
    flags = acq.set_index("source_file")
    out = defaultdict(int)
    per_class = defaultdict(lambda: defaultdict(int))
    for cls in sorted(df["label"].unique()):
        cls_df = df[df["label"] == cls]
        groups = cls_df["source_file"].unique()
        per_group = max(20, math.ceil(cap / len(groups)))
        collected = 0
        for src in groups:
            if collected >= cap:
                break
            sub = cls_df[cls_df["source_file"] == src]
            if len(sub) < window:
                continue
            starts = window_starts(len(sub), window, sub["timestamp_s"].to_numpy(dtype=float))
            n = min(per_group, len(starts), cap - collected)
            collected += n
            if _hash(src) not in train_hashes:
                continue
            i = sub["current_r"].to_numpy()
            branch = (sub["voltage_v"].to_numpy() >= FULL_SUPPLY_V) & (sub["rpm"].to_numpy() * RPM > float(sub["grid_speed"].iloc[0]))
            for s in starts[:n]:
                seg = i[s:s + window]
                if flags.at[src, "branch"]:
                    out["in_branch_acq"] += 1
                    per_class[cls]["in_branch_acq"] += 1
                if branch[s:s + window].any():
                    out["window_has_branch"] += 1
                    per_class[cls]["window_has_branch"] += 1
                out["train_windows"] += 1
                per_class[cls]["train_windows"] += 1
                if flags.at[src, "neg_current"]:
                    out["in_neg_current_acq"] += 1
                    per_class[cls]["in_neg_current_acq"] += 1
                if flags.at[src, "neg_strong"]:
                    out["in_neg_strong_acq"] += 1
                    per_class[cls]["in_neg_strong_acq"] += 1
                if flags.at[src, "over_2pct"]:
                    out["in_over_2pct_acq"] += 1
                    per_class[cls]["in_over_2pct_acq"] += 1
                if (seg < 0).any():
                    out["window_has_neg_current"] += 1
                    per_class[cls]["window_has_neg_current"] += 1
                if (seg < NEG_STRONG_A).any():
                    out["window_has_neg_strong"] += 1
                    per_class[cls]["window_has_neg_strong"] += 1
    return {**out, "per_class": {k: dict(v) for k, v in per_class.items()}}


def summarize(a: pd.DataFrame) -> dict:
    def frac(mask):
        return {"n": int(mask.sum()), "of": int(len(a)), "pct": round(100.0 * mask.mean(), 2) if len(a) else None}

    by = {}
    for key in ("label", "speed", "severity"):
        if key in a.columns and a[key].notna().any():
            by[key] = {str(k): {"acquisitions": int(len(g)), "branch": int(g["branch"].sum()), "neg_current": int(g["neg_current"].sum()), "neg_strong": int(g["neg_strong"].sum()),
                                "over_2pct": int(g["over_2pct"].sum()), "max_rel_speed_error_pct": round(100 * float(g["rel_speed_error"].max()), 1)}
                       for k, g in a.groupby(key)}
    return {"acquisitions": int(len(a)), "branch": frac(a["branch"]), "branch_without_over_2pct": frac(a["branch"] & ~a["over_2pct"]),
            "over_2pct_without_branch": frac(a["over_2pct"] & ~a["branch"]), "neg_strong_without_branch": frac(a["neg_strong"] & ~a["branch"]), "neg_current": frac(a["neg_current"]), "neg_strong": frac(a["neg_strong"]), "over_2pct": frac(a["over_2pct"]),
            "over_2pct_and_neg_strong": frac(a["over_2pct"] & a["neg_strong"]), "over_2pct_without_neg_current": frac(a["over_2pct"] & ~a["neg_current"]),
            "neg_strong_without_over_2pct": frac(a["neg_strong"] & ~a["over_2pct"]), "by": by}


def main() -> None:
    s19 = json.loads((ROOT / "docs" / "audit" / "plecs_om" / "r3_speed_tracking.json").read_text())
    s19_runs = {k: {r["run"] for r in v["euler_off_runs"]} for k, v in s19.items()}
    registry = yaml.safe_load((ROOT / "configs" / "registry.yaml").read_text())["entries"]
    models = {k: v for k, v in registry.items() if k.startswith("dc_motor/")}
    result = {"command": "python experiments/report_s21_dataset_scan.py", "simulated": True, "thresholds": {"neg_strong_a": NEG_STRONG_A, "over_frac": OVER_FRAC},
              "datasets": {}, "models": {}}
    acq_by_ds = {}
    for path in DATASETS:
        if not (ROOT / path).exists():
            result["datasets"][path] = {"note": "no ejecutado: file not found"}
            continue
        df = load(path)
        acq = scan_acquisitions(df)
        acq_by_ds[path] = (df, acq)
        s = summarize(acq)
        key = "5class" if "5class" in path else "3class"
        if "grid" in path and key in s19_runs:
            listed = acq["source_file"].str.replace(r"\.parquet$", "", regex=True).isin(s19_runs[key]) | acq["source_file"].isin(s19_runs[key])
            s["s19_listed"] = int(listed.sum())
            s["s19_listed_with_neg_strong"] = int((listed & acq["neg_strong"]).sum())
            s["s19_listed_with_neg_current"] = int((listed & acq["neg_current"]).sum())
            s["s19_listed_with_branch"] = int((listed & acq["branch"]).sum())
        result["datasets"][path] = s
        print(path, json.dumps({k: s[k] for k in ("acquisitions", "branch", "neg_current", "neg_strong", "over_2pct")}), flush=True)
    for name, run_dir in models.items():
        meta = json.loads((ROOT / run_dir / "run_metadata.json").read_text())
        metrics = json.loads((ROOT / run_dir / "metrics.json").read_text())
        cfg = yaml.safe_load((ROOT / run_dir / "config.yaml").read_text())
        path = meta["dataset_path"]
        if path not in acq_by_ds:
            result["models"][name] = {"run": run_dir, "dataset": path, "note": "no ejecutado: dataset not scanned"}
            continue
        df, acq = acq_by_ds[path]
        tw = training_windows(df, int(cfg["input_window"]), int(meta.get("cap_per_class", 500)), set(metrics["split"]["group_hashes"]["train"]), acq)
        n = tw.get("train_windows", 0)
        expected = metrics["split"]["n_windows"]["train"]
        res = {"run": run_dir, "dataset": path, "window": cfg["input_window"], "cap_per_class": meta.get("cap_per_class"), "train_windows_rebuilt": n,
               "train_windows_recorded": expected, "rebuild_matches": n == expected}
        for k in ("in_branch_acq", "window_has_branch", "in_neg_current_acq", "in_neg_strong_acq", "in_over_2pct_acq", "window_has_neg_current", "window_has_neg_strong"):
            res[k] = {"n": tw.get(k, 0), "pct": round(100.0 * tw.get(k, 0) / n, 2) if n else None}
        res["per_class"] = tw["per_class"]
        result["models"][name] = res
        print(name, json.dumps({k: res[k] for k in res if k not in ("per_class",)}), flush=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    write_md(result)


def write_md(result: dict) -> None:
    lines = ["# S21-1 — dataset scan for the one-quadrant converter effect (S-21)", "",
             "**Simulated data; read only** (no plant code run or changed). Decision S21-1 of 2026-11-02.", "",
             f"- **Command:** `{result['command']}`. Raw values: [`s21_dataset_scan.json`](s21_dataset_scan.json).",
             f"- **Signature of the converter branch (noise-free):** full supply (voltage_v ≥ {FULL_SUPPLY_V} V) while the speed is already above `grid_speed`. "
             "Validated on noise-free simulations: it coincides sample by sample with `current_r` < 0 (dc_pmg132_v1 at 120 rad/s: 2626 vs 2627; stock motor, outer race "
             "at high severity, 80 rad/s: 2750 vs 2750, final speed 128.6 rad/s; healthy stock runs at 120 and 220 rad/s: 0 and 0).",
             "- **`current_r` < 0 is NOT used as evidence** in the grids: they add sensor noise to `current_r` after the run (`grid_electrical_noise_pct`, up to ≈ 95 % of the run's std), "
             "and every healthy acquisition with a negative current has noise ≥ 31 %. Those counts are kept in the JSON for reference.",
             f"- Speed: mean over the second half more than {OVER_FRAC:.0%} above `grid_speed` (S-19's symptom).",
             "", "## Datasets", "",
             "| Dataset | Acquisitions | With the converter branch | Speed > +2 % | Branch without > +2 % | > +2 % without branch | S-19 listed (with branch) |",
             "|---|---|---|---|---|---|---|"]
    for p, s in result["datasets"].items():
        if "acquisitions" not in s:
            lines.append(f"| `{p}` | {s['note']} | | | | | |")
            continue
        s19 = f"{s['s19_listed']} ({s['s19_listed_with_branch']})" if "s19_listed" in s else "—"
        lines.append(f"| `{p}` | {s['acquisitions']} | {s['branch']['n']} ({s['branch']['pct']} %) | {s['over_2pct']['n']} ({s['over_2pct']['pct']} %) | "
                     f"{s['branch_without_over_2pct']['n']} | {s['over_2pct_without_branch']['n']} | {s19} |")
    lines += ["", "## Training windows of the promoted dc_motor classifiers", "",
              "| Model | Dataset | Train windows (rebuilt / recorded) | In acquisitions with the branch | Windows that contain the branch | In acquisitions > +2 % speed |",
              "|---|---|---|---|---|---|"]
    for name, m in result["models"].items():
        if "note" in m:
            lines.append(f"| `{name}` | `{m['dataset']}` | {m['note']} | | | |")
            continue
        lines.append(f"| `{name}` | `{m['dataset']}` | {m['train_windows_rebuilt']} / {m['train_windows_recorded']} | {m['in_branch_acq']['n']} ({m['in_branch_acq']['pct']} %) | "
                     f"{m['window_has_branch']['n']} ({m['window_has_branch']['pct']} %) | {m['in_over_2pct_acq']['n']} ({m['in_over_2pct_acq']['pct']} %) |")
    lines += ["", "Per class (train windows that contain the branch / train windows):", ""]
    for name, m in result["models"].items():
        if "per_class" in m:
            lines.append(f"- `{name}`: " + ", ".join(f"{c} {v.get('window_has_branch', 0)}/{v['train_windows']}" for c, v in sorted(m["per_class"].items())))
    lines += ["", "## Breakdown (Euler grids)", ""]
    for p in ("data/diagnosis_grid_dataset_5class.parquet", "data/diagnosis_grid_dataset.parquet"):
        s = result["datasets"].get(p, {})
        for key in ("label", "speed", "severity"):
            if key not in s.get("by", {}):
                continue
            lines += [f"**`{p}` by {key}**", "", f"| {key} | Acquisitions | With the branch | Speed > +2 % | Max speed error |", "|---|---|---|---|---|"]
            lines += [f"| {k} | {v['acquisitions']} | {v['branch']} | {v['over_2pct']} | {v['max_rel_speed_error_pct']} % |" for k, v in s["by"][key].items()]
            lines.append("")
    lines += ["## Not executed", "",
              "- How much the branch changes the classifiers' input signals and their accuracy: no ejecutado (needs the converged reference of S21-2; no dataset regenerated, no model retrained, S21-4).",
              "- Non-DC datasets (PMSM, VSC): not applicable (no one-quadrant converter)."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
