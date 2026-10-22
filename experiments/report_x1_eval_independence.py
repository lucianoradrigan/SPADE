"""X1 (round PLECS/OM): how many independent samples stand behind the "1.000" figures of the promoted
5-class classifier. Counts only -- nothing is re-measured and the evaluation script is unchanged.

For each evaluated condition and class of the two caches the evaluations used:
- seeds (runs), windows (as recorded in the evaluation report when available),
- DISTINCT trajectories of current_r, rpm and torque_nm across those seeds: identical arrays count
  once. The scenario seed changes the vibration channels and the sensor-noise realisation; it does
  not change the simulated current, rpm or torque (experiments/report_classifier_evaluation.py
  docstring, AUDIT_ANSWERS Q15). With noise, current differs only by its noise realisation, so the
  noise-free part is also compared (current minus the seed's added noise is not available, so the
  rpm and torque columns, which carry no noise, are the check).

    python experiments/report_x1_eval_independence.py
Writes docs/audit/ai/x1_eval_independence.{md,json}.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHES = {"classifier_evaluation_5class_w2048 (eval_cache_5class.parquet)": ROOT / "data" / "eval_cache_5class.parquet",
          "AI-N14 benign shifts (ood_eval_cache_custom.parquet)": ROOT / "data" / "ood_eval_cache_custom.parquet"}
EVAL_JSON = ROOT / "docs" / "audit" / "ai" / "classifier_evaluation_5class_w2048.json"
CHANNELS = ("current_r", "rpm", "torque_nm")


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def main():
    ev = json.loads(EVAL_JSON.read_text())
    windows = {}
    for m in ev["models"].values() if isinstance(ev["models"], dict) else ev["models"]:
        for cond, row in (m.get("conditions") or {}).items():
            windows.setdefault(cond, row.get("n"))
    rows = []
    for name, path in CACHES.items():
        df = pd.read_parquet(path, columns=["source_file", "condition", "label", "seed", *CHANNELS])
        for (cond, label), g in df.groupby(["condition", "label"]):
            if cond.startswith("ood_unknown"):
                continue
            runs = list(g.groupby("source_file"))
            row = {"cache": name, "condition": cond, "class": label, "seeds": len(runs), "windows_in_report_all_classes": windows.get(cond)}
            for ch in CHANNELS:
                row[f"distinct_{ch}"] = len({_h(r[ch].to_numpy()) for _, r in runs})
            rows.append(row)
    out = ROOT / "docs" / "audit" / "ai"
    rep = {"command": "python experiments/report_x1_eval_independence.py", "rows": rows}
    (out / "x1_eval_independence.json").write_text(json.dumps(rep, indent=1))
    L = ["# X1 — independent samples behind the evaluations of the promoted 5-class classifier", "",
         f"Command: `{rep['command']}`. Counts only; nothing was re-measured. Raw values: [`x1_eval_independence.json`](x1_eval_independence.json).", "",
         "The scenario seed changes the vibration channels and the sensor-noise realisation, **not** the simulated current, rpm or torque. "
         "A column with 1 distinct trajectory means every seed of that condition and class gave the same signal.", "",
         "| Cache | Condition | Class | Seeds | Windows in the report (all classes) | Distinct current_r | Distinct rpm | Distinct torque |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cache'].split(' (')[0]} | {r['condition']} | {r['class']} | {r['seeds']} | {r['windows_in_report_all_classes'] or '—'} | {r['distinct_current_r']} | {r['distinct_rpm']} | {r['distinct_torque_nm']} |")
    (out / "x1_eval_independence.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
