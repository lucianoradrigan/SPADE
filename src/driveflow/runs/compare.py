"""Run comparison on a shared condition grid -- expansion task E0 (spec Sec. 7).

Rules from the spec: runs whose condition grids do not overlap are "not comparable"; with a
partial overlap only the intersection is compared and the number of conditions left out is
reported; a missing or failed condition is shown as such and never imputed; a candidate that is
not better than the baseline is reported as "no improvement", with the same detail as a win.
"""

from __future__ import annotations

import math
import statistics

from driveflow.runs.schemas import Run, canonical_json


def _key(condition) -> str:
    return canonical_json(condition)


def numeric_metrics(runs: list[Run]) -> list[str]:
    names = set()
    for run in runs:
        for c in run.conditions:
            for k, v in (c.get("metrics") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    names.add(k)
    return sorted(names)


def compare_runs(runs: list[Run], metric: str, lower_is_better: bool = True) -> dict:
    """runs[0] is the baseline. Returns {comparable, reason?, rows, n_shared, n_excluded, verdicts}.
    Each row value is a float, "Failed", "Skipped" or "Missing" (metric absent)."""
    if len(runs) < 2:
        return {"comparable": False, "reason": "Select at least two runs.", "rows": [], "n_shared": 0, "n_excluded": 0, "verdicts": []}
    kinds = {r.kind for r in runs}
    if len(kinds) > 1:
        return {"comparable": False, "reason": f"Runs are not comparable: different experiment kinds ({', '.join(sorted(kinds))}).",
                "rows": [], "n_shared": 0, "n_excluded": 0, "verdicts": []}
    grids = [{_key(c["condition"]): c for c in r.conditions} for r in runs]
    shared = set(grids[0]).intersection(*grids[1:])
    union = set().union(*grids)
    if not shared:
        return {"comparable": False, "reason": "These runs do not share the same evaluation grid.", "rows": [], "n_shared": 0,
                "n_excluded": len(union), "verdicts": []}

    def value(c):
        if c["status"] != "ok":
            return c["status"].capitalize()
        v = (c.get("metrics") or {}).get(metric)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else "Missing"

    order = [k for k in grids[0] if k in shared]  # baseline order
    rows = [{"condition": grids[0][k]["condition"], **{r.run_id: value(g[k]) for r, g in zip(runs, grids)}} for k in order]
    base = runs[0].run_id
    verdicts = []
    for cand in runs[1:]:
        paired = [(row[base], row[cand.run_id]) for row in rows if isinstance(row[base], float) and isinstance(row[cand.run_id], float)]
        better = sum((c < b) if lower_is_better else (c > b) for b, c in paired)
        worse = sum((c > b) if lower_is_better else (c < b) for b, c in paired)
        new_failures = sum(isinstance(row[base], float) and row[cand.run_id] == "Failed" for row in rows)
        if paired:
            mb, mc = statistics.fmean(b for b, _ in paired), statistics.fmean(c for _, c in paired)
            improved = ((mc < mb) if lower_is_better else (mc > mb)) and new_failures == 0
        else:
            mb = mc = None
            improved = False
        verdicts.append({"candidate": cand.run_id, "baseline": base, "n_paired": len(paired), "better": better, "worse": worse,
                         "new_failures": new_failures, "baseline_mean": mb, "candidate_mean": mc, "improved": improved})
    return {"comparable": True, "rows": rows, "n_shared": len(shared), "n_excluded": len(union) - len(shared), "verdicts": verdicts,
            "metric": metric, "lower_is_better": lower_is_better}
