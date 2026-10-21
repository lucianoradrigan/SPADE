"""RunComparison (spec Sec. 7, 9): runs compared on the intersection of their condition grids.
The comparison itself is computed by driveflow.runs.compare.compare_runs; this component only
shows it, including "not comparable" and "no improvement" with full detail."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from driveflow.viz.components.state_panel import ComponentView, render_state


def comparison_view(result: dict | None, loading: bool = False) -> ComponentView:
    if loading:
        return ComponentView("loading", "Loading saved results…")
    if result is None or (not result["comparable"] and result.get("reason") == "Select at least two runs."):
        return ComponentView("empty", "Select at least two runs")
    if not result["comparable"]:
        return ComponentView("error", "Runs are not comparable", result.get("reason"))
    excluded = f" {result['n_excluded']} condition(s) outside the shared grid were left out." if result["n_excluded"] else ""
    if not all(v["improved"] for v in result["verdicts"]):
        return ComponentView("negative", "Candidate did not improve", "No improvement over the baseline was measured for at least one candidate." + excluded)
    return ComponentView("success", "Comparison ready", f"{result['n_shared']} shared condition(s).{excluded}")


def run_comparison(result: dict | None, loading: bool = False) -> None:
    view = comparison_view(result, loading)
    render_state(view)
    if not result or not result["comparable"]:
        return
    direction = "lower is better" if result["lower_is_better"] else "higher is better"
    st.caption(f"Metric: {result['metric']} ({direction}). Baseline = first selected run. 'Failed', 'Skipped' and 'Missing' are shown as such, never imputed.")
    table = pd.DataFrame([{"Condition": str(r["condition"]), **{k: v for k, v in r.items() if k != "condition"}} for r in result["rows"]])
    st.dataframe(table.astype(str), hide_index=True, width="stretch")
    verdicts = pd.DataFrame([{"Candidate": v["candidate"], "Paired conditions": v["n_paired"], "Better": v["better"], "Worse": v["worse"],
                              "New failures": v["new_failures"], "Baseline mean": v["baseline_mean"], "Candidate mean": v["candidate_mean"],
                              "Verdict": "Improved" if v["improved"] else "No improvement"} for v in result["verdicts"]])
    st.dataframe(verdicts, hide_index=True, width="stretch")
