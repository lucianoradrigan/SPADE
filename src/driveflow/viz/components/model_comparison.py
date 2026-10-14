"""Base vs. fine-tuned comparison on the SAME independent test set (AUDIT AI-05/T-AI-07): metric
table with the trivial baselines next to it, plus the confusion matrix or per-horizon RMSE."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driveflow.viz.theme.plotly import PLOTLY_CONFIG, series_color, style_figure


def metric_rows(report: dict) -> list:
    m, b = report["metrics"], report["baseline_metrics"]
    if report["task"] == "classification":
        keys = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1")]
    else:
        keys = [("RMSE (normalized)", "rmse"), ("MAE (normalized)", "mae")]
    rows = [{"Metric": label, "Base (same test set)": b.get(k), "Fine-tuned": m.get(k)} for label, k in keys]
    if report["task"] == "classification":
        rows.append({"Metric": "Majority-class predictor macro-F1", "Base (same test set)": m.get("majority_baseline_macro_f1"), "Fine-tuned": None})
    if report["task"] == "regression":
        rows.append({"Metric": "Persistence baseline RMSE", "Base (same test set)": m.get("persistence_rmse"), "Fine-tuned": None})
    return rows


def model_comparison(report: dict, test_evaluation: dict, classes: list, key: str) -> None:
    """report: validator report; test_evaluation: TL result's test_evaluation; classes: sorted
    class names of the base model (for the confusion matrix labels)."""
    st.markdown("##### Base vs. fine-tuned — independent test set")
    st.dataframe(pd.DataFrame(metric_rows(report)), hide_index=True, width="stretch")
    t = test_evaluation
    st.caption(f"{t['n_test_windows']} test windows, evaluated once after training ({'same' if t['same_test_for_base'] else 'DIFFERENT'} windows and preprocessing for both models).")
    if report["task"] == "classification" and report["metrics"].get("confusion_matrix"):
        st.markdown("Confusion matrix (fine-tuned; rows = true, columns = predicted)")
        st.dataframe(pd.DataFrame(report["metrics"]["confusion_matrix"], index=classes, columns=classes), width="stretch")
    if report["task"] == "regression" and report["metrics"].get("rmse_per_step"):
        fig = go.Figure(go.Scatter(y=report["metrics"]["rmse_per_step"], mode="lines", name="fine-tuned", line=dict(color=series_color("candidate"))))
        if report["baseline_metrics"].get("rmse_per_step"):
            fig.add_trace(go.Scatter(y=report["baseline_metrics"]["rmse_per_step"], mode="lines", name="base", line=dict(color=series_color("base"), dash="dash")))
        style_figure(fig, height=240, x_title="Forecast horizon step", y_title="RMSE (normalized)")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)
