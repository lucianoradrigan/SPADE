"""Forecasters on held-out operating conditions (docs/AI_TECHNICAL_AUDIT.md AI-N11).

Evaluates each vsc_dpc forecaster run on the windows of the TEST conditions of the promoted PC
run (its metrics.json "test_groups": load resistance x reference magnitude pairs never seen in
training, data/vsc_dpc_grid_dataset.parquet), next to the persistence baseline (repeat the last
context sample). The runs of 2026-09 were trained on phase-shifted copies of ONE condition.

Usage:
    python experiments/report_forecaster_condition_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "vsc_dpc_grid_dataset.parquet"
RUNS = ("pc_full/2026-10-19_run01", "rpi5_edge/2026-10-19_run01", "esp32_tiny/2026-10-19_run01",
        "pc_full/2026-09-01_run01", "rpi5_edge/2026-09-13_run01", "esp32_tiny/2026-09-13_run02")


def main():
    from driveflow.models.common import build_direct_forecast_windows
    from driveflow.models.regressors.builder import build_forecaster
    from driveflow.models.regressors.schemas import load_forecaster_config

    root = REPO_ROOT / "configs" / "regressors"
    test_groups = json.loads((root / RUNS[0] / "metrics.json").read_text())["test_groups"]
    df = pd.read_parquet(DATASET)
    df = df[df["source_file"].isin(test_groups)]
    out = {"dataset": str(DATASET.relative_to(REPO_ROOT)), "test_groups": test_groups, "runs": {}}
    for run in RUNS:
        d = root / run
        cfg = load_forecaster_config(d / "config.yaml")
        m = json.loads((d / "metrics.json").read_text())
        X, Y, _ = build_direct_forecast_windows(df, m["channels"], cfg.input_window, cfg.horizon)
        model = build_forecaster(cfg, n_channels=len(m["channels"]))
        model.load_weights(d / "model.weights.h5")
        pred = model.predict(X, verbose=0).reshape(Y.shape)
        persistence = np.repeat(X[:, -1:, :], cfg.horizon, axis=1)
        out["runs"][run] = {"n_windows": int(len(X)), "rmse": float(np.sqrt(np.mean((pred - Y) ** 2))),
                            "persistence_rmse": float(np.sqrt(np.mean((persistence - Y) ** 2))), "recorded_test_rmse": m.get("test_rmse")}
        print(run, out["runs"][run])
    path = REPO_ROOT / "docs" / "audit" / "ai" / "forecaster_condition_split.json"
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
