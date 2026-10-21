"""TPS modulation suggestion from a DAB performance model -- task E9 preparation (round 2026-10-21, D8).

``suggest`` evaluates the trained performance model (models/dab) on a grid of TPS inner phase shifts
(D1, D2) at one operating point and returns the grid point with the lowest predicted objective
(default: the inductor RMS current). It sits next to the single-phase-shift baseline D1 = D2 = 1.
It is a model prediction on a grid, never a verified operating point: the GUI shows it as
AI SUGGESTION — REQUIRES VERIFICATION, and "Compare suggested vs baseline" must run new simulations.
Only the model's own training range is searched: a point outside it is refused, not extrapolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUGGESTION_LABEL = "AI SUGGESTION — REQUIRES VERIFICATION"


def load_run(run_dir) -> tuple:
    from driveflow.models.dab.builder import build_dab_model
    from driveflow.models.dab.schemas import load_dab_config

    run_dir = Path(run_dir)
    config = load_dab_config(run_dir / "config.yaml")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model = build_dab_model(config)
    model.load_weights(run_dir / "model.weights.h5")
    return model, config, metrics


def predict(model, metrics: dict, rows: np.ndarray) -> np.ndarray:
    x = (np.asarray(rows, np.float32) - np.asarray(metrics["x_mean"])) / np.asarray(metrics["x_std"])
    return np.asarray(model(x.astype(np.float32), training=False)) * np.asarray(metrics["y_std"]) + np.asarray(metrics["y_mean"])


def suggest(model, metrics: dict, operating_point: dict, objective: str = "irms", n: int = 41, d_range: tuple = (0.0, 1.0),
            training_range: dict | None = None) -> dict:
    inputs, outputs = metrics["inputs"], metrics["outputs"]
    if objective not in outputs:
        raise ValueError(f"objective must be one of {outputs}, got {objective!r}")
    for k, v in operating_point.items():
        if training_range and k in training_range and not training_range[k][0] <= v <= training_range[k][1]:
            raise ValueError(f"{k} = {v:g} is outside the model's training range {training_range[k]}; no suggestion is extrapolated")
    grid = np.linspace(*d_range, n)
    D1, D2 = np.meshgrid(grid, grid, indexing="ij")
    rows = []
    for d1, d2 in zip(D1.ravel(), D2.ravel()):
        row = {**operating_point, "D1": d1, "D2": d2}
        rows.append([row[c] for c in inputs])
    pred = predict(model, metrics, np.array(rows))
    j = outputs.index(objective)
    k = int(np.argmin(pred[:, j]))
    base = predict(model, metrics, np.array([[{**operating_point, "D1": 1.0, "D2": 1.0}[c] for c in inputs]]))[0]
    return {"label": SUGGESTION_LABEL, "operating_point": dict(operating_point), "objective": objective,
            "suggested": {"D1": float(D1.ravel()[k]), "D2": float(D2.ravel()[k]), **{o: float(pred[k, i]) for i, o in enumerate(outputs)}},
            "baseline_sps": {"D1": 1.0, "D2": 1.0, **{o: float(base[i]) for i, o in enumerate(outputs)}},
            "grid": {"D1": grid.tolist(), "D2": grid.tolist(), objective: pred[:, j].reshape(n, n).tolist()}}
