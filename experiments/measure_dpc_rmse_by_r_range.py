"""Measures closed-loop settled RMSE across a range of load resistances R, for each of the 3
shipped DPC checkpoints (v1/v2/v3) -- Patch 9's Tarea 5 follow-up
(docs/patch9_correccion_divergencia_dpc.md Sec. 4): a per-sub-range breakdown, not just the
aggregate on-distribution number, to see where each fine-tuning round actually helped.

Reuses experiments/evaluate_dpc_closed_loop.py::run_closed_loop -- same physics/controller the
dashboard and tests/test_dpc_robustness_grid.py already exercise, just swept over R instead of
evaluated at one point.

Usage:
    python experiments/measure_dpc_rmse_by_r_range.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # running this file directly puts experiments/ (not the repo
# root) on sys.path, so "experiments.evaluate_dpc_closed_loop" wouldn't otherwise be importable
# (same fix as continue_finetune_dpc.py).

from experiments.evaluate_dpc_closed_loop import run_closed_loop
CHECKPOINTS = {
    "v1 (Data4train.mat only)": REPO_ROOT / "configs" / "dpc_trained.weights.h5",
    "v2 (closed-loop fine-tune)": REPO_ROOT / "configs" / "dpc_trained_v2_closed_loop.weights.h5",
    "v3 (extended fine-tune)": REPO_ROOT / "configs" / "dpc_trained_v3_closed_loop.weights.h5",
}
#: Below R*~=3.3707ohm (sim/vsc_system.py::load_feedback_spectral_radius) the plant itself is
#: open-loop unstable -- included here anyway (all checkpoints should show NaN there, confirming
#: the instability is structural, not checkpoint-dependent) plus the full dashboard slider range.
R_GRID = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 8.0064, 10.0, 15.0, 20.0]
N_STEPS = 2000
WARMUP = 50


def settled_rmse(weights_path: Path, r_ohm: float) -> float:
    records = run_closed_loop(weights_path, r_ohm=r_ohm, n_steps=N_STEPS)
    vc_real = np.array([r["vc_real"] for r in records])
    vc_imag = np.array([r["vc_imag"] for r in records])
    vref_real = np.array([r["vref_real"] for r in records])
    vref_imag = np.array([r["vref_imag"] for r in records])
    err = np.hypot(vc_real - vref_real, vc_imag - vref_imag)
    if not np.all(np.isfinite(err)):
        return float("nan")
    return float(np.sqrt(np.mean(err[WARMUP:] ** 2)))


def main():
    header = f"{'R (ohm)':>10} | " + " | ".join(f"{name:>28}" for name in CHECKPOINTS)
    print(header)
    print("-" * len(header))
    for r_ohm in R_GRID:
        row = [f"{r_ohm:>10.4f}"]
        for weights_path in CHECKPOINTS.values():
            rmse = settled_rmse(weights_path, r_ohm)
            row.append(f"{'diverges (NaN)' if np.isnan(rmse) else f'{rmse:.3f} V':>28}")
        print(" | ".join(row))


if __name__ == "__main__":
    main()
