"""GUI boundary for the DAB tab (task E9 preparation, round 2026-10-21, D8). No DAB plant runs in
SPADE yet: the circuit is a PLECS file (Fund. DAB_sample1.plecs) and no PLECS run exists. This module
only (1) draws the bridge-voltage pattern implied by the chosen phase shifts, a deterministic
illustration and not a simulation, and (2) reports which data and model would be needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

STRATEGIES = {
    "SPS": "Single phase shift: D1 = D2 = 1, only the outer shift D0",
    "DPS": "Dual phase shift: D1 = D2 < 1",
    "EPS": "Extended phase shift: D1 < 1, D2 = 1",
    "TPS": "Triple phase shift: D1, D2 and D0 independent",
}
CIRCUIT = Path(__file__).resolve().parents[4] / "data" / "upstream" / "fund" / "dab" / "DAB_sample1.plecs"
DAB_MODEL_DIR = Path(__file__).resolve().parents[4] / "configs" / "dab"


def constrain(strategy: str, d0: float, d1: float, d2: float) -> tuple[float, float, float]:
    if strategy == "SPS":
        return d0, 1.0, 1.0
    if strategy == "DPS":
        return d0, d1, d1
    if strategy == "EPS":
        return d0, d1, 1.0
    return d0, d1, d2


def bridge_waveforms(d0: float, d1: float, d2: float, n: int = 400) -> dict:
    """One switching period, normalized time t/Ts in [0, 1): primary and secondary bridge voltages
    (normalized to ±1). Convention used here: each bridge is a three-level square wave whose
    non-zero part lasts D·Ts/2 per half-period, centred in it; the secondary is delayed by D0·Ts/2.
    This shows what the parameters mean; it is not a circuit simulation."""
    t = np.arange(n) / n

    def wave(d, shift):
        tt = (t - shift) % 1.0
        half = (tt % 0.5) / 0.5            # position inside the half-period, 0..1
        on = np.abs(half - 0.5) <= d / 2   # centred pulse of width d
        sign = np.where(tt < 0.5, 1.0, -1.0)
        return np.where(on, sign, 0.0)

    return {"t": t, "v_p": wave(d1, 0.0), "v_s": wave(d2, d0 / 2)}


def data_status() -> dict:
    trained = sorted(p.parent.name for p in DAB_MODEL_DIR.glob("*/*/metrics.json")) if DAB_MODEL_DIR.exists() else []
    return {"circuit_available": CIRCUIT.is_file(), "trained_runs": trained}
