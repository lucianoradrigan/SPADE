"""R2 (round PLECS/OM): pins the measured diagnosis sensitivity to the DC plant's discretization
(S-16 / S-18), as pre-registered in docs/audit/plecs_om/r2_preregistration.md: PC model, steady
state, 150 rad/s, severity x1, one seed per class, NAT (production Euler x1) vs REF (RK45 1e-9).
Measured on 2026-10-25: agreement 1.000, delta accuracy 0.000. Declared tolerance: +-0.02."""

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def test_pc_steady_state_sensitivity_is_as_measured():
    sys.path.insert(0, str(ROOT / "experiments"))
    import report_s16_discretization_sensitivity as r2

    cells = r2.sens_cells([150.0], [1.0], 1)
    data, _ = r2._simulate_sens(argparse.Namespace(cache=None, workers=4), cells)
    import json

    metrics = json.loads((ROOT / r2.MODELS["PC"][0] / "metrics.json").read_text())
    X, labels, _ = r2._windows_paired(data, cells, metrics, 2048, "steady_state")
    p = {v: r2._predict("PC", X[v])[0].argmax(1) for v in X}
    y = [metrics["classes"].index(l) for l in labels]
    agreement = float((p["NAT"] == p["REF"]).mean())
    delta = float((p["REF"] == y).mean() - (p["NAT"] == y).mean())
    assert abs(agreement - 1.000) <= 0.02, agreement
    assert abs(delta - 0.000) <= 0.02, delta
