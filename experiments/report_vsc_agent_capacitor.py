"""The VSC simulation-based anomaly agent with capacitor-degradation hypotheses (A4 follow-up, option
c of the decision list): does it separate a degraded filter capacitor from healthy telemetry, and
does it name the right cause instead of a load change?

Telemetry: 0.2 s of the VSC closed loop (default controller) at the agent's nominal operating point
(8.0064 ohm, 50 V, seed 0), for: healthy; capacitor at 90 / 80 / 70 % of nominal; load drift to
12 and 20 ohm. Complex Gaussian measurement noise of 1, 2 and 5 % of each channel's RMS, 10 noise
draws each. The agent (agents/vsc_agent.py) compares against its own hypotheses -- healthy, load
x1.5 / x2.5, capacitor 80 % / 70 % -- simulated without noise.

Reported per noise level: ROC AUC of the anomaly score, healthy vs. each condition; and the share
of windows whose closest hypothesis is a capacitor one (should be high for capacitor conditions and
low for load drift). SYNTHETIC-ONLY fault definition.

Usage:

    python experiments/report_vsc_agent_capacitor.py --out docs/audit/ai/vsc_agent_capacitor.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector
from driveflow.datagen import Scenario, run_scenario

ROOT = Path(__file__).resolve().parents[1]
DURATION_S = 0.2
CONDITIONS = {
    "healthy": {},
    "capacitor_90pct": {"vsc_capacitor_scale": 0.9},
    "capacitor_80pct": {"vsc_capacitor_scale": 0.8},
    "capacitor_70pct": {"vsc_capacitor_scale": 0.7},
    "load_12ohm": {"load_resistance_ohm": 12.0},
    "load_20ohm": {"load_resistance_ohm": 20.0},
}
NOISE_PCTS = (1.0, 2.0, 5.0)
DRAWS = 10


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "audit" / "ai" / "vsc_agent_capacitor.json")
    args = ap.parse_args(argv)
    agent = VSCDPCAnomalyDetector()
    clean = {}
    for name, kw in CONDITIONS.items():
        df = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id=f"telemetry_{name}", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=DURATION_S, seed=0, **kw)))
        clean[name] = {f: df[f].to_numpy(dtype=float) for f in agent.monitored_features}
    res = {"duration_s": DURATION_S, "hypotheses": list(agent.HYPOTHESES), "noise": {}}
    for pct in NOISE_PCTS:
        rng = np.random.default_rng(int(pct * 100))
        scores, closest = {}, {}
        for name, sig in clean.items():
            scores[name], closest[name] = [], []
            for _ in range(DRAWS):
                noisy = {f: v + rng.normal(0, pct / 100 * np.sqrt(np.mean(v**2)), len(v)) for f, v in sig.items()}
                s, c = agent.detect_anomaly(noisy, operating_point={})
                scores[name].append(s)
                closest[name].append(c)
        row = {}
        for name in CONDITIONS:
            entry = {"mean_score": float(np.mean(scores[name])), "closest_hypotheses": {h: closest[name].count(h) / DRAWS for h in set(closest[name])},
                     "named_capacitor_share": float(np.mean([c.startswith("capacitor") for c in closest[name]]))}
            if name != "healthy":
                y = np.r_[np.zeros(DRAWS), np.ones(DRAWS)]
                entry["auc_vs_healthy"] = float(roc_auc_score(y, np.r_[scores["healthy"], scores[name]]))
            row[name] = entry
        res["noise"][f"{pct:g}pct"] = row
        print(f"noise {pct:g} %:", {n: (round(r.get('auc_vs_healthy', float('nan')), 3), round(r['named_capacitor_share'], 2)) for n, r in row.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
