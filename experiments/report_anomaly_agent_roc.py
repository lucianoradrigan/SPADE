"""Labelled evaluation of the DC motor simulation-based anomaly agent (docs/AI_TECHNICAL_AUDIT.md
AI-N09/AI-N10, AUDIT_ANSWERS Q25).

Runs DCMotorAnomalyDetector on fresh simulated runs whose truth is known -- healthy, outer_race and
inner_race at several severities and speeds, seeds never used by the hypotheses (seed 0) -- and
reports, per configuration of the agent:
  - ROC AUC of the anomaly score (healthy = negative, any fault = positive);
  - false-alarm and missed-detection rates at the dashboard thresholds 0.70 (alert) and 0.85 (high);
  - how often the closest hypothesis is "healthy" for healthy runs;
  - wall time per detect_anomaly call (simulation cache warm).

The window is the dashboard's monitoring window (0.6 s from standstill). Speeds other than 150 rad/s
check that the operating point is passed into the hypotheses (AI-N09).

Usage:
    python experiments/report_anomaly_agent_roc.py [--out docs/audit/ai/anomaly_agent_roc.json]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEEDS = (100.0, 130.0, 150.0, 180.0)
CASES = (
    ("healthy", None, 8.0, None),
    ("outer_race_x0.4", "outer_race", 3.0, 0.02),
    ("outer_race_x1", "outer_race", 8.0, 0.05),
    ("inner_race_x0.5", "inner_race", 4.0, 0.01),
    ("inner_race_x1", "inner_race", 8.0, 0.02),
)
SEEDS = (31, 32, 33)
DURATION_S = 0.6
THRESHOLDS = (0.70, 0.85)


def _evaluate_speed(speed: float) -> list:
    import warnings

    warnings.simplefilter("ignore")
    import pandas as pd

    from driveflow.agents.dc_motor_agent import DCMotorAnomalyDetector
    from driveflow.datagen import Scenario, run_scenario

    agent = DCMotorAnomalyDetector()
    rows = []
    for name, fault, es, ms in CASES:
        for seed in SEEDS:
            df = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="roc", fault_type=fault, electrical_severity=es, mechanical_severity=ms, omega_ref_rad_s=speed, duration_s=DURATION_S, seed=seed)))
            tel = {f: df[f].to_numpy(dtype=float) for f in agent.monitored_features}
            agent.detect_anomaly(tel, operating_point={"omega_ref_rad_s": speed})  # warm the cache
            t0 = time.perf_counter()
            score, closest = agent.detect_anomaly(tel, operating_point={"omega_ref_rad_s": speed})
            rows.append({"speed": speed, "case": name, "faulty": fault is not None, "seed": seed, "score": score, "closest": closest, "seconds": time.perf_counter() - t0})
    return rows


def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney estimate of P(score_pos > score_neg), ties counted half."""
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "audit" / "ai" / "anomaly_agent_roc.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(SPEEDS)) as pool:
        rows = [r for part in pool.map(_evaluate_speed, SPEEDS) for r in part]
    s = np.array([r["score"] for r in rows])
    y = np.array([r["faulty"] for r in rows])
    summary = {"auc": _auc(s[~y], s[y]), "n_healthy": int((~y).sum()), "n_faulty": int(y.sum()),
               "healthy_closest_is_healthy": float(np.mean([r["closest"] == "healthy" for r in rows if not r["faulty"]])),
               "median_seconds_per_call_cache_warm": float(np.median([r["seconds"] for r in rows]))}
    for th in THRESHOLDS:
        summary[f"false_alarm_rate_at_{th}"] = float(np.mean(s[~y] >= th))
        summary[f"missed_detection_rate_at_{th}"] = float(np.mean(s[y] < th))
    per_case = {}
    for name, *_ in CASES:
        sc = [r["score"] for r in rows if r["case"] == name]
        per_case[name] = {"min": float(np.min(sc)), "median": float(np.median(sc)), "max": float(np.max(sc))}
    report = {"command": "python experiments/report_anomaly_agent_roc.py", "speeds": SPEEDS, "seeds": SEEDS, "duration_s": DURATION_S,
              "summary": summary, "score_by_case": per_case, "runs": rows, "wall_time_s": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": summary, "score_by_case": per_case}, indent=2))


if __name__ == "__main__":
    main()
