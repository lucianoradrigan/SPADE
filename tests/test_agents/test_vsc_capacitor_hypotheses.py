"""VSC capacitor degradation as a Scenario option and as anomaly-agent hypotheses (A4 follow-up,
option c). Measured 2026-10-20 (docs/audit/ai/vsc_agent_capacitor.json): AUC 1.0 against healthy
for C = 90/80/70 % at 1-5 % noise; C = 80/70 % named as a capacitor hypothesis in 100 % of
windows; load drift never named as one."""

import numpy as np
import pandas as pd
import pytest

from driveflow.agents.vsc_agent import VSCDPCAnomalyDetector
from driveflow.datagen import Scenario, run_scenario


def _telemetry(agent, noise_pct=2.0, seed=0, **kw):
    df = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="t", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.2, seed=0, **kw)))
    rng = np.random.default_rng(seed)
    return {f: df[f].to_numpy(float) + rng.normal(0, noise_pct / 100 * np.sqrt(np.mean(df[f].to_numpy(float) ** 2)), len(df)) for f in agent.monitored_features}


def test_scenario_degrades_the_capacitor_and_labels_the_records():
    nominal = run_scenario(Scenario(scenario_id="a", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.05))
    degraded = run_scenario(Scenario(scenario_id="a", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.05, vsc_capacitor_scale=0.8))
    assert nominal[-1]["label"] == "normal" and degraded[-1]["label"] == "capacitor_degraded"
    assert abs(nominal[-1]["i_f_imag"] - degraded[-1]["i_f_imag"]) > 1e-3


@pytest.fixture(scope="module")
def agent():
    return VSCDPCAnomalyDetector()


def test_degraded_capacitor_is_named(agent):
    score, closest = agent.detect_anomaly(_telemetry(agent, vsc_capacitor_scale=0.8), operating_point={})
    healthy_score, _ = agent.detect_anomaly(_telemetry(agent, seed=1), operating_point={})
    assert closest.startswith("capacitor") and score > healthy_score


def test_load_drift_is_not_called_a_capacitor_fault(agent):
    _score, closest = agent.detect_anomaly(_telemetry(agent, load_resistance_ohm=20.0), operating_point={})
    assert not closest.startswith("capacitor")
