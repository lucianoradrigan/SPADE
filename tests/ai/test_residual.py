"""ai/residual.py and the healthy-only three-way forecaster training (expansion task E14)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from driveflow.ai import residual
from driveflow.models.regressors.schemas import ForecasterConfig, ForecasterConfigError

ROOT = Path(__file__).resolve().parents[2]
CH = ["current_r", "rpm"]


class _MeanModel:
    """Predicts the context mean (zeros in normalized units) for every horizon step."""

    def __init__(self, horizon):
        self.horizon = horizon

    def __call__(self, x, training=False):
        return np.zeros((len(x), self.horizon, x.shape[2]), np.float32)

    def predict(self, x, batch_size=None, verbose=0):
        return self(x)


def _df(n_runs=4, n=3000, ripple=0.0, label="normal", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t = np.arange(1, n + 1) * 1e-4
    for r in range(n_runs):
        cur = 5 + 0.05 * rng.standard_normal(n) + ripple * np.sin(2 * np.pi * 120 * t)
        rpm = 1400 + 0.5 * rng.standard_normal(n) + 20 * ripple * np.sin(2 * np.pi * 120 * t)
        rows.append(pd.DataFrame({"timestamp_s": t, "current_r": cur, "rpm": rpm, "label": label, "source_file": f"{label}_{r}"}))
    return pd.concat(rows, ignore_index=True)


def test_fit_threshold_refuses_faulted_windows():
    rmse = np.ones((200, 2))
    with pytest.raises(residual.ResidualDetectorError, match="healthy"):
        residual.fit_threshold(rmse, ["normal"] * 199 + ["outer_race"], CH)
    with pytest.raises(residual.ResidualDetectorError, match="too few"):
        residual.fit_threshold(np.ones((50, 2)), ["normal"] * 50, CH)


def test_threshold_is_the_99th_percentile_of_healthy_validation_scores():
    rng = np.random.default_rng(1)
    rmse = np.abs(rng.normal(1.0, 0.2, size=(1000, 2)))
    p = residual.fit_threshold(rmse, ["normal"] * 1000, CH)
    s = residual.scores_from_rmse(rmse, p)
    assert p["quantile"] == 0.99 and np.isclose(p["threshold"], np.quantile(s, 0.99))
    assert np.isclose(np.median(rmse[:, 0] / p["channel_scale"][0]), 1.0)
    assert p["fitted_on"]["labels"] == ["normal"]


def test_healthy_window_below_and_large_ripple_above_the_threshold():
    window, horizon = 256, 16
    model = _MeanModel(horizon)
    metrics = {"channels": CH, "input_window": window}
    seg_h, _, lab_h, _ = residual.raw_windows(_df(n_runs=12, n=8000), CH, window, horizon)
    pred = residual.forecast(model, seg_h, CH, window)
    metrics["residual_detector"] = residual.fit_threshold(residual.channel_rmse(seg_h, pred, CH, CH, window), lab_h, CH)
    healthy = residual.score(model, seg_h, metrics)
    assert np.mean(residual.is_anomalous(healthy, metrics)) <= 0.02
    seg_f, _, _, _ = residual.raw_windows(_df(ripple=3.0, label="outer_race", seed=5), CH, window, horizon)
    assert residual.is_anomalous(residual.score(model, seg_f, metrics), metrics).all()


def test_raw_windows_respect_the_time_limits():
    df = _df(n_runs=1, n=8000)
    seg, groups, labels, t0 = residual.raw_windows(df, CH, 256, 16, cap_per_group=1000, start_s=0.5)
    assert len(seg) and (t0 >= 0.5).all() and seg.shape[1:] == (272, 2)
    seg2, _, _, t02 = residual.raw_windows(df, CH, 256, 16, cap_per_group=1000, end_s=0.5)
    assert len(seg2) and (t02 + 271e-4 <= 0.5 + 1e-9).all()


@pytest.mark.parametrize("kwargs, msg", [
    (dict(channels=("current_r",), target_channels=("rpm",), split="three_way"), "subset"),
    (dict(channels=("current_r", "rpm"), target_channels=("rpm",)), "three_way"),
    (dict(split="random"), "split"),
])
def test_schema_guards(kwargs, msg):
    with pytest.raises(ForecasterConfigError, match=msg):
        ForecasterConfig(domain="dc_motor", tier="pc", input_window=64, horizon=8, recurrent_type="none", layers=(4,), **kwargs)


@pytest.mark.slow
def test_healthy_only_three_way_training_fits_the_detector_on_validation(tmp_path):
    sys.path.insert(0, str(ROOT))
    from driveflow.datagen import Scenario, export_parquet, run_scenario
    from experiments.train_model import train_from_config

    scen = [Scenario(scenario_id=f"h{w:g}_{s}", omega_ref_rad_s=w, duration_s=1.0, seed=s, electrical_noise_pct=20.0) for w in (100.0, 125.0, 150.0, 175.0, 200.0) for s in (1, 2)]
    scen += [Scenario(scenario_id="f150", omega_ref_rad_s=150.0, fault_type="outer_race", duration_s=0.3, seed=9)]
    data = tmp_path / "d.parquet"
    export_parquet([r for s in scen for r in run_scenario(s)], data)
    cfg = tmp_path / "res.yaml"
    cfg.write_text(yaml.safe_dump({"domain": "dc_motor", "tier": "pc", "input_window": 128, "horizon": 8, "recurrent_type": "none", "layers": [8],
                                   "channels": ["current_r", "rpm"], "target_channels": ["current_r", "rpm"], "label_filter": "normal", "split": "three_way"}))
    run = train_from_config(cfg, data, epochs=1, batch_size=16, seed=0)
    m = json.loads((run / "metrics.json").read_text())
    tr, va, te = set(m["train_groups"]), set(m["val_groups"]), set(m["test_groups"])
    assert tr and va and te and not (tr & va or tr & te or va & te)
    assert "f150" not in tr | va | te  # the faulted run never enters training or the threshold
    det = m["residual_detector"]
    assert det["fitted_on"] == {"windows": "validation", "labels": ["normal"], "n": det["fitted_on"]["n"]} and det["threshold"] > 0
    assert set(m["persistence_test_rmse_per_channel"]) == {"current_r", "rpm"} and m["horizon"] == 8
