"""DAB performance model and TPS suggestion (task E9 preparation): code tested on SYNTHETIC data only.
No model is declared trained on real PLECS data (none exists)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from driveflow.ai import dab_modulation as dm
from driveflow.models.dab.schemas import DabConfigError, DabPerformanceConfig

ROOT = Path(__file__).resolve().parents[2]


def _synthetic(seed=0) -> pd.DataFrame:
    """A toy stress surface with a known minimum at D1 = 0.7, D2 = 0.8 (not a DAB model)."""
    rng = np.random.default_rng(seed)
    rows = []
    for vref in (120.0, 160.0, 200.0, 240.0):
        for p in (200.0, 400.0, 600.0, 800.0):
            for d1 in np.linspace(0.3, 1.0, 8):
                for d2 in np.linspace(0.3, 1.0, 8):
                    i = p / vref * (1 + 2 * (d1 - 0.7) ** 2 + 2 * (d2 - 0.8) ** 2)
                    rows.append({"Vin": 200.0, "Vref": vref, "P": p, "D1": d1, "D2": d2, "irms": i + rng.normal(0, 0.01), "ipk2pk": 3 * i, "Validity": 1})
    rows.append({"Vin": 200.0, "Vref": 160.0, "P": 400.0, "D1": 0.5, "D2": 0.5, "irms": 99.0, "ipk2pk": 99.0, "Validity": 0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("kw, msg", [(dict(domain="dc_motor"), "domain 'dab'"), (dict(group_by=("Vin",)), "subset of inputs"),
                                     (dict(outputs=("D1",)), "both input and output")])
def test_config_guards(kw, msg):
    base = dict(domain="dab", tier="pc", inputs=("Vref", "P", "D1", "D2"), outputs=("irms",), group_by=("Vref", "P"))
    with pytest.raises(DabConfigError, match=msg):
        DabPerformanceConfig(**{**base, **kw})


def test_shipped_config_loads():
    from driveflow.models.dab.schemas import load_dab_config

    c = load_dab_config(ROOT / "configs" / "dab" / "dab_tps_pc.yaml")
    assert c.domain == "dab" and c.inputs == ("Vref", "P", "D1", "D2") and c.valid_column == "Validity"


@pytest.mark.slow
def test_train_model_and_suggestion_on_synthetic_data(tmp_path):
    sys.path.insert(0, str(ROOT))
    from experiments.train_model import train_from_config

    data = tmp_path / "dab.csv"
    _synthetic().to_csv(data, index=False)
    cfg = tmp_path / "dab_test.yaml"
    cfg.write_text(yaml.safe_dump({"task": "dab_performance", "domain": "dab", "tier": "pc", "inputs": ["Vref", "P", "D1", "D2"], "outputs": ["ipk2pk", "irms"],
                                   "group_by": ["Vref", "P"], "valid_column": "Validity", "hidden": [32, 32]}))
    run = train_from_config(cfg, data, epochs=60, batch_size=32, seed=0)
    m = json.loads((run / "metrics.json").read_text())
    assert m["domain"] == "dab" and m["n_train"] + m["n_val"] + m["n_test"] == 16 * 64  # the invalid row is dropped
    test_pts = set(m["test_groups"])
    assert len(test_pts) >= 1 and all("|" in g for g in test_pts)
    model, _, metrics = dm.load_run(run)
    s = dm.suggest(model, metrics, {"Vref": 160.0, "P": 400.0}, training_range={"Vref": [120, 240], "P": [200, 800]})
    assert s["label"] == dm.SUGGESTION_LABEL
    assert abs(s["suggested"]["D1"] - 0.7) <= 0.15 and abs(s["suggested"]["D2"] - 0.8) <= 0.15
    assert s["suggested"]["irms"] < s["baseline_sps"]["irms"]
    with pytest.raises(ValueError, match="outside the model's training range"):
        dm.suggest(model, metrics, {"Vref": 400.0, "P": 400.0}, training_range={"Vref": [120, 240], "P": [200, 800]})
