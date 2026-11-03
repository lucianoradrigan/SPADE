"""DECISIONES_S21_F1 §5.10: dc_pmg132_v1 has a native plant profile (batch only, no Gym environment) and
its scenario_sweep conditions carry the same per-condition speed metric as dc_perm_ex_v1. Nothing
visible depended on the absence (no GUI form launches this plant), so no finding is opened."""

import pandas as pd

from driveflow.jobs.handlers import _condition_metrics
from driveflow.runs.profiles import NATIVE_PROFILES, native_profile_for


def test_profile():
    p = native_profile_for("dc_pmg132_v1")
    assert p is not None and p.profile_id == "native:dc_pmg132_v1" and p.status == "published"
    assert p.capabilities.can_simulate_batch and not p.capabilities.can_step_interactively and not p.capabilities.can_train_rl
    assert set(NATIVE_PROFILES) == {"native:dc_perm_ex_v1", "native:dc_pmg132_v1", "native:vsc_dpc_v1", "native:pmsm_v1", "native:buck_v1"}


def test_condition_metric():
    df = pd.DataFrame({"timestamp_s": [0.1, 0.2, 0.3, 0.4], "rpm": [0.0, 10.0, 20.0, 30.0], "plant_config_id": ["dc_pmg132_v1"] * 4})
    assert _condition_metrics(df)["mean_rpm_second_half"] == 25.0
