"""H0 reproduction of AI-01..AI-06 and AI-08 against the UNMODIFIED code (commit eb5d878).

Run from the repo root with the project venv:  .venv/bin/python <this file>
Each section prints REPRODUCED / NOT REPRODUCED with the concrete evidence.
"""

import inspect
import os
import re
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

REPO = Path.cwd()


def verdict(name, reproduced, evidence):
    print(f"[{name}] {'REPRODUCED' if reproduced else 'NOT REPRODUCED'} -- {evidence}")


# ---------------------------------------------------------------- AI-01 row-level shuffling merge
from driveflow.ai.transfer.data_merger import DataMerger

def acq(name, n, fs=10_000.0, offset=0.0):
    t = offset + np.arange(1, n + 1) / fs
    return pd.DataFrame({"timestamp_s": t, "current_r": np.sin(2 * np.pi * 50 * t), "label": "normal", "source_file": name})

sim = pd.concat([acq("run_a", 400), acq("run_b", 400)], ignore_index=True)
ext = pd.concat([acq("run_a", 400), acq("ext_c", 400)], ignore_index=True)  # same source_file name as sim on purpose
merged = DataMerger(simulated_data=sim, external_data=ext, mix_ratio=0.7).merge(seed=0)
non_monotonic = 0
for (src, origin), g in merged.groupby(["source_file", "_data_source"]):
    non_monotonic += int((np.diff(g["timestamp_s"].to_numpy()) <= 0).sum())
collided = merged.groupby("source_file")["_data_source"].nunique().max() > 1
verdict("AI-01", non_monotonic > 0 or collided,
        f"{non_monotonic} non-increasing timestamp steps inside merged groups (rows sampled individually); "
        f"source_file collision sim/ext in one group: {collided}")

# ---------------------------------------------------------------- AI-02 upload validator drops metadata
from driveflow.viz.dpc_upload_validation import validate_dc_motor_upload

up = acq("ext_c", 50).assign(label="outer_race")
clean, _, _ = validate_dc_motor_upload(up)
lost = [c for c in ("label", "source_file", "timestamp_s") if c not in clean.columns]
verdict("AI-02", bool(lost), f"columns lost after validate_dc_motor_upload: {lost}")

# ---------------------------------------------------------------- AI-03 val == test in TL regression
from driveflow.viz import transfer_learning_dashboard as tld

src = inspect.getsource(tld._prepare_windows)
uses_same = bool(re.search(r'"val": \{"X": X\[idx_test\].*"test": \{"X": X\[idx_test\]', src))
verdict("AI-03", uses_same, "_prepare_windows regression branch builds 'val' and 'test' from the same idx_test")

# ---------------------------------------------------------------- AI-04 base model mutated in place
import keras
from driveflow.ai.transfer.pipeline import FineTuneStrategy, TransferLearningPipeline
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import ClassifierConfig, ConvBlockConfig

keras.utils.set_random_seed(0)
cfg = ClassifierConfig(domain="dc_motor", tier="pc", input_window=16, num_classes=2, blocks=(ConvBlockConfig(filters=4, kernel_size=3),))
base = build_classifier(cfg, n_channels=2)
before = [w.copy() for w in base.get_weights()]
pipe = TransferLearningPipeline.from_loaded_model(base, strategy=FineTuneStrategy.ADAPTER, train_config={"learning_rate": 1e-2})
rng = np.random.default_rng(0)
data = {"X": rng.normal(size=(32, 16, 2)).astype("float32"), "y": rng.integers(0, 2, 32)}
pipe.train(data, data, epochs=2, batch_size=8)
changed = any(not np.array_equal(a, b) for a, b in zip(before, base.get_weights()))
verdict("AI-04", changed and pipe.model is base, f"pipeline.model is the cached base object: {pipe.model is base}; base weights changed: {changed}")

# ---------------------------------------------------------------- AI-05 NaN / constant regressor passes
from driveflow.ai.transfer.validator import ModelValidator

class NaNRegressor:
    def predict(self, X, verbose=0):
        return np.full((len(X), 4, 2), np.nan, dtype="float32")

class ConstRegressor:
    def predict(self, X, verbose=0):
        return np.zeros((len(X), 4, 2), dtype="float32")

reg_data = {"X": rng.normal(size=(10, 8, 2)).astype("float32"), "y": rng.normal(size=(10, 4, 2)).astype("float32")}
p_nan, r_nan = ModelValidator({"test_rmse": 0.06}, {"max_accuracy_drop": 1.0}).validate(NaNRegressor(), reg_data)
p_const, _ = ModelValidator({"test_rmse": 0.06}, {"max_accuracy_drop": 1.0}).validate(ConstRegressor(), reg_data)
calls_convergence = "check_convergence" in inspect.getsource(tld._render_fase_tl)
verdict("AI-05", p_nan and p_const,
        f"NaN regressor passed={p_nan} ({r_nan}); constant regressor passed={p_const}; TL calls check_convergence: {calls_convergence}")

# ---------------------------------------------------------------- AI-06 promotion with failed checks
src_tl = inspect.getsource(tld._render_fase_tl)
LOOSE = '"max_accuracy_drop": 1.0'
button_not_gated = 'st.button("Promote to registry", key="tl_promote_button")' in src_tl
from driveflow.ai.registry import promote, _load_index

tmp = Path(tempfile.mkdtemp())
reg = tmp / "registry.yaml"
run = tmp / "configs" / "classifiers" / "pc_server_tl" / "2026-01-01_run01"
run.mkdir(parents=True)
shutil.copyfile(REPO / "configs/classifiers/pc_server/2026-09-01_run01/config.yaml", run / "config.yaml")
base.save_weights(run / "model.weights.h5")
(run / "metrics.json").write_text('{"accuracy_ok": false, "mode_collapse_ok": false, "fine_tuned_from": "x", "channels": ["a","b"]}')
promote(run, registry_path=reg)
promoted = "dc_motor/pc/classifier" in _load_index(reg)
verdict("AI-06", button_not_gated and promoted,
        f"Promote button has no disabled= gate: {button_not_gated}; registry.promote() accepted a run whose metrics say checks FAILED: {promoted}; "
        f"threshold used in TL: max_accuracy_drop=1.0 present: {LOOSE in src_tl}")
shutil.rmtree(tmp)

# ---------------------------------------------------------------- AI-08 last event overwrites status
from driveflow.monitoring.agents.agent_gateway import Alert
from driveflow.monitoring.agents.agent_server import ServerAgent

sa = ServerAgent()
sa.record_alert("dc_motor", Alert("rule_A", "high", "alert", 1.0))
sa.record_alert("dc_motor", Alert("rule_B", "high", "alert", 2.0))
sa.record_alert("dc_motor", Alert("rule_A", "high", "alert", 3.0, resolved=True))
status_after = sa.domain_status("dc_motor")
never_seen = ServerAgent().domain_status("never_reported")
verdict("AI-08", status_after == "ok",
        f"A+B active, resolve A -> domain_status={status_after!r} (B is still active); never-seen domain -> {never_seen!r}")
