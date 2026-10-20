"""Builds the reference vectors the edge benchmarks compare against -- TODO_CLAUDE_CODE.md P3.
Runs on the PC (needs driveflow + TensorFlow). NOTHING here is measured on edge hardware.

For each edge run it writes scripts/edge_benchmark/vectors/<name>.npz with:
- inputs: N normalized windows, float32 (the model's preprocessing already applied, so the
  device only runs the interpreter);
- keras_out: the same run's Keras model on those windows (the PC-side reference);
- host_tflite_out: the .tflite on the host interpreter (what the device should reproduce);
- meta: run directory, block, classes or horizon, and the .tflite's SHA-256.

The windows come from freshly simulated runs, not from a training dataset: dc_motor at the
nominal setpoint (healthy, outer race, inner race), vsc_dpc with the default DPC controller.

Usage:

    python scripts/edge_benchmark/make_reference_vectors.py [--n 64]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "vectors"
RUNS = {
    "dc_motor_rpi5_classifier": ("configs/classifiers/rpi5_resnet1d_se/2026-10-18_run01", "classifier"),
    "dc_motor_esp32_classifier": ("configs/classifiers/esp32_dscnn/2026-10-18_run01", "classifier"),
    "vsc_dpc_esp32_regressor": ("configs/regressors/esp32_tiny/2026-10-19_run01", "regressor"),
    "vsc_dpc_rpi5_tcn_candidate": ("configs/regressors/rpi5_tcn/2026-10-20_run01", "regressor"),
}


def _simulated_frames(domain: str):
    import pandas as pd

    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    if domain == "dc_motor":
        frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"edge_{f or 'normal'}", fault_type=f, seed=7, duration_s=2.0))) for f in (None, "outer_race", "inner_race")]
        frames = [d[d["timestamp_s"] >= 0.5] for d in frames]
    else:
        frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"edge_vsc_{m:g}", controller_type="DPC", plant_config_id="vsc_dpc_v1", reference_magnitude_v=m, duration_s=0.3, seed=7))) for m in (35.0, 50.0, 75.0)]
    return pd.concat(frames, ignore_index=True)


def _host_tflite(path: Path, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    res = []
    for w in x:
        w = w[None].astype(np.float32)
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            w = np.clip(np.round(w / s + z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], w)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            s, z = out["quantization"]
            y = (y.astype(np.float32) - z) * s
        res.append(y.reshape(-1))
    return np.array(res)


def main(argv=None):
    from driveflow.ai.registry import load_run_model
    from driveflow.models.common import build_classification_windows, build_direct_forecast_windows

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    frames = {}
    for name, (run, block) in RUNS.items():
        run_dir = ROOT / run
        tfl = run_dir / "model.tflite"
        if not tfl.exists():
            print(f"skip {name}: no model.tflite")
            continue
        model, config, metrics = load_run_model(run_dir, block)
        domain = "dc_motor" if block == "classifier" else "vsc_dpc"
        df = frames.setdefault(domain, _simulated_frames(domain))
        if block == "classifier":
            x, _y, _g = build_classification_windows(df, metrics["classes"], metrics["channels"], config.input_window, cap_per_class=10_000, level_scaling=metrics.get("level_scaling"))
        else:
            x, _y, _g = build_direct_forecast_windows(df, metrics["channels"], config.input_window, config.horizon)
        x = x[np.random.default_rng(0).choice(len(x), size=min(args.n, len(x)), replace=False)].astype(np.float32)
        keras_out = model.predict(x, verbose=0).reshape(len(x), -1)
        meta = {"run": run, "block": block, "classes": metrics.get("classes"), "horizon": getattr(config, "horizon", None),
                "tflite_sha256": hashlib.sha256(tfl.read_bytes()).hexdigest(), "measured_on": "PC host -- NOT edge hardware"}
        np.savez_compressed(OUT / f"{name}.npz", inputs=x, keras_out=keras_out, host_tflite_out=_host_tflite(tfl, x), meta=json.dumps(meta))
        print(f"{name}: {len(x)} windows -> {OUT / (name + '.npz')}")


if __name__ == "__main__":
    main()
