"""AUDIT AI-10 / Q22: numerical parity of every shipped model.tflite against the Keras weights it
was exported from, on the SAME windows (the run's recorded grouped test split, rebuilt with the
training seed, plus all windows of the dataset). Nothing is re-exported or re-trained.

What this measures: what quantization (FP16 for rpi5, full INT8 for esp32) costs in accuracy /
RMSE and how often the TFLite and Keras predictions disagree. Latency is measured on THIS HOST's
CPU with tf.lite.Interpreter -- it is NOT an ESP32 / Raspberry Pi 5 measurement and does not move
any run past `exported` in ai/deployment_status.py's lifecycle.

Usage:  python experiments/report_tflite_parity.py [--json]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from driveflow.ai.deployment_status import tflite_op_names
from driveflow.ai.registry import load_promoted_model, resolve
from driveflow.models.common import build_classification_windows, build_direct_forecast_windows, grouped_split, load_diagnosis_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"dc_motor": REPO_ROOT / "data" / "diagnosis_dataset.parquet", "vsc_dpc": REPO_ROOT / "data" / "vsc_dpc_dataset.parquet"}
EDGE_RUNS = (("dc_motor", "esp32", "classifier"), ("dc_motor", "rpi5", "classifier"), ("vsc_dpc", "esp32", "regressor"), ("vsc_dpc", "rpi5", "regressor"))
TRAIN_SEED = 42  # experiments/train_model.py default, used for every shipped run


def _load_df(domain: str):
    if domain == "dc_motor":
        return load_diagnosis_dataset(DATASETS[domain])
    import pandas as pd

    df = pd.read_parquet(DATASETS[domain])
    return df[df["plant_config_id"] == "vsc_dpc_v1"].reset_index(drop=True)


def _tflite_predict(path: Path, X: np.ndarray) -> tuple:
    """Runs the interpreter one window at a time (the shipped models have batch 1), quantizing /
    dequantizing with the file's own scale and zero point. Returns (predictions, median ms)."""
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    preds, times = [], []
    for x in X:
        x = x[None].astype(np.float32)
        if inp["dtype"] == np.int8:
            x = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1e3)
        y = interp.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            y = (y.astype(np.float32) - out_zp) * out_scale
        preds.append(y)
    return np.array(preds), float(np.median(times))


def _op_names(path: Path) -> list:
    try:
        return tflite_op_names(path)
    except Exception as exc:
        return [f"<unreadable: {exc}>"]


def _classifier_parity(model, tfl: Path, metrics: dict, df) -> dict:
    classes = metrics["classes"]
    X, y_str, groups = build_classification_windows(df, classes, metrics["channels"], model.input_shape[1])
    y = np.array([classes.index(c) for c in y_str])
    idx = np.arange(len(X))
    idx_tv, idx_test = grouped_split(idx, y, groups, 0.10, TRAIN_SEED)
    p_keras = model.predict(X, verbose=0)
    p_tfl, ms = _tflite_predict(tfl, X)
    k, t = p_keras.argmax(1), p_tfl.argmax(1)
    out = {"median_host_latency_ms": ms, "n_windows_all": int(len(X)), "n_windows_test": int(len(idx_test))}
    for name, ix in (("test", idx_test), ("all", idx)):
        out[name] = {
            "keras_accuracy": float((k[ix] == y[ix]).mean()),
            "tflite_accuracy": float((t[ix] == y[ix]).mean()),
            "keras_macro_f1": float(f1_score(y[ix], k[ix], average="macro", zero_division=0)),
            "tflite_macro_f1": float(f1_score(y[ix], t[ix], average="macro", zero_division=0)),
            "top1_agreement": float((k[ix] == t[ix]).mean()),
            "max_abs_prob_diff": float(np.abs(p_keras[ix] - p_tfl[ix]).max()),
        }
    out["recorded_test_accuracy"] = metrics.get("test_accuracy")
    return out


def _regressor_parity(model, tfl: Path, metrics: dict, df, config) -> dict:
    X, Y, groups = build_direct_forecast_windows(df, metrics["channels"], config.input_window, config.horizon)
    idx = np.arange(len(X))
    _, idx_test = grouped_split(idx, np.zeros(len(X), dtype=int), groups, test_frac=0.2, seed=TRAIN_SEED)
    p_keras = model.predict(X, verbose=0)
    p_tfl, ms = _tflite_predict(tfl, X)
    p_tfl = p_tfl.reshape(p_keras.shape)
    out = {"median_host_latency_ms": ms, "n_windows_all": int(len(X)), "n_windows_test": int(len(idx_test))}
    for name, ix in (("test", idx_test), ("all", idx)):
        out[name] = {
            "keras_rmse": float(np.sqrt(np.mean((p_keras[ix] - Y[ix]) ** 2))),
            "tflite_rmse": float(np.sqrt(np.mean((p_tfl[ix] - Y[ix]) ** 2))),
            "rmse_tflite_vs_keras": float(np.sqrt(np.mean((p_tfl[ix] - p_keras[ix]) ** 2))),
            "max_abs_diff_vs_keras": float(np.abs(p_tfl[ix] - p_keras[ix]).max()),
        }
    out["recorded_test_rmse"] = metrics.get("test_rmse")
    return out


def report() -> dict:
    rows = {}
    for domain, tier, block in EDGE_RUNS:
        key = f"{domain}/{tier}/{block}"
        run_dir = resolve(domain, tier, block)
        tfl = run_dir / "model.tflite"
        if not tfl.exists():
            rows[key] = {"status": "not_exported"}
            continue
        model, config, metrics = load_promoted_model(domain, tier, block)
        df = _load_df(domain)
        base = {"run": str(run_dir.relative_to(REPO_ROOT)), "tflite_bytes": tfl.stat().st_size, "ops": _op_names(tfl)}
        try:
            r = _classifier_parity(model, tfl, metrics, df) if block == "classifier" else _regressor_parity(model, tfl, metrics, df, config)
        except RuntimeError as exc:  # e.g. Flex (SELECT_TF_OPS) kernels: not runnable on TFLite Micro either
            rows[key] = {**base, "status": "not_runnable_builtin_interpreter", "error": str(exc).split("See instructions")[0].strip()}
            continue
        rows[key] = {**base, "status": "evaluated", **r}
    return {
        "measured_on": "host CPU, tf.lite.Interpreter " + tf.__version__ + " -- NOT target hardware",
        "runs": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = report()
    if args.json:
        print(json.dumps(r, indent=2))
        return
    print(f"Measured on: {r['measured_on']}")
    for key, row in r["runs"].items():
        if row.get("status") != "evaluated":
            print(f"\n{key}  ({row.get('run')}): {row.get('status')} -- {row.get('error', '')}\n  ops: {row.get('ops')}")
            continue
        print(f"\n{key}  ({row.get('run')}, {row.get('tflite_bytes', 0) / 1024:.0f} KiB, host median {row.get('median_host_latency_ms', float('nan')):.2f} ms/window)")
        flex = [o for o in row["ops"] if o.startswith("Flex")]
        print(f"  ops: {len(row['ops'])} distinct{'; FLEX ops (need TF Select delegate): ' + str(flex) if flex else ', all builtin'}")
        for side in ("test", "all"):
            s = row[side]
            n = row[f"n_windows_{side}"]
            if "keras_accuracy" in s:
                print(f"  {side:4s} ({n:3d} windows): accuracy keras {s['keras_accuracy']:.3f} -> tflite {s['tflite_accuracy']:.3f}   macro-F1 {s['keras_macro_f1']:.3f} -> {s['tflite_macro_f1']:.3f}   top-1 agreement {s['top1_agreement']:.3f}   max |dp| {s['max_abs_prob_diff']:.3f}")
            else:
                print(f"  {side:4s} ({n:3d} windows): RMSE keras {s['keras_rmse']:.4f} -> tflite {s['tflite_rmse']:.4f}   RMSE(tflite, keras) {s['rmse_tflite_vs_keras']:.4f}   max |diff| {s['max_abs_diff_vs_keras']:.4f}")


if __name__ == "__main__":
    main()
