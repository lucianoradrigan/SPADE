"""Edge benchmark for a Raspberry Pi 5 (or any Linux board) -- TODO_CLAUDE_CODE.md P3.
NOT EXECUTED ON HARDWARE: this script was only run on the PC host to check that it works.

Standalone: needs numpy and ONE TFLite interpreter, tried in this order:
ai_edge_litert (LiteRT), tflite_runtime, tensorflow (host fallback, reported as such). It does
not import driveflow.

Measures, for one .tflite and its reference vectors (make_reference_vectors.py):
- latency per window: median (p50) and p99 over --repeats passes, after --warmup windows;
- resident memory: peak RSS of the process (resource.getrusage), before and after loading;
- agreement with the PC: top-1 agreement with the Keras model (classifiers) or RMSE against it
  (regressors), plus the same against the host TFLite run.

Usage on the device:

    python3 rpi5_benchmark.py --model model.tflite --vectors vectors/dc_motor_rpi5_classifier.npz --out result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import numpy as np


def _interpreter(model_path: str):
    for module, name in (("ai_edge_litert.interpreter", "ai_edge_litert"), ("tflite_runtime.interpreter", "tflite_runtime")):
        try:
            mod = __import__(module, fromlist=["Interpreter"])
            return mod.Interpreter(model_path=model_path), name
        except ImportError:
            continue
    import tensorflow as tf  # host fallback only

    return tf.lite.Interpreter(model_path=model_path), f"tensorflow {tf.__version__} (HOST FALLBACK)"


def _peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0 if platform.system() == "Linux" else kb / 2**20


def run(model_path: str, vectors_path: str, warmup: int = 10, repeats: int = 3) -> dict:
    data = np.load(vectors_path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    x = data["inputs"]
    rss_before = _peak_rss_mb()
    interp, backend = _interpreter(model_path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def infer(w):
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
        return y.reshape(-1)

    for w in x[:warmup]:
        infer(w)
    times, preds = [], None
    for _ in range(repeats):
        cur = []
        for w in x:
            t0 = time.perf_counter()
            cur.append(infer(w))
            times.append((time.perf_counter() - t0) * 1e3)
        preds = np.array(cur)
    keras_out, host_out = data["keras_out"], data["host_tflite_out"]
    result = {
        "model": model_path, "vectors": vectors_path, "run": meta["run"], "backend": backend,
        "platform": {"machine": platform.machine(), "system": platform.system(), "release": platform.release(), "python": platform.python_version()},
        "n_windows": int(len(x)), "repeats": repeats,
        "latency_ms_p50": float(np.percentile(times, 50)), "latency_ms_p99": float(np.percentile(times, 99)),
        "peak_rss_mb_before_load": rss_before, "peak_rss_mb_after_run": _peak_rss_mb(),
    }
    if meta["block"] == "classifier":
        result["top1_agreement_vs_keras"] = float(np.mean(preds.argmax(1) == keras_out.argmax(1)))
        result["top1_agreement_vs_host_tflite"] = float(np.mean(preds.argmax(1) == host_out.argmax(1)))
    else:
        result["rmse_vs_keras"] = float(np.sqrt(np.mean((preds - keras_out) ** 2)))
        result["rmse_vs_host_tflite"] = float(np.sqrt(np.mean((preds - host_out) ** 2)))
    # True only when a native edge interpreter ran on an ARM board; a host run is never reported as one.
    result["ran_on_arm_board_with_native_interpreter"] = "HOST FALLBACK" not in backend and platform.machine() in ("aarch64", "armv7l")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run(args.model, args.vectors, args.warmup, args.repeats)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
