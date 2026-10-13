"""Edge-deployment lifecycle of a run directory, from evidence only (AUDIT AI-10, Q21/Q22).

A `model.tflite` next to the weights proves that an export happened -- nothing more. It does not
prove the model runs on an ESP32 / Raspberry Pi 5, nor its latency, RAM/Flash or post-quantization
accuracy. The lifecycle is therefore explicit and each stage needs its own evidence:

    exported -> static_compatibility_passed -> bench_verified -> hardware_verified -> production_approved

- `exported` is read from the filesystem (model.tflite exists).
- Quantization is read from the .tflite itself (input/output tensor dtypes and whether any
  float16 constant tensors exist) -- not assumed from the tier name.
- Every later stage is "not_verified" unless an optional `deployment_evidence.json` in the run
  directory records it (board, runtime version, latency/RAM/flash, accuracy-after-quantization,
  who/when). No such file exists in this repository today, so no run is shown as hardware-verified.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

STAGES = ("exported", "static_compatibility_passed", "bench_verified", "hardware_verified", "production_approved")
EVIDENCE_FILE = "deployment_evidence.json"


@lru_cache(maxsize=64)
def _inspect_tflite(path: str, mtime: float) -> dict:
    try:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=path)
        ins = [str(d["dtype"].__name__) for d in interp.get_input_details()]
        outs = [str(d["dtype"].__name__) for d in interp.get_output_details()]
        tensor_types = {str(t["dtype"].__name__) for t in interp.get_tensor_details()}
    except Exception as exc:  # unreadable / incompatible file is itself a finding
        return {"readable": False, "error": str(exc)}
    if "int8" in ins and "int8" in outs:
        quant = "INT8 (full integer, int8 I/O)"
    elif "float16" in tensor_types:
        quant = "FP16 weights (float32 I/O)"
    else:
        quant = "FP32 or dynamic-range (not full-integer)"
    return {"readable": True, "input_dtypes": ins, "output_dtypes": outs, "quantization": quant}


def deployment_status(run_dir) -> dict:
    run_dir = Path(run_dir)
    tflite = run_dir / "model.tflite"
    status = {stage: "not_verified" for stage in STAGES}
    info: dict = {"artifact": None}
    if tflite.exists():
        status["exported"] = "done"
        info["artifact"] = {"path": str(tflite), "size_bytes": tflite.stat().st_size, **_inspect_tflite(str(tflite), tflite.stat().st_mtime)}
    else:
        status["exported"] = "not_exported"
    evidence_path = run_dir / EVIDENCE_FILE
    if evidence_path.exists():
        try:
            evidence = json.loads(evidence_path.read_text())
        except json.JSONDecodeError:
            evidence = {}
            info["evidence_error"] = "deployment_evidence.json unreadable"
        for stage in STAGES[1:]:
            if evidence.get(stage):
                status[stage] = "done"
        info["evidence"] = evidence
    return {"stages": status, **info}
