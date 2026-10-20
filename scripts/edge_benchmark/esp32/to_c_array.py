"""Converts a .tflite model and its reference vectors into C sources for the ESP32 skeleton --
TODO_CLAUDE_CODE.md P3. NOT EXECUTED ON HARDWARE.

Writes esp32/main/model_data.{h,cc} (the flatbuffer as a 16-byte-aligned byte array) and
esp32/main/test_vectors.h (int8-quantized inputs with the model's own input scale/zero point, plus
the host TFLite outputs the device must reproduce). Needs numpy and TensorFlow (only to read the
model's quantization parameters).

Usage:

    python esp32/to_c_array.py --model ../../configs/classifiers/esp32_dscnn/2026-10-18_run01/model.tflite \
        --vectors vectors/dc_motor_esp32_classifier.npz --n 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parent / "main"


def _array(name: str, data: bytes, ctype: str = "unsigned char") -> str:
    body = ",".join(f"0x{b:02x}" for b in data)
    return f"alignas(16) const {ctype} {name}[] = {{{body}}};\nconst unsigned int {name}_len = {len(data)};\n"


def main(argv=None):
    import tensorflow as tf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--vectors", required=True, type=Path)
    ap.add_argument("--n", type=int, default=16, help="number of test windows to embed (flash is small)")
    args = ap.parse_args(argv)

    model = args.model.read_bytes()
    interp = tf.lite.Interpreter(model_content=model)
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    if inp["dtype"] != np.int8:
        raise SystemExit("the ESP32 skeleton expects a fully int8 model (experiments/export_tflite.py --tier esp32)")
    in_s, in_z = inp["quantization"]
    out_s, out_z = out["quantization"]
    d = np.load(args.vectors)
    x = d["inputs"][: args.n]
    q = np.clip(np.round(x / in_s + in_z), -128, 127).astype(np.int8)
    expected = d["host_tflite_out"][: args.n].astype(np.float32)
    meta = json.loads(str(d["meta"]))

    MAIN.mkdir(exist_ok=True)
    (MAIN / "model_data.h").write_text("#pragma once\nextern const unsigned char g_model[];\nextern const unsigned int g_model_len;\n")
    (MAIN / "model_data.cc").write_text(f'#include "model_data.h"\n// {meta["run"]}, sha256 {meta["tflite_sha256"]}\n' + _array("g_model", model))
    lines = ["#pragma once", f"// {len(q)} windows from {args.vectors.name}; host TFLite outputs as the expected values.",
             f"constexpr int kNumVectors = {len(q)};", f"constexpr int kInputSize = {q[0].size};", f"constexpr int kOutputSize = {expected.shape[1]};",
             f"constexpr float kOutputScale = {out_s!r}f;", f"constexpr int kOutputZeroPoint = {int(out_z)};",
             "const signed char kInputs[kNumVectors][kInputSize] = {" + ",".join("{" + ",".join(str(int(v)) for v in w.reshape(-1)) + "}" for w in q) + "};",
             "const float kExpected[kNumVectors][kOutputSize] = {" + ",".join("{" + ",".join(f"{v:.6g}f" for v in e) + "}" for e in expected) + "};", ""]
    (MAIN / "test_vectors.h").write_text("\n".join(lines))
    print(f"wrote {MAIN / 'model_data.cc'} ({len(model)} bytes) and {MAIN / 'test_vectors.h'} ({len(q)} vectors)")


if __name__ == "__main__":
    main()
