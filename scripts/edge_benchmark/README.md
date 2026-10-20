# Edge benchmark kit (Raspberry Pi 5, ESP32)

> **Not executed on hardware.** Nothing in this folder has run on a Raspberry Pi 5 or an ESP32. The RPi5 script was only run on the PC host with TensorFlow's interpreter as a fallback, to check that it works. The ESP32 sources were not compiled. Treat every number they produce on a PC as a host measurement, never as an edge result.

This kit is TODO_CLAUDE_CODE.md P3: it is what to run once hardware is available.

## 1. Reference vectors (on the PC)

```bash
python scripts/edge_benchmark/make_reference_vectors.py --n 64
```

This writes `vectors/<run>.npz` for:
- the promoted dc_motor RPi5 and ESP32 classifiers;
- the promoted vsc_dpc ESP32 forecaster;
- the RPi5 TCN forecaster candidate.

Each file holds the normalized input windows (from freshly simulated runs), the Keras outputs of the same run, the host TFLite outputs, and the `.tflite` SHA-256. The vectors are regenerated on demand and are not committed.

## 2. Raspberry Pi 5

Copy the run's `model.tflite`, its `.npz` and `rpi5_benchmark.py` to the board, then:

```bash
pip install ai-edge-litert            # or tflite-runtime; numpy is the only other dependency
python3 rpi5_benchmark.py --model model.tflite --vectors dc_motor_rpi5_classifier.npz --out result.json
```

The script reports:
- latency per window (p50 and p99);
- the peak resident memory;
- agreement with the PC: top-1 agreement for classifiers, RMSE for forecasters, against both the Keras and the host TFLite outputs.

`ran_on_arm_board_with_native_interpreter` is `true` only when LiteRT or tflite-runtime ran on an ARM board. A host run reports the backend as `HOST FALLBACK`.

The promoted RPi5 forecaster (`rpi5_edge`, a GRU) needs Flex ops and will **not** load on LiteRT or tflite-runtime. Use the TCN candidate (`rpi5_tcn/2026-10-20_run01`, AI_TECHNICAL_AUDIT.md §8.2) or promote it first.

## 3. ESP32 (TensorFlow Lite Micro, ESP-IDF)

```bash
python scripts/edge_benchmark/esp32/to_c_array.py \
    --model configs/classifiers/esp32_dscnn/2026-10-18_run01/model.tflite \
    --vectors scripts/edge_benchmark/vectors/dc_motor_esp32_classifier.npz --n 16
cd scripts/edge_benchmark/esp32 && idf.py add-dependency "espressif/esp-tflite-micro" && idf.py build flash monitor
```

**What `to_c_array.py` generates.** It writes `main/model_data.cc`, which holds the model as an aligned byte array, and `main/test_vectors.h`, which holds the int8-quantized inputs and the expected host outputs.

**What `main/main.cc` reports.** The arena size used, the mean and worst latency (`esp_timer`), the RMSE against the host TFLite outputs, and the top-1 agreement.

**Before the first build:**
- **Arena size.** `kArenaSize` is a guess; tune it to `arena_used_bytes()`.
- **Op list.** Check that the op resolver lists every op of the model. `docs/audit/tflite_parity_report.txt` lists the ops of each export.
