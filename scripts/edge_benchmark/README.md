# Edge benchmark kit (Raspberry Pi 5, ESP32, TI C2000)

> **Not executed on hardware (Raspberry Pi 5, ESP32).** Nothing for those boards has run on a Raspberry Pi 5 or an ESP32. The C2000 kit of section 4 is the exception: it did run on its board. The RPi5 script was only run on the PC host with TensorFlow's interpreter as a fallback, to check that it works. The ESP32 sources were not compiled. Treat every number they produce on a PC as a host measurement, never as an edge result.

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

## 4. TI C2000 (LAUNCHXL-F28379D)

**This one was executed on the board:** [`docs/audit/c2000/report.md`](../../docs/audit/c2000/report.md).

The C28x has no 8-bit type, so TFLite Micro does not apply. `experiments/export_c2000.py` writes `<run>/c2000/`, a self-contained project that builds without CCS:

```bash
python -m experiments.export_c2000 --domain dc_motor --tier esp32 --block classifier
make -C configs/classifiers/esp32_dscnn/2026-10-18_run01/c2000
python3 scripts/edge_benchmark/c2000_collect.py configs/classifiers/esp32_dscnn/2026-10-18_run01/c2000
```

The Makefile expects TI's CGT, UniFlash and C2000Ware under `~/ti` and the target file at `~/post_doc/C2000/TMS320F28379D.ccxml`; override `CGT`, `UNIFLASH`, `C2000WARE` or `CCXML` otherwise.

**To use the model in your own firmware:** copy `spade_model.c`, `spade_model.h`, `spade_nn.c`, `spade_nn.h`, add the linker regions of `lnk_spade_f28379d.cmd`, and call `spade_model_run(raw)` with a full window of physical samples in the channel order listed at the top of `spade_model.c`. Then read `spade_model_output()`, and for a classifier `spade_model_decide()`. The window must arrive at the rate the model was trained on: 5 kHz.

Re-exporting a run deletes its previous board evidence, because that evidence no longer describes the kit.
