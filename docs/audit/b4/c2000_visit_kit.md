# C2000 visit kit (B4, D-B4-4 option B)

Prepared 2026-11-10 by Claude Code for Luciano. **Nothing in this file has been measured on the board.** The result fields are empty, for you to fill in. Every model input is simulated; the board replays simulated windows from flash (no ADC, no real signal), as in October.

- **Board:** LAUNCHXL-F28379D, CPU1 at 200 MHz. Toolchain as in October (`~/ti/ti-cgt-c2000_25.11.1.LTS`, `~/ti/uniflash`, C2000Ware, `~/post_doc/C2000/TMS320F28379D.ccxml`).
- **Criteria:** C1–C5 in [`../c2000/criteria.md`](../c2000/criteria.md), unchanged. **C5 now uses the data's rate, 10 kHz (AI-N26):** the budget is `samples / fs_hz` from the kit's manifest, so 25.6 ms for 256 samples and 51.2 ms for 512.
- **Why new kits:** the October kits have no `fs_hz` in their manifest, so `c2000_collect.py` refuses them. They are **not** re-exported in place: their `hardware_run.json` stays as measured. The new kits go to `docs/audit/b4/c2000_kit/<name>/`.

## Kits

| Kit | Source run | Samples | C5 budget at 10 kHz | Built here | Status |
|---|---|---|---|---|---|
| `c2000_kit/esp32_tiny` | `esp32_tiny/2026-10-19_run01` (vsc_dpc regressor) | 256 | 25.6 ms | host parity 4.8e-7 (gcc, this PC); `make` with cl2000 exit 0 | **ready** |
| `c2000_kit/esp32_b4` | the Keras run of the selected B4 ESP32 variant (`b4_results.md`) | 512 | 51.2 ms | — | **pending B4** (see below) |

`model_id` of the new `esp32_tiny` kit is `08626a228bca`, the same as the October kit: same weights.

**What to expect (not a result):** in October `esp32_tiny` took 19.7 ms (budget 25.6 ms) and `esp32_dscnn` 68.5 ms (budget 51.2 ms). The B4 ESP32 keeps the `esp32_dscnn` architecture and its 512-sample input, so its latency should be close to 68.5 ms, which **would fail C5** at 10 kHz. The visit measures it; it does not assume it.

**To make the B4 kit** (Claude Code, after B4 selects a variant; about 2 min):

```bash
.venv/bin/python scripts/edge_benchmark/c2000_visit_kit.py --domain dc_motor --block classifier \
    --run-dir <the Keras run of the selected variant> --out docs/audit/b4/c2000_kit/esp32_b4
```

The C2000 kit uses SPADE's float32 C kernels from the Keras weights, not the int8 TFLite file, as in October.

## Steps for Luciano (about 1 h 15 min)

| # | What | Command or action | Time |
|---|---|---|---|
| 1 | Connect the LaunchPad by USB (XDS100v2). Check the serial port | `ls /dev/ttyUSB*` (the UART is usually the second one, `ttyUSB1`; use the one that worked in October) | 10 min |
| 2 | Preflight both kits: `fs_hz` 10000 and the budget | `.venv/bin/python scripts/edge_benchmark/c2000_visit_kit.py --check-only --out docs/audit/b4/c2000_kit/esp32_tiny` (same for `esp32_b4`) | 5 min |
| 3 | Build and measure `esp32_tiny` | `.venv/bin/python scripts/edge_benchmark/c2000_collect.py docs/audit/b4/c2000_kit/esp32_tiny --port <port>` (flashes, then reads one complete report; timeout 120 s) | 10 min |
| 4 | Build and measure `esp32_b4` | same, with `docs/audit/b4/c2000_kit/esp32_b4` | 10 min |
| 5 | Repeat 3 and 4 once with `--no-flash` after a reset of the board (the S1 button), to check that the latency repeats. Copy each `hardware_run.json` aside before the repeat, because the collector overwrites it | `cp <kit>/hardware_run.json <kit>/hardware_run_1.json`, then the collector with `--no-flash` | 20 min |
| 6 | Fill in the tables below and commit the kit directories (`hardware_run*.json`, `deployment_evidence.json`) | `git add docs/audit/b4/c2000_kit docs/audit/b4/c2000_visit_kit.md` | 10 min |
| — | Buffer for flashing or port problems | | 10 min |

If the collector says `the board runs model X, the kit is Y`, the flash did not take: repeat step 3 or 4 without `--no-flash`.

## Results (to be filled in by Luciano; empty = not measured)

| Kit | Date | Port | C1 build | C2 memory | C3 parity (max abs err) | C4 top-1 | Worst latency (ms) | C5 budget (ms) | C5 | Repeat: worst latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| `esp32_tiny` | | | | | | n/a (regressor) | | 25.6 | | |
| `esp32_b4` | | | | | | | | 51.2 | | |

Notes from the bench:

-

## What this closes

- AI-N26 for the two models that met C5 in October (option B): `esp32_tiny` and the ESP32 classifier that comes out of B4, measured at 10 kHz.
- Not included: the RPi5 classifier (its arena of 448 KB does not fit in 128 KB, AI-N25) and `rpi5_tcn` (already known to fail, AI-N24).
