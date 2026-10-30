# C2000 export: pre-registered criteria

Written 2026-10-30, before the first cl2000 build and the first board measurement. Target: LAUNCHXL-F28379D, CPU1 at 200 MHz.

Runs under test: `esp32_dscnn/2026-10-18_run01` (dc_motor classifier), `esp32_tiny/2026-10-19_run01` (vsc_dpc regressor), `rpi5_tcn/2026-10-20_run01` (vsc_dpc regressor, promoted for the rpi5 tier; "candidate" in the first version of this file was a wrong label, corrected without changing any criterion).

| ID | Criterion | Pass |
|---|---|---|
| C1 | Build | `make` with CGT 25.11.1.LTS exits 0 and links without section overflow |
| C2 | Memory | arena ≤ 131 056 B; weights + reference windows ≤ 327 680 B |
| C3 | Parity | on the board, max absolute output error against Keras ≤ 1e-3 over all reference windows, in the model's output units (probability, or normalized signal) |
| C4 | Top-1 | classifier only: 100 % agreement with Keras over the reference windows |
| C5 | Real time | worst latency (normalization + inference) below the window duration: 102.4 ms for 512 samples at 5 kHz, 51.2 ms for 256 samples at 5 kHz |

C5 is the non-overlapping case: the board finishes one window before the next one is complete. It says nothing about any control-loop deadline.

Replaying simulated windows from flash is a parity and timing test. It does not exercise ADC acquisition or real sensors.
