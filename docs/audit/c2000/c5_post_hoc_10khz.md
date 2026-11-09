# C5 at 10 kHz (AI-N26) · **POST HOC**

**Post-hoc analysis, not pre-registered** (2026-11-09). The pre-registered C5 verdicts in [`criteria.md`](criteria.md) and [`report.md`](report.md) are **not changed**. Nothing was re-measured: the latencies are the board's own reports, as already recorded in [`ai_n26_sample_rate.json`](ai_n26_sample_rate.json).

- **Command:** `python experiments/report_c5_post_hoc_10khz.py`. Raw values: [`c5_post_hoc_10khz.json`](c5_post_hoc_10khz.json).
- **Rate:** every training dataset has `fs_hz = 1 / TAU = 10 000 Hz` (TAU = 1e-4 s). C5 assumed 5 kHz.
- **Windows at 10 kHz:** 512 samples = **51.2 ms**; 256 samples = **25.6 ms**.

| Run | Samples | Worst latency (board) | Pre-registered window (5 kHz) → C5 | Window at 10 kHz → C5 (post hoc) | Latency / window |
|---|---|---|---|---|---|
| `esp32_dscnn/2026-10-18_run01` | 512 | 68.5 ms | 102.4 ms → pass | 51.2 ms → **FAIL** | 1.34× |
| `esp32_tiny/2026-10-19_run01` | 256 | 19.7 ms | 51.2 ms → pass | 25.6 ms → **pass** | 0.77× |
| `rpi5_tcn/2026-10-20_run01` | 256 | 520.1 ms | 51.2 ms → FAIL | 25.6 ms → **FAIL** | 20.32× |

## Reading

- `esp32_dscnn/2026-10-18_run01` passed the pre-registered C5, but at the data's real rate it does not keep up with non-overlapping windows.
- `esp32_tiny/2026-10-19_run01` keeps up at either rate.
- `rpi5_tcn/2026-10-20_run01` failed C5 already (AI-N24); at 10 kHz it takes about 20× its window.

## Not executed

- Re-measuring on the boards: **no ejecutado / not executed**; it belongs to B4. The latency does not depend on the assumed rate, only the budget does.
