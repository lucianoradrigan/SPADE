# C2000 export: board results

Measured 2026-10-30 on a LAUNCHXL-F28379D (silicon rev. C), CPU1 at 200 MHz, against the criteria in [`criteria.md`](criteria.md). Raw reports: the `.json` files in this folder.

| Run | Arena | Weights | Worst latency | Window | Max error vs. Keras | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `esp32_dscnn/2026-10-18_run01` (classifier, 3 classes) | 48.0 KB | 1.8 KB | 68.5 ms | 102.4 ms | 1.2e-7 | pass | pass | pass | pass (6/6) | **pass** |
| `esp32_tiny/2026-10-19_run01` (regressor) | 16.0 KB | 10.3 KB | 19.7 ms | 51.2 ms | 4.8e-7 | pass | pass | pass | n/a | **pass** |
| `rpi5_tcn/2026-10-20_run01` (regressor, promoted for `vsc_dpc/rpi5`) | 64.0 KB | 109.1 KB | 520.1 ms | 51.2 ms | 1.2e-6 | pass | pass | pass | n/a | **FAIL** |

The latency covers normalization and inference. The six classifier windows hold two windows of each class.

## Findings

- **AI-N24:** `rpi5_tcn` runs correctly on the board but takes 10 times the window (C5 fail). See `docs/AI_TECHNICAL_AUDIT.md`.
- **AI-N25:** the promoted `dc_motor/rpi5` classifier (`rpi5_resnet1d_se_5class_w2048/2026-10-20_run01`) needs a 448 KB arena against 128 KB, so no kit is produced for it.

## Commands

```bash
python -m experiments.export_c2000 --domain dc_motor --tier esp32 --block classifier
python -m experiments.export_c2000 --domain vsc_dpc --tier esp32 --block regressor
python -m experiments.export_c2000 --domain vsc_dpc --block regressor --run-dir configs/regressors/rpi5_tcn/2026-10-20_run01
python -m experiments.export_c2000 --domain dc_motor --tier rpi5 --block classifier --check
make -C <run>/c2000
python3 scripts/edge_benchmark/c2000_collect.py <run>/c2000
```

## Not executed

- ADC acquisition, or any real signal: **no ejecutado / not executed**, the board has no sensors attached. Every window was simulated and replayed from flash, so the model's accuracy on real signals is unknown.
- `spade_model_decide()` (temperature and abstention) on the board: **no ejecutado / not executed**. The firmware does not call it; only the gcc test covers it (`tests/test_c2000_export.py`).
- Out-of-distribution scoring: not ported, so the device never answers UNKNOWN.
- Two-CPU or CLA execution: not used; everything ran on CPU1.
