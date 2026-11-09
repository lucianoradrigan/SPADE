# SPADE — platform status (2026-11-09)

What the platform does today, what is only simulated and what is open. It replaces
[`plataforma_estado_actual.md`](plataforma_estado_actual.md) (2026-09-15, Spanish, superseded).
The audit trail behind every statement is [`VALIDATION_SUMMARY.md`](VALIDATION_SUMMARY.md); when the
two disagree, the validation summary and the audit reports win.

> **Everything is simulated.** SPADE ingests no measured drive data. No `.tflite` file has run on a
> Raspberry Pi 5 or an ESP32. The only board measurements are on a TI C2000 (LAUNCHXL-F28379D),
> replaying simulated windows from flash: they check numerical parity and timing, not accuracy on a
> real machine. PLECS itself has never been run here.

## 1. Plants

| System | Plants (`plant_config_id`) | Controllers | Status |
|---|---|---|---|
| DC motor | **`dc_perm_ex_v2`, `dc_pmg132_v2` (default in the GUI, the assistant and the DC dataset scripts)**; `dc_perm_ex_v1`, `dc_pmg132_v1` (**GEM-faithful mode**, frozen bit for bit) | PI cascade, linear MPC | v2 uses a discontinuous-conduction converter (`dc_converter_model="dcm_exact"`) that corrects S-21; v1 keeps GEM's one-quadrant converter and its braking artifact (S-21 open there, with a non-blocking warning). The `Scenario` dataclass default is still v1, so library callers that pass no plant get v1 |
| PMSM | `pmsm_v1` | FOC with MTPA or naive current split | Produces datasets; no fault model and no diagnosis |
| VSC + L-C filter | `vsc_dpc` | Trained DPC network | Robustness probes and a dataset-upload evaluator |
| Synchronous buck | native averaged buck, surrogate, OpenModelica FMU (`SPADE/OmBuck-v0`) | PI cascade | Gymnasium environments; FMU plants pass through a 13-check verification before publication |

Sources: `docs/audit/assistant/e3_v2_plants.md`, `dc_perm_ex_v2_convergence.md`,
`dc_perm_ex_v2_equivalence.md`; S-21 in [`SIMULATION_STABILITY_REPORT.md`](SIMULATION_STABILITY_REPORT.md).

## 2. Dashboard pages

`src/driveflow/viz/app.py` (`PAGES`), checked against the README by `tests/test_readme_pages.py`:

- **Workspace:** Overview, Experiment Runs.
- **Simulation:** DC Motor, PMSM / FOC, VSC / DPC, DC-DC Converters, PLECS Plant Studio.
  The DC Motor page's advanced options choose the plant mode (v2 default, v1 GEM-faithful), load torque,
  supply, limits and nominal values.
- **Design & Control:** Tuning Studio, RL Lab.
- **Intelligence:** AI Diagnostics, Edge Deployment, Transfer Learning, Model Registry, Assistant.
- **Operations:** Monitoring, Alert History.
- **Help:** Documentation, Algorithm Guide.

## 3. AI layer

**Promoted models** (`configs/registry.yaml`; every result on simulated windows):

| Entry | Run | Notes |
|---|---|---|
| `dc_motor/pc/classifier` | `pc_server_5class_w2048/2026-11-04_run01` | Trained on corrected-converter data (AI-N27 mitigated). Confidently wrong at 50 rad/s, below its grid (AI-N29, open) |
| `dc_motor/rpi5/classifier` | `rpi5_resnet1d_se_5class_w2048/2026-11-04_run01` | Corrected-converter data; too large for the C2000 (AI-N25) |
| `dc_motor/esp32/classifier` | `esp32_dscnn/2026-10-18_run01` | 3 classes; trained on data with the S-21 artifact (AI-N27 open for ESP32). The pre-registered int8 repair did not pass (AI-N28) |
| `vsc_dpc/pc/regressor` | `pc_full/2026-10-19_run01` | |
| `vsc_dpc/rpi5/regressor` | `rpi5_tcn/2026-10-20_run01` | About 20× its window on the C2000 at 10 kHz (AI-N24, AI-N26) |
| `vsc_dpc/esp32/regressor` | `esp32_tiny/2026-10-19_run01` | Keeps up on the C2000 at 10 kHz (19.7 ms against 25.6 ms) |

Ball and cage faults are synthetic-only, with uncalibrated amplitude. Every promoted classifier has
an `audit_findings.json` shown in the Model Registry.

**Sample rate.** All diagnosis data are sampled at 10 kHz (TAU = 100 µs). Every run records its
`fs_hz` in `run_metadata.json`, and the edge examples and the C2000 exporter read it from the run
(AI-N26, mitigated in software; board re-measurement pending).

## 4. Edge deployment

| Board | What exists | What is checked |
|---|---|---|
| ESP32 | int8 TFLite export, generated inference example, tutorial | Example: syntax and normalization only. Never run on a board |
| Raspberry Pi 5 | TFLite export, Python and C++ examples | Python example equals Keras on a PC; C++: syntax and normalization only. Never run on a board |
| TI C2000 (LAUNCHXL-F28379D) | float32 C kits (`experiments/export_c2000.py`, output in `<run>/c2000/`) | Measured on the board on 2026-10-30 with simulated windows from flash ([report](audit/c2000/report.md)); C5 re-read at 10 kHz post hoc ([`c5_post_hoc_10khz.md`](audit/c2000/c5_post_hoc_10khz.md)) |

## 5. External plants and portability

- **OpenModelica FMUs:** Model Exchange variants compiled from source on Linux, Windows and macOS in
  CI; numerical results identical on the three OSes; the per-step latency criterion depends on the
  runner ([`x1_fmu_portability.md`](audit/plecs_om/x1_fmu_portability.md)). Release `fmu-v2`.
- **PLECS:** RPC sweeps and PLECS-exported FMUs are supported by the code; never run with PLECS here.
- **Windows installer:** double-click installer (release `app-v4`), checked only on a GitHub
  Windows runner, not on a real PC (`packaging/windows/RELEASE_NOTES.md`).

## 6. Assistant

**Experimental / internal demo.** Local LLM through Ollama (`qwen3.5:9b`, tag and digest in
`configs/assistant.yaml`); no data leave the machine. The LLM translates and Python decides: every
number enters through a tool with its origin, SPADE facts are cards from the catalog, and only the
user confirms `run_experiments` / `run_tuning`. Phase 1 closed on 2026-11-09: the last pre-registered
benchmark, F1.13, met 11 of 13 criteria (false statements and the off-lab p95 failed;
[`f1_13_benchmark.md`](audit/assistant/f1_13_benchmark.md)), so by the pre-committed closing rule it
is not for laboratory use (AI-N30).

The same tools are also available to a local MCP client (`python -m driveflow.assistant.mcp_server`,
[`assistant/MCP.md`](assistant/MCP.md)), with the same policies; a launch needs the user's
confirmation through MCP elicitation.

## 7. Open findings

| ID | Status | What |
|---|---|---|
| S-21 | Corrected in v2 · open in v1 | GEM's one-quadrant converter artifact |
| S-22 | Documented · test stabilized | An FMU timing check fails with the machine loaded; its test has a declared tolerance |
| S-24 | Open | `Scenario` compares the voltage limit with the supply only when both are given |
| AI-N24 | Open | `rpi5_tcn` too slow on the C2000 |
| AI-N25 | Open | Promoted RPi5 classifier does not fit the C2000 |
| AI-N26 | Mitigated (software) · board re-measurement pending in B4 | 10 kHz data, 5 kHz budget |
| AI-N27 | Mitigated for PC and RPi5 · open for ESP32 | S-21 artifact in training data |
| AI-N28 | Open | ESP32 int8 fragility |
| AI-N29 | Open | PC classifier wrong at 50 rad/s |
| AI-N30 | Open | Assistant does not meet its benchmarks |

Full lists: [`SIMULATION_STABILITY_REPORT.md`](SIMULATION_STABILITY_REPORT.md),
[`AI_TECHNICAL_AUDIT.md`](AI_TECHNICAL_AUDIT.md).

## 8. Tests

Last full suite on `dev` before round B (`b3ca97c`, 2026-11-08): 1853 passed, 1 skipped, 9 xfailed,
0 failed (`PYTHONPATH=src python -m pytest -q`). Byte-for-byte regression tests pin the v1 plants.
Tests that call the real LLM are marked `llm` and run only with `--run-llm`.
