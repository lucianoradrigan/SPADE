# SPADE v0.2.0 — release notes

**All results are simulated.** SPADE ingests no measured drive data. No `.tflite` file has run on a
Raspberry Pi 5 or an ESP32. The only board measurements are on a TI C2000 (LAUNCHXL-F28379D),
replaying simulated windows from flash; PLECS itself has never been run here.

Changes since `main` at `df1950f` (2026-11-02). The evidence for each item is in
[`docs/VALIDATION_SUMMARY.md`](docs/VALIDATION_SUMMARY.md) §12–§14 and the files under
`docs/audit/`.

## New

### Phase 0: simulator options for describing another motor
- **DC motor options:** `Scenario.load_torque_nm` (constant passive load), supply voltage, motor limits and nominal values, and a controller-design check.
- **PMSM:** `Scenario.pmsm_motor_parameters`, a partial override of the motor's parameters.
- **Catalog:** `describe_plant()` and `configs/plant_catalog.yaml`, which describe every plant, its fields, their defaults and their valid ranges.
- **Simulation cost:** CPU seconds per simulated second for each native plant, measured on one machine only.
- **Tuning another DC motor:** the tuning grids scale with the base motor's ratings. The default load of a non-stock motor is 2 % of its rated torque.
- **New plant `dc_pmg132_v1`,** with the PMG 132 datasheet values (S-20).
- All new fields are optional. When a field is left out, the original code path runs, bit for bit.

### v2 DC plants (default of new simulations) and finding S-21
- **S-21:** GEM's one-quadrant converter applies the full supply voltage for a whole period when the armature current is negative. With a light load, the speed then rises instead of coasting.
- **New plants `dc_perm_ex_v2` and `dc_pmg132_v2`.** Their converter, `dcm_exact`, solves discontinuous conduction in closed form.
  - It converges: worst case 3.0e-5 against 10000 substeps, and it is 28× cheaper than 1000 substeps.
  - The classifiers give the same answers on its data as on the reference data: agreement 1.000.
- **Where v2 is the default:** the GUI, the assistant and the DC dataset scripts use the v2 plants. The `Scenario` dataclass default is still `dc_perm_ex_v1`.
- **The v1 plants** stay bit for bit as the **GEM-faithful mode**, with a non-blocking S-21 warning.
- A draft upstream issue for gym-electric-motor is in `docs/audit/upstream/`; it is not published.

### AI-N27: classifiers retrained without the artifact
- The promoted dc_motor classifiers had been trained on data containing the S-21 artifact.
- **PC and RPi5:** the grids were regenerated with the corrected converter and the models retrained under a pre-registered criterion. The new models were promoted on 2026-11-04: `pc_server_5class_w2048/2026-11-04_run01` and `rpi5_resnet1d_se_5class_w2048/2026-11-04_run01`.
- **ESP32:** the ESP32 model keeps the earlier run, with its audit finding.

### Assistant (experimental / internal demo)
- **The Assistant page and CLI** (`python -m driveflow.assistant`) draft simulations and tuning runs from a description.
- **Local model:** `qwen3.5:9b` through Ollama, pinned by tag and digest. No data leave the machine.
- **Safeguards:**
  - every number enters the draft through a tool, with its origin;
  - SPADE facts are shown as cards from the catalog;
  - only the user can confirm a launch;
  - the runs it launches are labelled `SIMULATED · AI-PLANNED`.
- **Benchmarks:** the pre-registered benchmarks F1.11, F1.12 and F1.13 were **not met** (F1.13: 11 of 13 criteria). The assistant is therefore an internal demo, not a laboratory tool (AI-N30). How to run the benchmarks: `docs/assistant/BENCHMARKS.md`.
- **Avatar ("PhD. Radrigan"):** its images are **not under the MIT license**. They have their own terms in `licenses/avatar/`, and can be replaced through `persona.avatar_dir`.

### Other
- **AI-N26 (sample rate):** every run records its sampling rate (`fs_hz`, 10 kHz), and the edge examples and the C2000 exporter read it from the run. The C2000 timing is re-read at 10 kHz post hoc; the pre-registered verdicts are unchanged.
- **Documentation:**
  - `docs/PLATFORM_STATUS.md` (current status);
  - `docs/VALIDATION_SUMMARY.md` §14 (X1–X4, W1, C2000, Edge Deployment, E3, the S and AI-N findings, the assistant);
  - the README page table, checked against the app's pages by a test.
- **Tests:** the last full suite passed with 0 failures. It uses recorded model answers, so it needs no GPU or network. Counts are given below.

### Found by the fresh-clone check before release
- **AI-N32 (fixed):** the assistant now honours `OLLAMA_HOST`, Ollama's standard variable. The loopback rule is unchanged.
- **AI-N33 (corrected):** AI Diagnostics' default sample (300 rad/s, 0.6 s) classified a window inside the start-up transient. The promoted classifier then said 'cage' on a healthy motor.
  - The page now classifies the last window aligned like the training windows.
  - Its default duration is 0.82 s, the shortest that puts that window in steady state.
  - A warning remains for shorter samples.
  - Evidence: `docs/audit/release/ai_n33.md`.

### Repository housekeeping
- **Avatar license:** it now covers every image that shows the avatar, including the screenshots in `docs/screenshots/assistant/`.
- **Superseded documents:** moved to `docs/archive/`.
- **Internal planning documents (in Spanish):** moved to `docs/planning/`.

## Open findings (read before using the results)

| ID | Severity | Status | What it means for you |
|---|---|---|---|
| **S-21** | Major | Corrected in v2 · **open in v1** | The v1 (GEM-faithful) plants keep the converter artifact when braking with a light load |
| **AI-N26** | Medium | Mitigated in software · **board re-measurement pending** | The C2000 timing verdicts assumed 5 kHz; at the real 10 kHz the 512-sample ESP32 classifier does not keep up (68.5 ms for a 51.2 ms window) |
| **AI-N28** | High | **Open** | The ESP32 int8 classifier is fragile; the pre-registered repair did not pass. The promoted ESP32 model was trained on data with the S-21 artifact (AI-N27 open for ESP32) |
| **AI-N29** | High | **Open** | The promoted PC classifier is confidently wrong at 50 rad/s, below its training grid |
| **AI-N30** | High | **Open** | The assistant does not meet its benchmarks; it can state false things about SPADE in free text |

Also open, unchanged from v0.1.0:
- AI-N24: `rpi5_tcn` is too slow on the C2000;
- AI-N25: the promoted RPi5 classifier does not fit the C2000;
- S-22 (documented): one FMU timing check depends on CPU load.

Full lists: `docs/SIMULATION_STABILITY_REPORT.md` and `docs/AI_TECHNICAL_AUDIT.md`.

## Version

- The package, the citation (in the app and in the README) and the Windows installer say **0.2.0**.

## Compatibility

- **Python** 3.11 or 3.12.
- **No new required dependencies.** The assistant talks to Ollama through the standard library; Ollama itself is needed only for the assistant.
- **Datasets made before v0.2.0** used the v1 plant. Pass `--plant dc_perm_ex_v1` to the dataset scripts to reproduce them.

## Verification of this release

Run on `release/v0.2.0` at `721da2f` (2026-11-09, Linux, Python 3.12, Ryzen 5 5600X, machine free):

- **Full suite** (`PYTHONPATH=src python -m pytest -q`): **1906 passed, 1 skipped, 9 xfailed, 0 failed** (20 min 6 s). Every xfail is strict and cites an open finding.
- **Smoke test** (`python scripts/smoke_test.py --launch`): **7/7 PASS**. It checks the app, the Assistant page with the local model runtime down, the Streamlit health endpoint, short DC/PMSM/VSC simulations and inference with a registered model.
- **CI on GitHub:** not run for this branch (it is not pushed). The last CI run on `dev` was on 2026-10-26.
