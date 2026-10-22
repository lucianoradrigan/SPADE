# Data dependencies

Neither dataset below is redistributed in this repository (see `.gitignore`'s `*.parquet`/`/data/`
rules and Design notes in `README.md`). Both must be downloaded separately, under their own
license, to reproduce the pieces of this repo that depend on them.

## KAt-DataCenter (Paderborn) bearing dataset

- **Source**: Chair of Design and Drive Technology, Paderborn University -- the "KAt-DataCenter"
  bearing damage benchmark.
- **Download**: https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download
  (~20.8 GB, distributed as per-bearing-code `.rar` archives; extract to one subfolder per code,
  e.g. `<dataset_root>/K001/N15_M07_F10_K001_1.mat`).
- **License**: CC BY-NC 4.0 -- non-commercial use only. Citation required:
  Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). "Condition Monitoring of Bearing
  Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A
  Benchmark Data Set for Data-Driven Classification." *PHM Society European Conference*, 3(1).
  (Verify the exact citation/BibTeX against the download page above before publishing derived
  work -- not re-verified against the live page as part of this document.)
- **What in this repository depends on it**:
  - `src/driveflow/sim/vibration/calibration.py` -- fits Module B's modal filter and
    background-noise gain against real healthy-bearing PSDs (`experiments/calibrate_module_b.py`
    is the script that runs this fit; its output is the small, already-committed
    `configs/vibration_module_b.yaml`, not the dataset itself).
  - `experiments/verify_torque_vibration_coherence.py`, `experiments/verify_cross_fault_frequency_coherence.py`,
    `experiments/verify_vibration_separability_auc.py` -- the three empirical-reproduction scripts
    (see their own docstrings), all read real recordings directly via `--dataset-root`.
- **Not redistributed**: confirmed -- no `.mat`/`.rar` file, nor anything derived from raw
  recordings, is committed to this repository. Only small fitted scalars (`configs/vibration_module_b.yaml`)
  and printed/reported statistics are versioned.

## Data4train.mat (DPC training/evaluation data)

- **Source**: ships inside the original [DPC4PowerElectronics](https://github.com/aipoweraau/DPC4PowerElectronics)
  repository (the MATLAB toolbox `src/driveflow/control/dpc/` is a Keras/TensorFlow port of --
  see `README.md`'s Credits section), not this repository.
- **License**: MIT (same as DPC4PowerElectronics itself).
- **What depends on it**: `src/driveflow/control/dpc/train.py::load_training_data`,
  `experiments/train_dpc.py`, `experiments/evaluate_dpc.py`, and the fine-tuning scripts
  (`experiments/finetune_dpc_closed_loop.py`, `experiments/continue_finetune_dpc.py`) -- all take
  its path as an explicit `--data-path`/`--dataset-root` argument, never a bundled default.
- **Not redistributed**: confirmed -- not present anywhere in this repository. The 3 shipped DPC
  checkpoints (`configs/*.weights.h5`) are the small (~21KB) *fitted weights*, not the training
  data itself; retraining or re-evaluating from scratch requires downloading `Data4train.mat` from
  the original repository first.

## NASA IGBT accelerated aging dataset (task E11, round 2026-10-21)

- **Source:** NASA Prognostics Center of Excellence (PCoE), "Insulated Gate Bipolar Transistor (IGBT) Accelerated Aging". https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging. Maintainer listed: Christopher Teubert. Contact in the archive's Readme: Jose R. Celaya and Kai Goebel.
- **Download:** `IGBTAgingData_04022009.zip` (240,501,751 bytes, SHA-256 `77cc930c68d7674e694b4aa4f33b94e854bb66b147deb170d0131662f176c2e0`), from https://data.nasa.gov/docs/legacy/IGBTAgingData_04022009.zip on 2026-10-21. Stored in `data/upstream/fund/igbt/`, extracted to `extracted/`. Not committed.
- **License:** the catalogue entry's license field points to https://www.usa.gov/government-works (U.S. Government work, public domain in the U.S.). Access level "public". No use restriction or citation is stated on the page.
  - The archive cites Sonnenfeld, Goebel, Celaya, "An Agile Accelerated Aging, Characterization and Scenario Simulation System for Gate Controlled Power Transistors", IEEE Autotestcon 2008, and includes that PDF. Cite it in derived work.
  - The `.mat` copy inside Fund. (`9_Case_Studies_PE/IGBT_Maintenance/`) was not used. Fund.'s non-code content is CC BY-NC 4.0, so the original NASA download is used instead.
- **Contents** (from the archive's Readme files):
  - 6 aged devices: one with a DC gate bias (continuous slow signals, 4.2 h, failed by latch-up); one with a square-wave gate (A17, 2.9 h, steady-state records plus transients); four with a square-wave gate and SMU characterization (Devices 2–5, 0.4–0.9 h per file, two to three files each).
  - SMU characterization of 40 new devices.
  - The Readme of the four-device set states: "many of the transient measurements are missing… the collector-current measurements are not accurate and have a drift of ~600mA… the steady-state measurements are not scaled properly". Those devices were sent to the University of Padova, which suggests they were not aged to failure. Confirming that needs the aging log `Device Aging log.xls`, which needs an `.xls` reader that is not a SPADE dependency.
- **What depends on it:** nothing yet. E11 stopped at the feasibility check: `docs/audit/expansion/t7_igbt_feasibility.md`.
## DAB_sample1.plecs (task E9 preparation, round 2026-10-21)

- **Source:** Fund. = XinzeLee/Fundamentals_of_AI_for_PE, `8_PE_Simulation_Automation/PlecsAutomation/DAB_sample1.plecs` (PLECS 4.7 model, author Xinze Li). Downloaded 2026-10-21 from `https://raw.githubusercontent.com/XinzeLee/Fundamentals_of_AI_for_PE/main/8_PE_Simulation_Automation/PlecsAutomation/DAB_sample1.plecs`. 389,722 bytes, SHA-256 `15c457462dc61820a7324e002f2ba9e8ee3ff766ba524bc295255742e8078a3a`. Stored in `data/upstream/fund/dab/`. **Not committed.**
- **License: unclear.** Fund.'s NOTICE dual-licenses "code" ("executable or functional components": Apache 2.0) and "educational content" ("text, figures, explanatory and instructional material": CC BY-NC 4.0). A circuit model fits neither definition cleanly. Following decision D8, it is treated as CC BY-NC 4.0: **non-commercial use only**, attribution to Xinze Li / Fund., and not redistributed.
- **What depends on it:** only the hand-declared DAB manifest template (`integrations/plecs/manifest.EXAMPLE_DAB`), which records its ModelVars. The circuit has no top-level outports, so an RPC sweep needs outports added in PLECS first. No dataset was generated from it (not run: no PLECS).

## OpenModelica toolchain, Modelica Standard Library and FMU runtime (round PLECS/OM, task B0)

- **OpenModelica:** official Docker image `openmodelica/openmodelica:v1.27.1-minimal` (OpenModelica 1.27.1), pinned by digest in `scripts/fmu/OPENMODELICA_VERSION`.
  - It is used only to BUILD the FMUs (`scripts/fmu/om.sh`, `scripts/fmu/build_fmus.sh`); it is not a Python dependency.
  - Licence of the tool: OSMC-PL. Recipients may choose BSD New, AGPL v3 or OSMC-PL 1.8, per the runtime header https://openmodelica.org/osmc-pl/osmc-pl-1.8-runtime.txt.
- **Modelica Standard Library (MSL) 4.1.0+maint.om:** installed once with `installPackage(Modelica, "4.1.0+maint.om", exactMatch=true)` into `data/fmu/openmodelica_home/libraries/` (not committed).
  - Licence: **BSD 3-Clause** (Modelica Association).
  - SPADE's models in `modelica/SPADE/` use its `Electrical.PowerConverters`, `Electrical.Machines`, `Electrical.Analog` and `Blocks` packages.
- **What an exported FMU contains:**
  - the model's generated code, which carries the licence of its source: SPADE's `.mo` files (MIT) and MSL (BSD-3);
  - OpenModelica's simulation runtime, under the BSD New / AGPL v3 / OSMC-PL 1.8 choice above;
  - third-party code, e.g. SUNDIALS CVODE, BSD 3-Clause.
  - Choosing **BSD New** for the runtime permits redistributing the binary FMU **with the licence notices included**.
  - Before any FMU is attached to a release (task B8), the license files actually inside the built FMUs are listed and checked per component. If a component's licence does not permit redistribution, B8 stops and reports it.
  - Sources: the runtime header above; OpenModelica forum, "licensing of runtime" (https://openmodelica.org/forum/default-topic/24-licensing-of-runtime.html) and "binary executable or fmu to commercial use" (https://www.openmodelica.org/forum/default-topic/2666-binary-executable-or-fmu-to-commercial-use.html). Those forum answers come from the OpenModelica developers but are not a legal opinion.
- **fmpy 0.3.32** (extra `[fmu]`, decision D3): BSD 2-Clause.
