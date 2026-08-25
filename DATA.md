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
