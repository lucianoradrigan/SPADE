# Guide: training, evaluating and adapting diagnosis models in SPADE

This guide covers three tasks in the GUI and what happens behind them:
1. **Train** a classifier on your own simulated grid (AI Diagnostics → *Train a model*).
2. **Evaluate** any classifier on a sample run or a file (AI Diagnostics → *Evaluate a model*).
3. **Adapt** a model to new data or new classes (Transfer Learning).

All data in these workflows is **SIMULATED** unless you upload your own. No model is promoted
without a separate, explicit step. Evidence for everything below:
[`audit/ai/custom_training_tl.md`](audit/ai/custom_training_tl.md).

---

## 0. Before you start

- Install SPADE and open the app (README → *Installation, step by step*).
- **Start the worker.** Training runs as a background job: Experiment Runs → *Start local worker*,
  or in a second terminal (environment active):
  ```bash
  python -m driveflow.jobs.worker serve
  ```
  Without a worker, the job waits in the queue. You can keep using SPADE while it runs.

---

## 1. Train a classifier (AI Diagnostics → *Train a model*)

Pick the **Domain** at the top of the page (`dc_motor` or `vsc_dpc`), open the *Train a model* tab
and fill in the form from top to bottom.

| Section | What you choose | Notes |
|---|---|---|
| 1 · Classes | Two or more fault modes: `healthy`, `outer_race`, `inner_race`, `ball`, `cage`, and your custom fault types. `vsc_dpc`: `normal`, `capacitor_degraded`. | Custom fault types are defined on DC Motor → *Parameters* (a characteristic order, i.e. events per shaft revolution). They need a mechanical severity, because they have no calibrated amplitude. |
| 2 · Operating conditions | PI and/or MPC; speed or torque control; a setpoint range and its number of levels; severity scales; electrical severity; PI setpoint weight. `vsc_dpc`: load resistances, reference magnitudes and frequencies, voltage limit. | Every class gets the **same** operating cells, so the classes are balanced. |
| 3 · Robustness | Max sensor noise, share of noise-free runs, per-run drift of ψ_e, J and r_a, base motor parameters. | Noise and drift are drawn per run from one seeded RNG, so the grid is reproducible. |
| 4 · Runs | Duration, start-up to discard, runs per cell, grid seed. | The speed loop settles in about 0.45 s. Discarding about 0.5 s avoids training on the transient. |
| 5 · Model | Architecture preset, window length, **input channels**, epochs, batch size, windows per class, validation and test shares, training seed, rpm level, OOD detector. | Choose channels to build e.g. a current-only (MCSA) or a vibration-only model. One window must fit in the kept part of each run. |
| 6 · Start | Shows the number of runs per class and a rough time estimate. | The estimate was measured on the development laptop: about 3.7 s of computer time per simulated second (dc_motor) and about 14 s (vsc_dpc). More processes did not shorten it there. |

The page refuses requests it cannot run and says what to change. Examples:
- only one class;
- more than 600 runs;
- fewer than 3 runs per class (train, validation and test are split **by run**);
- a window longer than the kept part of each run;
- a channel the domain does not have.

**Result.** Select the training run on the right to see:
- test macro-F1 and accuracy;
- per-class recall;
- the confusion matrix;
- whether an OOD detector was fitted.

The model is a **candidate** in `configs/classifiers/custom/<name>/<date>_runNN/`:

| File | Content |
|---|---|
| `config.yaml` | the architecture |
| `model.weights.h5` | the weights |
| `metrics.json` | metrics, classes, channels, calibration, OOD detector |
| `training_request.json` | the exact form you submitted |
| `run_metadata.json` | commit, library versions, seed, dataset SHA-256 |

The dataset is saved as `data/custom/<name>_<run>.parquet`. Both folders are gitignored: they are
your local work.

*Evaluate this model* selects the candidate in the Evaluate tab.

**Read the metrics critically.** In the evidence run, a 4-class model trained on 72 short runs
reached macro-F1 0.417: it never recognised inner_race. A small grid can train a model that is not
usable. Add runs, lengthen the runs or the window, and check per-class recall, not only the average.

---

## 2. Evaluate a model (AI Diagnostics → *Evaluate a model*)

1. **Classifier:** the promoted model, or any other classifier on this machine (custom-trained,
   transfer-learning candidates, imported, historical runs). A non-promoted model carries the badge
   **CANDIDATE — NOT PROMOTED**.
2. **Data source:**
   - *Generate a sample run*, with the same options as the simulation pages: controller, fault
     (custom ones included), speed or torque mode, severities, noise, motor parameters, PI setpoint
     weight. For VSC: R, reference, voltage limit, capacitor degradation.
   - *Upload a file* (CSV or JSON, channels only; see *Required format*).
3. **Read the outcome:**
   - the condition state (known / uncertain / unknown / invalid input);
   - the class probabilities;
   - the OOD check;
   - the optional simulated-agent hypothesis.

   *Export evidence (JSON)* saves the assessment with its provenance.

---

## 3. Adapt a model (Transfer Learning)

### 3.1 Choose the base model

| Option | When to use it |
|---|---|
| **Promoted model (registry)** | Adapt what is in production. |
| **Another model on this machine** | Start from a model trained in §1, an earlier transfer-learning candidate, or a historical run. |
| **Import a model file** | Start from a third-party model. |

**Supported formats** (also shown on the page):

| Format | Supported? | Notes |
|---|---|---|
| SPADE run directory | Yes | `config.yaml` + `model.weights.h5` + `metrics.json` |
| Keras v3 `.keras` (full model) | Yes | Loaded in safe mode. `Lambda` and non-Keras custom layers are refused, because they would run code stored in the file. |
| Keras HDF5 `.h5` (full model, `model.save()`) | Yes | Same checks |
| `.weights.h5` (weights only) | No | No architecture to rebuild |
| TensorFlow SavedModel, `.tflite` | No | Inference-only in Keras 3: cannot be fine-tuned |
| ONNX `.onnx`, PyTorch `.pt` / `.pth` | No | Would need converter dependencies SPADE does not install. Convert to `.keras` first. |

To import a model:
1. Upload the file.
2. Declare its **input channels**, in input order, from SPADE's channel names.
3. Declare its **output classes**, in output order. Use `normal` for the healthy class.

SPADE checks the shapes (one input `(window, channels)`, one softmax `Dense` output) and saves the
model as a run in `configs/classifiers/external/<name>_<sha12>/`. SPADE feeds it per-window
z-scored windows sampled at 10 kHz. If the model was trained with other preprocessing, it starts
poorly and needs fine-tuning.

### 3.2 Step 1 — Data

Simulate acquisitions:
- fault classes (custom included);
- a speed range and maximum noise for `dc_motor`;
- a degraded capacitance for the `vsc_dpc` classifier.

You can optionally add your own training file (`label`, `source_file`, `timestamp_s` and the
channels, at 10 kHz). *Prepare dataset* merges whole acquisitions, windows them and splits by
acquisition. It reports the dataset hash, the split and the integrity checks.

Classes the base model does not know are allowed. The page warns you that only a **new output
layer** can learn them.

### 3.3 Step 2 — Transfer mode

| Mode | Strategy | What is trained | Use it when |
|---|---|---|---|
| **Full transfer** | Full fine-tune (`FULL_FINETUNE`) | Every layer, one learning rate | You have enough data and the conditions changed a lot |
| Full transfer | Discriminative LR (`DISCRIMINATIVE_LR`) | Every layer, larger rate on the output layer | As above, protecting the early features more |
| **Feature transfer** | Freeze first N (`FEATURE_EXTRACTOR`) | Layers after the first N | Little data; the low-level features still apply |
| Feature transfer | Simplified adapter (`ADAPTER`) | Only the last N layers (this is **not** LoRA) | Very little data |
| Feature transfer | New output layer (`NEW_HEAD`) | A new output layer (plus, optionally, the last N feature layers) | **New classes**, or a third-party model whose outputs do not match |

With a new class set, only `NEW_HEAD` is offered: full fine-tuning is refused.

### 3.4 Steps 3 and 4 — Validate and promote

- The base and the candidate are evaluated **once**, on the same independent test windows.
- The base model is never trained: a copy is, and its weights fingerprint is checked afterwards.
- **When the classes change,** the base cannot be scored on them. The base-vs-candidate check is
  then NOT_APPLICABLE, and an absolute **minimum macro-F1** becomes required. The reviewer must
  enter it; without it, the gate stays blocked.
- *Save as candidate* writes the run next to its base (`<base>_tl/<date>_runNN/`) without
  touching the registry.
- *Promote* is enabled only when the gate is APPROVED, which includes a named reviewer. The
  registry re-checks the evidence anyway, and *Model Registry* can roll back.

---

## 4. Where things are

| What | Where |
|---|---|
| Custom-trained models | `configs/classifiers/custom/` (gitignored) |
| Imported models | `configs/classifiers/external/` (gitignored) |
| Transfer-learning candidates | `<base config folder>_tl/<date>_runNN/` |
| Datasets of custom training | `data/custom/` (gitignored) |
| Jobs, logs, artifacts | Experiment Runs (workspace folder, `$SPADE_WORKSPACE`) |
| Promoted models | `configs/registry.yaml`; history in `configs/registry_history.yaml` |

Code:
- `driveflow/ai/custom_training.py`: request, grid, simulation, training;
- `driveflow/jobs/handlers.py`: the `classifier_training` job;
- `driveflow/ai/transfer/external.py`: import;
- `driveflow/ai/transfer/pipeline.py` and `workflow.py`: strategies, gate inputs;
- `driveflow/models/training.py`: shared training code, also behind `experiments/train_model.py`.

The same steps from a script, without the GUI:

```python
from driveflow.ai import custom_training as ct

req = ct.default_request("dc_motor")          # edit classes / operating / model as in the form
df, outcomes = ct.simulate(ct.build_conditions(req))
run_dir = ct.train(req, df)
```

---

## 5. Limits

- Every signal is simulated. Ball, cage, custom faults and capacitor degradation have
  **uncalibrated** amplitudes, and the GUI labels them synthetic-only.
- A good test score on a simulated grid says nothing about a real machine.
- Imported models: SPADE cannot know their original preprocessing. Only Keras full models are
  accepted.
- The time estimate is an order of magnitude from one laptop.
- No acceptance threshold has been confirmed for models with a new class set. The page asks the
  reviewer for it.
