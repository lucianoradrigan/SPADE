# SPADE

**S**imulation **P**latform for **A**nalysis, **D**iagnosis & **E**lectric drives.

A unified simulation, differentiable predictive control (DPC), and ML fault-diagnosis workbench
for electric drives and power electronics converters, with a live Streamlit dashboard -- now also
covering transfer learning on already-promoted models and a live multi-agent monitoring view.
Nothing on the dashboard is pre-generated or looked up from a table -- every chart is the direct
output of a real simulation, run fresh each time you click Generate.

![SPADE -- Overview page (real screenshot of the running app)](assets/landing.png)

> **Status of the evidence.** Every signal in SPADE is simulated; there is no hardware ingestion.
> Edge `.tflite` files are exported, not hardware-verified. The vibration model is a coarse,
> one-axis calibration that does not generalise per bearing. See
> [`AUDIT_ANSWERS.md`](AUDIT_ANSWERS.md) (Q01–Q30) and
> [`CHANGELOG_IMPLEMENTATION.md`](CHANGELOG_IMPLEMENTATION.md) for the audit of the AI layer and
> what was changed.

**Demo video:** [SPADE on YouTube](https://www.youtube.com/watch?v=qoHtqhicAgY)

## What's inside

Three physically distinct systems, unified under one interface:

| System | Plant | Controller | Fault/robustness angle |
|---|---|---|---|
| **1. DC motor** | Permanently-excited DC motor (same physics/parameters as [gym-electric-motor](https://github.com/upb-lea/gym-electric-motor)) | Native cascaded PI or linear MPC (speed or torque, `controller_type="PI"`/`"MPC"` -- genuinely interchangeable, same plant) | Bearing-fault injection on two independent paths: torque ripple (electrical/MCSA) and synthetic 3-axis vibration |
| **2. PMSM FOC/MTPA** | Salient permanent-magnet synchronous motor | Native dq-frame field-oriented current control, MTPA vs. naive policy | Control-law comparison against the analytic MTPA locus and current-limit circle |
| **3. DPC / Voltage Source Converter** | VSC + L-C filter, identified 2-state model (power electronics, no rotating machinery) | Trained Direct Power Control (DPC) neural network, ported from [DPC4PowerElectronics](https://github.com/aipoweraau/DPC4PowerElectronics) | Off-distribution robustness probes (load resistance, reference magnitude/frequency) + a dataset-upload evaluator for your own data |

System 1 also feeds a fault-injection/diagnosis data-generation pipeline (`src/driveflow/datagen/`)
and a config-driven ML diagnosis stack (`src/driveflow/models/`, `src/driveflow/ai/`,
`src/driveflow/monitoring/`) -- see the "AI layer" section below.

**System 2 (PMSM) produces datasets but has no diagnosis.** Since 2026-10-20 it is a `Scenario`
plant (`plant_config_id="pmsm_v1"`, `controller_type` `"FOC_MTPA"` or `"FOC_naive"`, reference
`pmsm_current_ref_a`): `run_scenario` emits records with the shared schema plus `i_d`, `i_q`, their
references, `u_d`, `u_q` and `omega_rad_s` (`experiments/generate_pmsm_dataset.py`). It still has
**no fault model**, so every record is `normal` and no classifier exists for it; the dashboard's
PMSM page keeps calling `control/classical/pmsm_foc.py` directly for its MTPA-vs-naive comparison.

The dashboard is a multipage Streamlit app (*SPADE · Engineering Workbench*, `src/driveflow/viz/app.py`)
with native navigation; the five original macro-phases map onto its sections:

| Section / page | Former phase | What it does |
|---|---|---|
| Workspace · **Overview** | landing | Real workspace checks, engineering workflow, this session's runs, motor status (UNKNOWN until monitored), clickable system diagram |
| Simulation · **DC Motor** | A | PI or MPC, bearing-fault injection (MCSA + synthetic vibration), KPIs, time signals, spectrum (FFT) with simulator-computed BPFO/BPFI markers, AI suggestion, operating envelope, provenance + export; Advanced Flow mode |
| Simulation · **PMSM / FOC** | A | FOC/MTPA control-law demo -- explicitly *no diagnosis* |
| Simulation · **VSC / DPC** | B | Trained DPC network tracking, complex plane, dataset-upload evaluator |
| Intelligence · **AI Diagnostics** | IA | Promoted classifier/forecaster on a sample run or upload, labelled as AI suggestions with model version |
| Intelligence · **Transfer Learning** | TL | 4-step gated wizard (Data → Setup → Validate → Promote) -- see below |
| Intelligence · **Model Registry** | IA | Every run on disk: status from evidence, lineage, gate, artifacts with SHA-256, edge-export lifecycle, rollback |
| Operations · **Monitoring** | LM | SIMULATED monitoring: agent consensus, status from *all* active alerts, motor status, acknowledge |
| Operations · **Alert History** | LM | Filterable, exportable alert log |
| Help · **Documentation** | -- | Status semantics, data formats, known limitations |

A full, phase-by-phase snapshot of what the platform does today is kept in
[`docs/plataforma_estado_actual.md`](docs/plataforma_estado_actual.md).

## AI layer

A config-driven classifier/regressor stack, a model registry, edge-tier deployment, and
rule-based monitoring agents -- transversal to both domains (`dc_motor`, `vsc_dpc`) without ever
mixing their data or models. Full design rationale in
[`docs/design_ai_layer_transversal.md`](docs/design_ai_layer_transversal.md); this is the short
version.

**Classifiers & regressors** (`src/driveflow/models/{classifiers,regressors}/`): a YAML config
(architecture, tier, domain) plus a generic builder (`build_classifier`/`build_forecaster`) --
adding a layer or changing a kernel size means editing a config file, not touching Python. One
script trains any of them:

```bash
python experiments/train_model.py --config configs/classifiers/pc_server.yaml --dataset data/diagnosis_dataset.parquet
```

The promoted `dc_motor` classifier (`pc_server/2026-10-18_run01`) was trained on an
operating-condition grid (7 speeds x 5 severities, sensor noise, plant drift), keeps the rpm level
as a speed reference, and is temperature-calibrated with an abstention threshold:

```bash
python experiments/generate_diagnosis_grid_dataset.py            # ~8 min, data/diagnosis_grid_dataset.parquet
python experiments/train_model.py --config configs/classifiers/pc_server.yaml \
    --dataset data/diagnosis_grid_dataset.parquet --keep-rpm-level --cap-per-class 3000 --epochs 30
```

Held-out evaluation and its limits (no ball/cage class, simulated data only):
[docs/AI_TECHNICAL_AUDIT.md](docs/AI_TECHNICAL_AUDIT.md) §7.

**Model registry** (`src/driveflow/ai/registry.py`): a single manifest
(`configs/registry.yaml`) resolving `(domain, tier, block)` to whichever trained run is promoted
to production (`experiments/promote_run.py`) -- a deliberate, separate step from training.

**Edge deployment** (Raspberry Pi 5, ESP32): the promoted PC-tier model is distilled into a
smaller architecture per tier (`--distill` on the training script -- DS-CNN/TCN blocks for ESP32,
which structurally can't use recurrent layers) and exported to TFLite (`experiments/export_tflite.py`
-- float16 for Raspberry Pi 5, int8 calibrated against real data for ESP32). The dashboard's IA
tab has a direct download for each promoted `.tflite` file.

**Monitoring agents** (`src/driveflow/monitoring/`): a rule schema (`rules/schema.py`, validated
YAML conditions, no `eval()`) plus one agent per tier -- ESP32 is pure hard-coded thresholds (no
ML), a Raspberry Pi 5 `GatewayAgent` evaluates rules with hysteresis/debounce and can run
disconnected from the PC tier, and a PC-tier `ServerAgent` aggregates alerts across domains and
tracks classifier confidence over time to flag drift.

Try it in the dashboard: pick the "IA" macro-phase, generate a sample run or upload a file, and
see the trained classifier/regressor evaluate it live -- the landing page also has a clickable
system diagram walking through how these pieces connect end to end.

## Transfer Learning workbench

Fine-tuning a model that is already promoted, without retraining it from scratch
(`src/driveflow/ai/transfer/`, plus the dashboard's "Transfer Learning" macro-phase):

- **`loader.py`** wraps the existing registry API (`resolve`/`load_promoted_model`) -- no parallel
  YAML schema and no registry format change. `load_by_generation()` additionally loads *any* past
  run directory by path (promoted or not), so a fine-tune can be stacked on a previous fine-tune.
- **`data_merger.py`** merges freshly-simulated data with external data by **whole acquisitions**
  (`source_file`), never individual rows; `mix_ratio` is measured in training windows and the
  realized ratio is reported. IDs are prefixed `sim::`/`ext::`; groups with out-of-order
  timestamps or a sampling rate other than 10 kHz are excluded, never re-sorted or resampled.
- **`dataset_validation.py`** is the *training* contract for uploads (keeps `label`, `source_file`,
  `timestamp_s` and other metadata; base-model channels and class vocabulary required) -- distinct
  from the inference-only validators used by AI Diagnostics.
- **`pipeline.py`** implements three fine-tune strategies for real: `FEATURE_EXTRACTOR` (freeze the
  first N layers, then a normal `fit()`), `DISCRIMINATIVE_LR` (a manual per-layer-group
  `tf.GradientTape` loop -- Keras `fit()` has no per-layer learning rate), and `ADAPTER` (freeze
  everything but the last N trainable layers; a deliberately simplified, architecture-agnostic
  reading of parameter-efficient fine-tuning, *not* LoRA). An `on_epoch_end` hook feeds the
  dashboard's live loss chart and fires identically on both paths.
  The pipeline always fine-tunes a **clone** of the base model and verifies the base weights'
  SHA-256 fingerprint afterwards.
- **`validator.py`** is task-specific: classification (accuracy, macro-F1, per-class recall,
  confusion matrix, mode collapse) and regression (MAE, RMSE per horizon step, finiteness,
  constant-output detection, vs. base and vs. a persistence baseline), plus convergence. Each check
  is PASS / FAIL / NOT_APPLICABLE / INCOMPLETE; INCOMPLETE blocks.
- **`workflow.py`** runs the whole path without Streamlit: grouped train/val/test split by
  acquisition (no leaky fallback), single evaluation of base *and* candidate on the same test set,
  lineage, candidate saving.
- **`../promotion.py`** is the fail-closed `PromotionGate` (dataset integrity, split
  independence, single test evaluation, validation checks, base unchanged, lineage, human
  approval). `ai/registry.py`'s `promote()` re-evaluates the persisted evidence of any
  transfer-learning run and refuses it unless approved; promotions are atomic, appended to
  `configs/registry_history.yaml`, and reversible with `rollback()`.

In the dashboard the page is a 4-step wizard: **Data** (simulate + optionally upload, prepare
dataset → hash, split manifest, integrity checks), **Setup** (strategy, epochs, proposed
thresholds flagged as requiring confirmation, train), **Validate** (loss curves, base vs.
fine-tuned on the same independent test set, checks), **Promote** (gate decision; *Save as
candidate* writes `<config>_tl/<date>_runNN/` without touching the registry; *Promote* is enabled
only when the gate is APPROVED, and the registry re-checks it anyway). Pushing the new run down to
Raspberry Pi 5/ESP32 reuses the existing `--distill` flag on `experiments/train_model.py`; it is
not a separate mechanism.

The tab deliberately does *not* offer a "pick a new architecture" control: fine-tuning changes
weights, not architecture -- choosing a different one is training from scratch, which
`experiments/train_model.py` already covers.

## Simulation-based agents & Live Monitoring

A second, complementary detection path that needs no trained classifier at all
(`src/driveflow/agents/`): compare a real telemetry window against *simulations* of hypothetical
scenarios (one healthy + N anomalous) and score how much closer the telemetry sits to healthy than
to the nearest fault hypothesis.

- **`base.py`** holds the one shared, concrete `detect_anomaly` algorithm, driven by a `HYPOTHESES`
  class attribute (name -> partial `Scenario` kwargs) that each subclass defines. Only
  `simulate_scenario` is abstract, because only that part is genuinely domain-specific.
- **`detector.py`** implements the distance metrics directly against numpy/scipy -- DTW (exact
  O(n*m) dynamic programming, vectorized per row; no DTW library is a project dependency), strict
  equal-shape Euclidean, and Mahalanobis (pseudo-inverse for near-singular covariances).
- **`dc_motor_agent.py` / `vsc_agent.py`** define hypotheses that match each domain's *actual*
  `Scenario` shape: outer- and inner-race severity pairs for `dc_motor`, simulated at the observed
  speed, and variations of `load_resistance_ohm` away from the DPC network's training value for
  `vsc_dpc`, which has no fault model at all. Features are z-scored with the healthy hypothesis's
  statistics; vibration and current are compared by log-PSD (AI-N09/AI-N10). On 60 labelled
  simulated runs the DC score separates healthy from faulty with ROC AUC 1.0
  (`experiments/report_anomaly_agent_roc.py`).
- **`cache.py`** is a real TTL cache so a hypothesis simulation isn't re-run per telemetry window.
- **`explainer.py`** returns feature importance for the score -- which channels explain most of the
  distance between the telemetry and the closest hypothesis.

The **Monitoring page** (`viz/views/monitoring.py`, labelled *SIMULATED MONITORING* -- each pass
synthesises telemetry with `run_scenario()`) is where the three
detection mechanisms meet as one **Agent Consensus**, all feeding a single session-scoped
`ServerAgent`:

1. the simulation-based detector above,
2. the rule-based `GatewayAgent`, where a real rule exists for that domain (today: only `vsc_dpc`),
3. **ML classifier confidence drift** -- each run loads whatever PC-tier classifier is currently
   promoted for the domain (including one just promoted from the Transfer Learning tab, no extra
   wiring), records its confidence, and reports baseline-vs-recent drift once enough history has
   accumulated.

The output is a per-domain status (`ServerAgent.domain_health`: the highest severity among
**all** active alerts, keyed by `(domain, rule, source)`; UNKNOWN until the domain has been
evaluated, never HEALTHY by default), an alert history table
(`ServerAgent.alert_history`), and the feature-importance explanation for the simulation-based
score. Agent state lives in `st.session_state`, not `st.cache_resource` -- the latter is shared
across every session on the same Streamlit server process, which for a stateful agent (alert
history, hysteresis timers) would leak one user's monitoring history into another's.

Note on the `vsc_dpc` rule: it fires for any load resistance below R* ~= 3.37 ohm (the whole
open-loop-unstable region; it used to watch only [1, 3] ohm -- docs/AI_TECHNICAL_AUDIT.md AI-N03)
and is evaluated on every sample of the simulated run (telemetry time, not the time between
clicks) with a 10 ms on-delay and a 10 ms off-delay, so one "Run monitoring" pass with R < R*
fires it (AI-N02/AI-N08).

## Quick start

Requires Python 3.11 or 3.12, and [`uv`](https://docs.astral.sh/uv/) (or plain `pip`). No GPU or
CUDA is needed. Step-by-step guides with troubleshooting, per operating system:
[Ubuntu](docs/INSTALL_UBUNTU.md) · [Windows (PowerShell)](docs/INSTALL_WINDOWS.md) ·
[macOS (Apple Silicon / Intel)](docs/INSTALL_MACOS.md) -- Intel Macs need
`-c constraints/macos-intel.txt`.

### Assisted setup with an AI coding assistant

[`SETUP_CON_AGENTE.md`](SETUP_CON_AGENTE.md) (in Spanish) is a self-contained setup guide for
Windows, Linux and macOS, written to be handed to a coding assistant such as Claude Code or Codex.
Attach the file, or point the assistant at it, and ask:

> Sigue las instrucciones de `SETUP_CON_AGENTE.md` para instalar y abrir SPADE en mi computador.
> *(Follow the instructions in `SETUP_CON_AGENTE.md` to install and open SPADE on my computer.)*

The guide makes the assistant:
1. Detect the operating system and CPU architecture.
2. Install into a local `.venv`.
3. Run the smoke test.
4. Open the GUI at http://localhost:8501.
5. Finish with a short report.

Before any system-level change (installing Python or Git, `sudo`, the Windows registry, the
PowerShell execution policy), the assistant asks for confirmation. It does not edit the repository
or retrain models.

A person can follow the same file by hand.

### Manual setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,viz]"
```

Run the dashboard (either command opens the same multipage app):

```bash
streamlit run src/driveflow/viz/app.py
streamlit run src/driveflow/viz/dashboard.py   # legacy command, still supported
```

The exact environment the test suite was verified in is pinned in `requirements-lock.txt`
(`uv pip install -r requirements-lock.txt && uv pip install -e . --no-deps`). Light and dark
themes are both native (app menu ⋮ → Settings → Theme); the app follows the OS preference.

Check the installation (imports, app start, modules, a short simulation of every plant, one
model inference -- same command on every OS):

```bash
python scripts/smoke_test.py            # add --launch to also start the real Streamlit server
```

Run the tests:

```bash
pytest                       # everything
pytest -m "not slow"         # what CI runs per commit
pytest -m simulation         # or: control, ai, compat, slow
```

Validation and audit reports: [docs/VALIDATION_SUMMARY.md](docs/VALIDATION_SUMMARY.md) (start
here) · [simulation stability](docs/SIMULATION_STABILITY_REPORT.md) ·
[AI technical audit](docs/AI_TECHNICAL_AUDIT.md) ·
[demo validation scenarios](docs/DEMO_VALIDATION_SCENARIOS.md).

`dev` (`pytest`) and `viz` (`streamlit`, `plotly`) are both needed to run the full test suite --
some tests drive the dashboard itself via `streamlit.testing.v1.AppTest`, so `viz` isn't only a
dashboard-runtime dependency. `pip install -e ".[dev]"` alone will fail to even collect those
test files. Core simulation/control code (`numpy`, `scipy`, `tensorflow`, ...) installs with the
package itself.

## Repository layout

```
src/driveflow/
  sim/            physics: DC/PMSM/induction motor models, VSC plant, bearing-fault injection,
                  synthetic vibration synthesis
  control/
    classical/    PI controller, PMSM FOC/MTPA
    mpc/          linear MPC (QP per step) for the DC motor -- interchangeable with PI, same plant
    dpc/          DPC network, model-based training loss, receding-horizon controller
  datagen/        Scenario dataclass + runner -- turns a config into a simulated dataset/trace
  models/         config-driven classifier/regressor builders + schemas, common windowing/splits
  ai/             model registry (configs/registry.yaml) + TFLite export (edge deployment)
    transfer/     transfer-learning stack: registry-backed loader, data merger, 3 fine-tune
                  strategies, promotion validator
  agents/         simulation-based anomaly agents: shared detect_anomaly + per-domain hypotheses,
                  DTW/Euclidean/Mahalanobis metrics, TTL simulation cache, feature-importance
                  explainer
  monitoring/     rule schema/YAML + the ESP32/Raspberry Pi 5/PC monitoring agents
  analysis/       display-grade spectrum (Hann-windowed amplitude spectrum, peak)
  validation/     stability/performance bench: plants with a selectable ODE solver, load-step
                  disturbance, step-response metrics, Euler stability bounds
  viz/            app.py (navigation) + dashboard.py (legacy entry); pages/ (thin page scripts),
                  views/ (page UI), components/ (header, KPI, charts, motor status, presets),
                  theme/ (tokens, scoped CSS, Plotly conventions), adapters/ (orchestration over
                  the library -- no numerics); ai_/transfer_learning_/live_monitoring_dashboard.py
                  are compatibility shims
experiments/      standalone scripts: train/promote/export models, fine-tune/evaluate the DPC
                  network, generate diagnosis datasets, calibrate the vibration module;
                  run_stability_battery.py / report_classifier_evaluation.py regenerate docs/audit/
scripts/          smoke_test.py -- cross-platform installation check
tests/            pytest suite; simulation/, control/, ai/, compat/ hold the categorised
                  validation tests (markers: simulation, control, ai, compat, slow)
constraints/      macos-intel.txt -- dependency pins for Intel Macs
configs/          the 4 shipped DPC checkpoints (see below), promoted classifier/regressor runs
                  (weights + config + metrics, some with a .tflite export), the model registry
                  manifest, and vibration module calibration
```

## The DPC checkpoints

`configs/*.weights.h5` ships 4 small (~21KB) trained checkpoints so the dashboard and
`experiments/evaluate_dpc*.py` work immediately, with no local training run required:

- `dpc_trained.weights.h5` -- base checkpoint, trained on `Data4train.mat`'s open-loop holdout split.
- `dpc_trained_v2_closed_loop.weights.h5` / `dpc_trained_v3_closed_loop.weights.h5` -- successive
  closed-loop fine-tunes at the single load R = 8.0064 ohm (see `experiments/finetune_dpc_closed_loop.py` /
  `continue_finetune_dpc.py`).
- `dpc_trained_v4_r_range.weights.h5` -- v3 fine-tuned on closed-loop rollouts over R = 3.5-1000 ohm
  and |v_ref| = 25-100 V (`experiments/finetune_dpc_r_range.py`). It tracks within ~3 V over the dashboard's load range
  (v3: up to 18 V), at the cost of the nominal point (2.5 V vs. v3's 1.18 V at 8 ohm) -- see
  docs/SIMULATION_STABILITY_REPORT.md S-05.
- **What every DPC `Scenario` and the live dashboard run (since 2026-10-20):**
  `ScheduledDpcController`, which uses v3 near 8 ohm (7.25-8.75 ohm, |v_ref| <= 62.5 V) and at
  small references (<= 10 V, 5-12 ohm) and v4 everywhere else: 1.18 V at 8 ohm, <= 3.1 V from 4 ohm
  to 1 kohm (docs/SIMULATION_STABILITY_REPORT.md §4.4).

To retrain from scratch you'll need `Data4train.mat` from the original
[DPC4PowerElectronics](https://github.com/aipoweraau/DPC4PowerElectronics) repository (not
redistributed here) -- see `experiments/train_dpc.py --help` and `DATA.md` for the full dependency
note (also covers the KAt-DataCenter/Paderborn bearing dataset the vibration module depends on).

## Classifier/regressor artifacts

`configs/{classifiers,regressors}/` also ships real, already-trained runs for the AI layer above
(one directory per `(config, timestamp)`, each with its weights + the config that produced it +
its metrics) -- the ones actually registered in `configs/registry.yaml` are what the dashboard's
IA tab and edge-deployment downloads use, no local training run required either. Currently
registered: a `dc_motor` classifier (PC/Raspberry Pi 5/ESP32 tiers) and a `vsc_dpc` regressor (same
3 tiers) -- the `vsc_dpc` classifier is deliberately not built yet (blocked on a separability
verdict, see the design doc), and `dc_motor` has no regressor yet. Retraining or adding a new
`(domain, tier, block)` combination needs a real dataset first (`experiments/generate_diagnosis_grid_dataset.py`,
`generate_diagnosis_dataset.py`
/ `generate_vsc_dpc_dataset.py`, not committed -- see `.gitignore`), then
`experiments/train_model.py --config ... --dataset ...` and `experiments/promote_run.py`.

## Design notes

- One ML framework throughout: Keras/TensorFlow (no PyTorch, no sklearn/XGBoost as a final model).
- Simulation is decoupled from `gymnasium.Env` -- `SCMLSystem` (`sim/scml_system.py`) is driven
  directly, not through the full Gym environment layer.
- DPC is NOT a drop-in alternative to the classical PI controller: it operates on a different
  physical domain entirely (a Voltage Source Converter -- no motor, no mechanical side, no rotor)
  with no state space, plant, or control objective in common with PI/MPC's motor domain. There is
  no "same conditions" under which to compare DPC vs. PI/MPC performance -- see `docs/patch5_alcance_macrofase_B.md`.
- No large datasets are committed to the repository (see `.gitignore`); the dashboard and
  `datagen/` generate everything on demand.

## Credits

This project consolidates and ports code from two MIT-licensed upstream projects:

- **[DPC4PowerElectronics](https://github.com/aipoweraau/DPC4PowerElectronics)** (Copyright (c)
  2024 AI-Power) -- the original MATLAB Direct Predictive Control implementation for a Voltage
  Source Converter. `src/driveflow/control/dpc/` and `src/driveflow/sim/vsc_system.py` are a
  Keras/TensorFlow port: network architecture, the identified discrete-time plant matrices, and
  the training loss were verified line-for-line against this source.
- **[gym-electric-motor](https://github.com/upb-lea/gym-electric-motor)** (Copyright (c) 2019
  Paderborn University -- LEA) -- `src/driveflow/sim/motors/` adapts its motor physics/parameter
  models, decoupled from its `gymnasium.Env` step-callback interface.

## License

MIT -- see [LICENSE](LICENSE).
