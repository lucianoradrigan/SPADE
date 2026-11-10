# Fresh clone of release/v0.2.0 (round C, 2026-11-09, before P2)

Raw values: [`fresh_clone.json`](fresh_clone.json). Screenshots: [`fresh_clone/`](fresh_clone/), one per page, resized to 720 px wide. **Simulated.**

## Setup

- **Clone:** `git clone --branch release/v0.2.0 --single-branch https://github.com/lucianoradrigan/SPADE.git /mnt/datos/spade/fresh_clone`. The final run is on commit `d43865e`.
- **Environment:** a new `.venv` with `uv venv --python 3.12` and `uv pip install -e ".[dev,viz]"`.
  - Playwright 1.65.0 is installed **in this test venv only**, not in `pyproject.toml`, and drives the system Google Chrome.
- **Conditions:** no `data/`, no `workspace/` at start, `SPADE_WORKSPACE` unset, and **`OLLAMA_HOST=127.0.0.1:9`** (a closed port; checked before the run).
- **Commands:** `python scripts/smoke_test.py --launch`. Then the app runs on port 8790, and the Playwright script (`/mnt/datos/spade/fresh_pages.py`, not part of the repository) does the following for each page:
  - opens it;
  - clicks its simulation or its first primary action (none on the Assistant page);
  - waits until Streamlit has finished;
  - collects exceptions and error alerts;
  - takes a screenshot.

## Results

**smoke `--launch`:**
- [PASS] 1. dependencies import (2.0 s) numpy 2.5.4; scipy 1.18.1; pandas 3.1.1; pyarrow 26.0.1; sklearn 1.9.1; yaml 6.0.3; gymnasium 1.4.0; threadpoolctl 3.8.0; tensorflow 2.21.0; keras 3.15.1; streamlit 1.67.0; plotly 7.1.0
- [PASS] 2. application starts (Overview page renders) (0.6 s) AppTest: no exceptions
- [PASS] 2c. Assistant page renders, also with the local model runtime down (0.2 s) AppTest: no exceptions; 'Assistant unavailable' shown
- [PASS] 2b. Streamlit server answers its health endpoint (1.3 s) http://127.0.0.1:60039 -> ok
- [PASS] 3. platform modules load (0.0 s) rule sets: ['vsc_dpc']
- [PASS] 4. short simulations (DC+PI, DC+MPC, VSC+DPC, PMSM FOC) (0.5 s) final speed 470.3 rpm, VSC |vc| 50.2 V
- [PASS] 5. inference with a registered model (0.8 s) pc_server_5class_w2048/2026-11-04_run01 -> uncertain (0.28, calibrated)
- 6. no exceptions: PASS -- 7/7 checks passed

**Pages: 18 of 18 open without an exception.**

| Page | URL | Heading shown | Action (simulation) | Time (s) | No exception | Screenshot |
|---|---|---|---|---|---|---|
| Overview | `/` | Electric drives, simulation & AI | none on this page (no primary action) | — | yes | [01_overview.png](fresh_clone/01_overview.png) |
| Experiment Runs | `/experiment_runs` | Experiment Runs | none on this page (no primary action) | — | yes | [02_experiment_runs.png](fresh_clone/02_experiment_runs.png) |
| DC Motor | `/dc_motor` | Motor health & signal diagnostics | Run simulation | 5.3 | yes | [03_dc_motor.png](fresh_clone/03_dc_motor.png) |
| PMSM / FOC | `/pmsm` | PMSM field-oriented control: MTPA vs. naive | Generate PMSM cloud | 4.9 | yes | [04_pmsm.png](fresh_clone/04_pmsm.png) |
| VSC / DPC | `/vsc_dpc` | Direct Power Control of a Voltage Source Converter | first primary button: '\n\nGenerate DPC scenario' | 3.2 | yes | [05_vsc_dpc.png](fresh_clone/05_vsc_dpc.png) |
| DC-DC Converters | `/dcdc_converters` | DC-DC Converters | first primary button: '\n\nGenerate Buck scenario' | 2.7 | yes | [06_dcdc_converters.png](fresh_clone/06_dcdc_converters.png) |
| PLECS Plant Studio | `/plecs_plant_studio` | PLECS Plant Studio | none on this page (no primary action) | — | yes | [07_plecs_plant_studio.png](fresh_clone/07_plecs_plant_studio.png) |
| Tuning Studio | `/tuning_studio` | Tuning Studio | first primary button: '\n\nStart tuning' | 2.7 | yes | [08_tuning_studio.png](fresh_clone/08_tuning_studio.png) |
| RL Lab | `/rl_lab` | RL Lab | none on this page (no primary action) | — | yes | [09_rl_lab.png](fresh_clone/09_rl_lab.png) |
| AI Diagnostics | `/ai_diagnostics` | Evaluate and train diagnosis models | first primary button: '\n\nGenerate' | 4.2 | yes | [10_ai_diagnostics.png](fresh_clone/10_ai_diagnostics.png) |
| Edge Deployment | `/edge_deployment` | Run your models on edge hardware | first primary button: '\n\nDownload ESP32 model (.tflite)' | 2.7 | yes | [11_edge_deployment.png](fresh_clone/11_edge_deployment.png) |
| Transfer Learning | `/transfer_learning` | Adapt a model safely | first primary button: '\n\nPrepare ' | 40.4 | yes | [12_transfer_learning.png](fresh_clone/12_transfer_learning.png) |
| Model Registry | `/model_registry` | Model registry & artifacts | none on this page (no primary action) | — | yes | [13_model_registry.png](fresh_clone/13_model_registry.png) |
| Assistant | `/assistant` | Assistant | none on this page (no primary action) | — | yes | [14_assistant.png](fresh_clone/14_assistant.png) |
| Monitoring | `/monitoring` | Agent consensus & alert analysis | first primary button: '\n\nRun monitoring' | 13.3 | yes | [15_monitoring.png](fresh_clone/15_monitoring.png) |
| Alert History | `/alert_history` | Alert history | none on this page (no primary action) | — | yes | [16_alert_history.png](fresh_clone/16_alert_history.png) |
| Documentation | `/documentation` | How to read SPADE | none on this page (no primary action) | — | yes | [17_documentation.png](fresh_clone/17_documentation.png) |
| Algorithm Guide | `/algorithm_guide` | Which algorithm family fits my problem? | none on this page (no primary action) | — | yes | [18_algorithm_guide.png](fresh_clone/18_algorithm_guide.png) |

**Pages without a simulation:**
- Overview, Experiment Runs, Model Registry, Alert History, Documentation and Algorithm Guide have no simulation.
- PLECS Plant Studio needs PLECS.
- RL Lab trains through a queued job.
- The jobs queued by Tuning Studio and Transfer Learning stay queued, because no worker runs in this test.

**Assistant with the model runtime unreachable:** the page shows
> Assistant unavailable: the local model runtime does not answer at http://127.0.0.1:9 (<urlopen error [Errno 111] Connection refused>). Is Ollama installed and running? (systemctl status ollama). Start the local runtime with: systemctl start ollama (or ollama serve).

A message was sent (`yes`). The page keeps the same notice and raises no exception.

**Browser console:** 404s on `<page>/_stcore/health` and `<page>/_stcore/host-config`. These are Streamlit's own frontend probes when a page is opened by its URL; they are harmless and do not come from SPADE. There is also one Chrome "compute-pressure" policy notice.

## Findings of this check (both fixed or mitigated on the release branch, then re-run)

1. **AI-N32 (Low, fixed):** the assistant ignored `OLLAMA_HOST`. With the variable pointing to a closed port it would still have used the real Ollama on port 11434, so the test could not simulate a missing runtime. It now honours `OLLAMA_HOST`, and the loopback rule is unchanged (`tests/assistant/test_assistant_ollama_host.py`).
2. **AI-N33 (Medium, mitigated):** with AI Diagnostics' **default** sample (healthy, PI, 300 rad/s, 0.6 s), the classified window lies in the start-up transient, and the promoted PC classifier answered **'cage' at 100 % calibrated confidence as a known condition**. The page now warns on start-up windows (see screenshot 10), and the DC Motor AI card says "start-up window: unreliable". **Update (same night, decision N-D1):** AI-N33 is now **corrected**. The page uses the training window rule, and its default duration is 0.82 s; see [`ai_n33.md`](ai_n33.md). The model is unchanged.
3. **No finding:** a first run of the page script failed on the Assistant page. The script waited for "network idle", which never comes on that page because its job and avatar fragments refresh. This was the harness, not SPADE; it now waits for Streamlit's run indicator.

## Not executed

- A worker processing the queued jobs (Tuning Studio, Transfer Learning).
- RL Lab training.
- PLECS.
- A session with a running Ollama (this check is about its absence).
