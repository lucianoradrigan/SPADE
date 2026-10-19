# Installing SPADE on Ubuntu Linux

This guide takes a fresh Ubuntu machine to a running SPADE GUI and a passing smoke test. You do not need to know the repository layout, and you need neither a GPU nor CUDA: everything runs on the CPU.

> **Verification status (2026-10-18).**
> - **Executed:** the commands in Sections 2–5 on **Ubuntu 26.04.1 LTS (x86_64)**, using Python 3.11.15 and 3.12.13 installed with `uv`. Results are in [`docs/audit/install/`](audit/install/).
> - **Not executed:** **Ubuntu 22.04** and **24.04**. No machine or container was available. For these releases the guide relies on two checks:
>   - dependency resolution (`uv pip compile --python-platform x86_64-unknown-linux-gnu`, Python 3.11 and 3.12), which succeeded;
>   - the CI job on `ubuntu-latest`, which is currently Ubuntu 24.04 (see `.github/workflows/tests.yml`). It runs only after this workflow is pushed to GitHub.
>
> Treat 22.04/24.04 as **expected to work, not verified**.

## 1. Supported versions

| Item | Supported | Notes |
|---|---|---|
| Python | **3.11 or 3.12** | `pyproject.toml` declares `>=3.11,<3.13`. |
| Ubuntu 24.04 | system `python3` is 3.12 → usable as is | Install `python3.12-venv`. |
| Ubuntu 22.04 | system `python3` is 3.10 → **not supported** | Install 3.11/3.12 with `uv` (recommended) or the deadsnakes PPA. Do not replace the system Python. |
| Ubuntu 26.04 | system `python3` is 3.14 → **not supported** | Install 3.11/3.12 with `uv`. |
| CPU | x86_64 | aarch64 was not checked. |
| GPU | not required | TensorFlow runs on the CPU. CUDA messages in the log are harmless. |
| Disk | ~3 GB | Most of it is TensorFlow. |

The exact environment the test suite was verified in is pinned in `requirements-lock.txt`: Python 3.11.15, TensorFlow 2.21.0, Streamlit 1.62.0.

## 2. System packages

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

## 3. Get the code

```bash
git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE
```

## 4. Create the environment

### Option A — `uv` (recommended; same commands on 22.04, 24.04 and 26.04)

`uv` downloads a supported Python on its own, so the system Python version does not matter.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"          # or open a new terminal
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,viz]"
```

To install the exact package set the tests were verified with, use this instead of the last line:

```bash
uv pip install -r requirements-lock.txt
uv pip install -e . --no-deps
```

### Option B — standard `venv` + `pip`

On **Ubuntu 24.04**, the system Python 3.12 works directly:

```bash
sudo apt install -y python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

On **Ubuntu 22.04**, add Python 3.11 from the deadsnakes PPA first. It installs alongside the system Python and does not replace it:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

The two extras are both needed:
- `dev` installs `pytest`;
- `viz` installs `streamlit` and `plotly`. The GUI needs them, and so do part of the test suite and the smoke test.

## 5. Verify the installation

```bash
python scripts/smoke_test.py            # imports, app render, modules, simulation, inference (~1 min)
python scripts/smoke_test.py --launch   # same, plus starting the real Streamlit server
```

All six checks must report `PASS`. The last line must read `6. no exceptions: PASS -- N/N checks passed`, and the exit code must be 0. What each check covers:

| # | Check | How it is verified |
|---|---|---|
| 1 | Dependencies import | Each package is imported and its version printed. |
| 2 | Application starts | The Overview page renders under Streamlit `AppTest` with no exception. With `--launch`, the real server must also answer `GET /_stcore/health` with `ok`. |
| 3 | Modules load | Simulation, control, datagen, AI registry, monitoring and rule YAMLs are imported or loaded. |
| 4 | Short simulation | DC motor with PI, DC motor with MPC, VSC with DPC, and PMSM FOC each run briefly. All outputs must be finite. |
| 5 | Inference | The registered PC classifier predicts on a freshly simulated window. The result must be a finite probability vector that sums to 1. |
| 6 | No exceptions | Summary of checks 1–5. |

Optional, longer test runs:

```bash
pytest -m "compat"                 # the smoke checks under pytest (~1 min)
pytest -m "not slow"               # everything except the >20 s tests (~8 min on 12 cores)
pytest                             # full suite
```

## 6. Run SPADE

```bash
source .venv/bin/activate
streamlit run src/driveflow/viz/app.py
```

Open the URL Streamlit prints, normally http://localhost:8501.

For a basic manual test:
1. Open **DC Motor** and press **Generate**. Current, speed and vibration plots should appear.
2. Open **AI Diagnostics**. The classifier panel should show a prediction labelled *AI SUGGESTION — REQUIRES VERIFICATION*.

On a headless server, add `--server.headless true --server.address 0.0.0.0` and open port 8501 in the firewall.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: Package 'driveflow' requires a different Python: 3.10.12 not in '<3.13,>=3.11'` | Ubuntu 22.04's system Python | Use Option A (`uv venv --python 3.11`) or deadsnakes. |
| Same error with `3.13`/`3.14` | Newer Ubuntu's system Python | `uv venv --python 3.12 .venv`. |
| `The virtual environment was not created successfully because ensurepip is not available` | `python3.X-venv` is missing | `sudo apt install python3.12-venv` (or `python3.11-venv`). |
| `ModuleNotFoundError: No module named 'streamlit'` (or a pytest collection error in `test_ui_contract.py`) | Installed without the `viz` extra | `pip install -e ".[dev,viz]"`. |
| `Could not find cuda drivers on your machine, GPU will not be used` / `Unable to register cuDNN factory` | Informational TensorFlow log | Ignore. SPADE runs on the CPU. To hide it: `export TF_CPP_MIN_LOG_LEVEL=2`. |
| `Illegal instruction (core dumped)` when importing TensorFlow | CPU without AVX (very old or emulated CPU) | The official TensorFlow wheels need AVX. Use a machine or VM with AVX enabled. |
| `streamlit: command not found` | Environment not activated | `source .venv/bin/activate`, or run `python -m streamlit run src/driveflow/viz/app.py`. |
| Port 8501 already in use | Another Streamlit process | `streamlit run ... --server.port 8502`. |
| Very slow MPC pages or tests | BLAS thread oversubscription | Already limited inside the MPC (`threadpoolctl`). For tests you can also set `export OMP_NUM_THREADS=1`. |
