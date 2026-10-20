# Installing SPADE on macOS (Apple Silicon and Intel)

This guide takes a Mac to a running SPADE GUI and a passing smoke test. It needs no GPU, CUDA or Metal plugin: everything runs on the CPU.

> **Verification status (2026-10-18): NOT executed on a Mac.** The guide relies on these checks:
> - **Dependency resolution for both architectures.**
>   - `uv pip compile ... --python-platform aarch64-apple-darwin` (Apple Silicon) resolved TensorFlow 2.21.0, Keras 3.15.1 and NumPy 2.x for Python 3.11 and 3.12. This is the same set as the verified Linux environment.
>   - `--python-platform x86_64-apple-darwin` (Intel) showed that an unconstrained install **breaks**; see §1.
> - **Intel dependency set emulated on Linux.** The constrained Intel set (Python 3.12, TensorFlow 2.16.2, NumPy 1.26.4, pyarrow 24, Streamlit 1.60.0) was installed on Linux x86_64. The smoke test passed 6/6, including the real server health check. The non-slow suite gave 631 passed and 3 failed; the 3 failures are explained in §1/§7 and in [`docs/audit/install/macos_intel_emulated_on_linux.txt`](audit/install/macos_intel_emulated_on_linux.txt). This checks the **package versions**, not macOS itself.
> - **CI.** The `macos-latest` (Apple Silicon) job in `.github/workflows/tests.yml` runs the smoke test. It takes effect once the workflow is pushed.
>
> No Intel-Mac CI runner is configured. Treat both architectures as **expected to work, not verified**.

## 1. Architecture differences (read first)

| | Apple Silicon (M1–M4, `arm64`) | Intel (`x86_64`) |
|---|---|---|
| TensorFlow | 2.21 (current; same as the verified Linux environment) | **2.16.2 at most.** Google publishes no newer x86_64 macOS wheel. |
| NumPy | 2.x | **1.26.x.** TF 2.16 is built against NumPy 1.x. |
| Python | 3.11 or 3.12 | **3.12 only.** With 3.11 the resolver falls back to TF 2.14 + Keras 2.14, which cannot run SPADE's Keras-3 code. |
| Extra step | none | install with `-c constraints/macos-intel.txt` (§4). Without it, pip pairs NumPy 1.26 with pyarrow 26, which fails at import with *"pyarrow requires NumPy 2.0 or newer"*. |
| Streamlit | current (1.62 or later) | **1.61 at most.** TF 2.16 needs protobuf < 5, and Streamlit ≥ 1.62 needs protobuf ≥ 5.26, so the resolver installs 1.60. An earlier version of the Transfer Learning page failed one `AppTest` flow under it (a `KeyError` after jumping to *Promote*); the page now keeps each step's widget state, and the full non-slow suite passes with this pinned set emulated on Linux (`docs/audit/install/macos_intel_emulated_on_linux_p2.txt`). A real Intel Mac was not used. |
| `tensorflow-metal` | **do not install.** SPADE does not need it, and it was not tested. | n/a |

Check which Mac you have: `uname -m` prints `arm64` (Apple Silicon) or `x86_64` (Intel).

Under Rosetta, a Terminal or Python on Apple Silicon reports `x86_64`. Use a native arm64 Python (Homebrew in `/opt/homebrew`, or `uv`). Otherwise you get the Intel limitations above.

## 2. Install the tools

```bash
xcode-select --install                       # Git + compilers (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   # if Homebrew is missing
brew install python@3.12 git
```

Alternatively, use `uv` instead of Homebrew's Python: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## 3. Get the code

```bash
git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE
```

## 4. Create the environment

**Apple Silicon:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

**Intel Mac (Python 3.12 + constraints):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,viz]" -c constraints/macos-intel.txt
```

With `uv`, the equivalent is `uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev,viz]"`. On Intel, add `-c constraints/macos-intel.txt`.

`requirements-lock.txt` is a Linux freeze (TF 2.21, NumPy 2.4). It may work on Apple Silicon, but it **cannot** work on Intel Macs.

## 5. Verify the installation

```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --launch
```

Every check must print `[PASS]`, the last line must read `6. no exceptions: PASS`, and `echo $?` must print `0`. The checks are described in [INSTALL_UBUNTU.md §5](INSTALL_UBUNTU.md#5-verify-the-installation).

```bash
pytest -m compat
pytest -m "not slow"
```

## 6. Run SPADE

```bash
source .venv/bin/activate
streamlit run src/driveflow/viz/app.py
```

The app opens at http://localhost:8501. macOS may ask whether Python can accept incoming connections; for local use you can **Deny**.

For a basic manual test:
1. Open **DC Motor** and press **Generate**.
2. Open **AI Diagnostics** and check that a prediction is shown.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` | Intel Mac installed without constraints | `pip install -e ".[dev,viz]" -c constraints/macos-intel.txt` |
| `ERROR: No matching distribution found for tensorflow==2.21...` on an Intel Mac | No x86_64 macOS wheel after 2.16.2 | Use the Intel constraints. If `uname -m` shows `x86_64` on Apple Silicon, switch to a native arm64 Python. |
| `AttributeError`/`ImportError` mentioning `keras` 2.x | Python 3.11 on an Intel Mac resolved TF 2.14 + Keras 2 | Recreate the venv with Python 3.12 plus the constraints. |
| `zsh: illegal hardware instruction python` | x86_64 TensorFlow under Rosetta, or a mixed-arch environment | Recreate `.venv` with a native arm64 Python (`/opt/homebrew/bin/python3.12`). |
| `xcrun: error: invalid active developer path` | Command Line Tools missing | `xcode-select --install` |
| `requires a different Python: 3.13.x` | Homebrew's default `python3` is too new | Use `python3.12` explicitly. |
| `streamlit: command not found` | Environment not active | `source .venv/bin/activate`, or `python -m streamlit run ...`. |
| Very slow first import of TensorFlow | Gatekeeper scanning the new binaries | Expected once per install. |
