# Installing SPADE on Windows 10 / 11 (PowerShell)

This guide takes a fresh Windows 10 or 11 machine (x86_64) to a running SPADE GUI and a passing smoke test. It uses only PowerShell, not WSL or Linux commands, and needs no GPU or CUDA.

> **Verification status (2026-10-18): NOT executed on Windows.** No Windows machine was available. The guide relies on these checks:
> - **Dependency resolution.** `uv pip compile pyproject.toml --extra dev --extra viz --python-platform x86_64-pc-windows-msvc` succeeded for Python 3.11 and 3.12. It resolved TensorFlow 2.21.0, Keras 3.15.1 and Streamlit 1.66, and every package has a Windows wheel. Output: [`docs/audit/install/`](audit/install/).
> - **Path portability.** No absolute or POSIX-only paths in the package. Covered by `tests/compat/test_smoke.py::test_repository_paths_are_platform_neutral`.
> - **CI.** The `windows-latest` job in `.github/workflows/tests.yml` runs the smoke test. It takes effect once the workflow is pushed.
>
> Treat Windows as **expected to work, not verified** until that CI job has passed.

## 1. Requirements

| Item | Supported |
|---|---|
| Windows | 10 (21H2 or later) or 11, 64-bit |
| Python | **3.11 or 3.12**, 64-bit. Python 3.13+ is rejected by `pyproject.toml`. |
| GPU | not required. TensorFlow on native Windows is CPU-only, which is all SPADE needs. |
| Disk | ~3 GB |

## 2. Install Python and Git

From PowerShell (uses `winget`, which ships with Windows 10 21H2+ and 11):

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
```

Close and reopen PowerShell, then check:

```powershell
py -3.12 --version     # Python 3.12.x
git --version
```

If you install Python from python.org instead, tick **"Add python.exe to PATH"**. Make sure you take the **64-bit** installer.

## 3. Allow long paths (once, as Administrator)

TensorFlow's wheel contains very deep paths. With the 260-character limit, `pip install` can fail with `OSError: [Errno 2] No such file or directory` (or `[WinError 206]`). Enable long paths in an **Administrator** PowerShell, then reboot:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Also keep the clone in a short path, e.g. `C:\dev\SPADE`.

## 4. Get the code

```powershell
mkdir C:\dev -Force
cd C:\dev
git clone https://github.com/lucianoradrigan/SPADE.git
cd SPADE
```

## 5. Create the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

If `Activate.ps1` fails with *"running scripts is disabled on this system"*, allow local scripts for your user only, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Alternatively, skip activation and call the environment's Python directly: `.\.venv\Scripts\python.exe ...`.

<details><summary>Alternative: <code>uv</code></summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,viz]"
```

`requirements-lock.txt` was frozen on Linux. It contains no Linux-only packages, but it was **not** tried on Windows. Prefer the unpinned `-e ".[dev,viz]"` install there.
</details>

## 6. Verify the installation

```powershell
python scripts\smoke_test.py
python scripts\smoke_test.py --launch   # also starts the real Streamlit server
```

Every check must print `[PASS]`, and the last line must read `6. no exceptions: PASS`. To check the exit code: `$LASTEXITCODE` must be `0`. The checks are described in [INSTALL_UBUNTU.md §5](INSTALL_UBUNTU.md#5-verify-the-installation).

```powershell
pytest -m compat            # smoke checks under pytest
pytest -m "not slow"        # broader suite
```

## 7. Run SPADE

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src\driveflow\viz\app.py
```

The browser opens at http://localhost:8501. The first time, Windows Defender Firewall may ask about `python.exe`. Allowing **private networks** is enough.

For a basic manual test:
1. Open **DC Motor** and press **Generate**.
2. Open **AI Diagnostics** and check that a prediction is shown.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `... cannot be loaded because running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `.\.venv\Scripts\python.exe`. |
| `OSError: [Errno 2] No such file or directory: ...tensorflow\include\...` / `WinError 206` | 260-character path limit | §3: enable `LongPathsEnabled`, reboot, and clone into a short path. |
| `ERROR: Could not find a version that satisfies the requirement tensorflow` | 32-bit Python, or Python 3.13+ | Install 64-bit Python 3.12 (`py -3.12 -m venv .venv`). |
| `requires a different Python: 3.13.x not in '<3.13,>=3.11'` | Python too new | Use `py -3.12`. |
| `ImportError: DLL load failed while importing _pywrap_tensorflow_internal` | Missing Microsoft Visual C++ Redistributable | `winget install Microsoft.VCRedist.2015+.x64`, then reopen PowerShell. |
| `'python' is not recognized` / Microsoft Store opens | App-execution alias, or no Python on PATH | Use the `py` launcher (`py -3.12`), or disable the "App execution aliases" for python in Settings. |
| `streamlit` not recognized | Environment not active | `.\.venv\Scripts\Activate.ps1`, or `python -m streamlit run src\driveflow\viz\app.py`. |
| oneDNN / CUDA messages in the log | Informational TensorFlow log | Ignore. To hide them: `$env:TF_CPP_MIN_LOG_LEVEL = "2"`. |
