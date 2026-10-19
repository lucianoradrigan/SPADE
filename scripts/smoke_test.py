"""SPADE installation smoke test -- works the same on Linux, Windows (PowerShell) and macOS.

    python scripts/smoke_test.py            # checks 1-6 below (~1 min, CPU only)
    python scripts/smoke_test.py --launch   # also starts the Streamlit server and probes /_stcore/health

Checks:
  1. dependencies import (and their versions are printed);
  2. the Streamlit application starts (AppTest renders the Overview page without exceptions;
     with --launch the real server answers its health endpoint);
  3. the platform modules load (simulation, control, datagen, AI registry, monitoring);
  4. a short simulation runs for each plant (DC motor + PI, DC motor + MPC, VSC + DPC, PMSM FOC)
     and produces finite values;
  5. one inference with a registered model (PC classifier on a fresh simulated window) returns a
     valid probability vector;
  6. no exception escaped any of the above.
Exit code 0 only if every check passed. Nothing is written to the repository.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-only on purpose: no GPU/CUDA required

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "driveflow" / "viz" / "app.py"
RESULTS: list = []


def check(name):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.perf_counter()
            try:
                detail = fn(*a, **kw) or ""
                RESULTS.append((name, True, detail, time.perf_counter() - t0))
                print(f"[PASS] {name} ({time.perf_counter() - t0:.1f} s) {detail}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- the point is to report, not to crash
                RESULTS.append((name, False, repr(exc), time.perf_counter() - t0))
                print(f"[FAIL] {name}: {exc!r}", flush=True)
                traceback.print_exc()
        return run
    return deco


@check("1. dependencies import")
def check_imports():
    import importlib

    versions = {}
    for mod in ("numpy", "scipy", "pandas", "pyarrow", "sklearn", "yaml", "gymnasium", "threadpoolctl", "tensorflow", "keras", "streamlit", "plotly"):
        m = importlib.import_module(mod)
        versions[mod] = getattr(m, "__version__", "?")
    return "; ".join(f"{k} {v}" for k, v in versions.items())


@check("3. platform modules load")
def check_modules():
    import driveflow.ai.registry  # noqa: F401
    import driveflow.control.classical  # noqa: F401
    import driveflow.control.mpc  # noqa: F401
    import driveflow.datagen  # noqa: F401
    import driveflow.monitoring.agents.agent_server  # noqa: F401
    import driveflow.sim  # noqa: F401
    from driveflow.monitoring.rules.schema import load_all_rulesets

    rules = load_all_rulesets(ROOT / "src" / "driveflow" / "monitoring" / "rules")
    return f"rule sets: {sorted(rules)}"


def _finite(values, what):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise AssertionError(f"{what}: empty or non-finite output")


@check("4. short simulations (DC+PI, DC+MPC, VSC+DPC, PMSM FOC)")
def check_simulation():
    from driveflow.control.classical.pmsm_foc import run_current_step
    from driveflow.datagen import Scenario, run_scenario

    pi = run_scenario(Scenario(scenario_id="smoke_pi", fault_type="outer_race", duration_s=0.06))
    _finite([r["rpm"] for r in pi] + [r["current_r"] for r in pi] + [r["acc_x"] for r in pi], "DC+PI")
    mpc = run_scenario(Scenario(scenario_id="smoke_mpc", controller_type="MPC", duration_s=0.01))
    _finite([r["rpm"] for r in mpc], "DC+MPC")
    vsc = run_scenario(Scenario(scenario_id="smoke_vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.01))
    _finite([r["vc_real"] for r in vsc], "VSC+DPC")
    traj = run_current_step(-50.0, 100.0, duration_s=0.005)
    _finite(traj, "PMSM FOC")
    return f"final speed {pi[-1]['rpm']:.1f} rpm, VSC |vc| {abs(complex(vsc[-1]['vc_real'], vsc[-1]['vc_imag'])):.1f} V"


@check("5. inference with a registered model")
def check_inference():
    import numpy as np
    import pandas as pd

    from driveflow.ai.calibration import decide
    from driveflow.ai.registry import load_promoted_model, resolve
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.models.common import normalize_window_like_training

    model, config, metrics = load_promoted_model("dc_motor", "pc", "classifier")
    df = pd.DataFrame.from_records(run_scenario(Scenario(scenario_id="smoke_inf", fault_type="outer_race", duration_s=0.06)))
    window = df[metrics["channels"]].to_numpy(np.float32)[-config.input_window :]
    # The run's own preprocessing (level_scaling) and calibration, as the GUI applies them.
    x, _, _ = normalize_window_like_training(window, metrics["channels"], metrics.get("level_scaling"))
    probs = model(x[np.newaxis], training=False).numpy()[0]
    if probs.shape != (len(metrics["classes"]),) or not np.all(np.isfinite(probs)) or abs(float(probs.sum()) - 1.0) > 1e-4:
        raise AssertionError(f"invalid probability vector {probs}")
    run = resolve("dc_motor", "pc", "classifier")
    label, cal = decide(probs, list(metrics["classes"]), metrics)
    kind = "calibrated" if "calibration" in metrics else "uncalibrated"
    return f"{run.parent.name}/{run.name} -> {label} ({cal.max():.2f}, {kind})"


@check("2. application starts (Overview page renders)")
def check_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    if at.exception:
        raise AssertionError(f"exceptions on the Overview page: {[e.message for e in at.exception]}")
    return "AppTest: no exceptions"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@check("2b. Streamlit server answers its health endpoint")
def check_server(timeout_s: float = 120.0):
    port = _free_port()
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    try:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"streamlit exited with code {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                    body = r.read().decode().strip()
                    if r.status == 200 and body == "ok":
                        return f"http://127.0.0.1:{port} -> ok"
            except OSError:
                time.sleep(1.0)
        raise AssertionError(f"no health answer within {timeout_s:.0f} s")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", action="store_true", help="also start the real Streamlit server and probe it")
    ap.add_argument("--skip-app", action="store_true", help="skip the AppTest render (check 2)")
    args = ap.parse_args(argv)
    print(f"SPADE smoke test -- Python {platform.python_version()} on {platform.system()} {platform.release()} ({platform.machine()})", flush=True)
    check_imports()
    if not args.skip_app:
        check_app()
    if args.launch:
        check_server()
    check_modules()
    check_simulation()
    check_inference()
    failed = [r for r in RESULTS if not r[1]]
    print(f"6. no exceptions: {'PASS' if not failed else 'FAIL'} -- {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
