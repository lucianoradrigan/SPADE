"""B1 (round PLECS/OM): the OpenModelica FMUs of modelica/SPADE (built by scripts/fmu/build_fmus.sh)
load with fmpy, expose the expected inputs and outputs with their causality, and simulate 10 ms.
Marker fmu: skipped, not failed, when fmpy or the built FMUs are missing (they are not committed, D8).
FMU folder: $SPADE_FMU_DIR, else <repo>/data/fmu."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fmu
fmpy = pytest.importorskip("fmpy")

FMU_DIR = Path(os.environ.get("SPADE_FMU_DIR", Path(__file__).resolve().parents[2] / "data" / "fmu"))
MODELS = {
    "BuckAveraged": ({"duty"}, {"v_out", "i_L"}, 2e-5),
    "BuckSwitching": ({"duty"}, {"v_out", "i_L"}, 2e-5),
    "DcMotorAveraged": ({"v_a"}, {"omega", "i_a", "torque"}, 1e-4),
    "DcMotorSwitching": ({"duty"}, {"omega", "i_a", "torque"}, 1e-4),
}


def _fmu(name):
    p = FMU_DIR / f"{name}.fmu"
    if not p.exists():
        pytest.skip(f"{p} not built (scripts/fmu/build_fmus.sh)")
    return p


def _sim(path, inp, value, step, stop=0.01):
    sig = np.array([(0.0, value), (stop, value)], dtype=[("time", np.float64), (inp, np.float64)])
    return fmpy.simulate_fmu(str(path), stop_time=stop, output_interval=step, input=sig, fmi_type="CoSimulation", logger=lambda *a: None)


@pytest.mark.parametrize("name", list(MODELS))
def test_fmu_interface(name):
    md = fmpy.read_model_description(str(_fmu(name)))
    ins, outs, _ = MODELS[name]
    assert md.fmiVersion == "2.0" and md.coSimulation is not None
    causal = {v.name: v.causality for v in md.modelVariables}
    assert {n for n, c in causal.items() if c == "input"} == ins
    assert {n for n, c in causal.items() if c == "output"} == outs


def test_manifest_hashes_match():
    man = FMU_DIR / "manifest.json"
    if not man.exists():
        pytest.skip("no data/fmu/manifest.json")
    doc = json.loads(man.read_text())
    for name, rec in doc["fmus"].items():
        assert hashlib.sha256((FMU_DIR / rec["file"]).read_bytes()).hexdigest() == rec["sha256"], name
    src = Path(__file__).resolve().parents[2] / "modelica" / "SPADE" / "package.mo"
    assert doc["source_mo_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest(), "FMUs built from a different model version"


@pytest.mark.parametrize("name, inp, value", [("BuckAveraged", "duty", 0.5), ("DcMotorAveraged", "v_a", 30.0)])
def test_averaged_fmus_simulate_10_ms(name, inp, value):
    r = _sim(_fmu(name), inp, value, MODELS[name][2])
    assert all(np.all(np.isfinite(r[o])) for o in MODELS[name][1])


# S-15 (open): with CVODE, which communication steps these FMUs survive changed between builds of the same
# model (docs/audit/plecs_om/b1_modelica_plants.md); the builds of 2026-10-24 fail at the PWM period. Strict, as
# every xfail (T4): a build where they pass turns this into a failure that must be looked at, not ignored.
@pytest.mark.xfail(strict=True, reason="S-15 (open): OpenModelica 1.27.1 switching FMUs fail with CVODE (CV_TOO_CLOSE) at the PWM period; outcome depends on the build")
@pytest.mark.parametrize("name", ["BuckSwitching", "DcMotorSwitching"])
def test_switching_fmus_simulate_10_ms_at_the_pwm_period(name):
    r = _sim(_fmu(name), "duty", 0.5, MODELS[name][2])
    assert all(np.all(np.isfinite(r[o])) for o in MODELS[name][1])
