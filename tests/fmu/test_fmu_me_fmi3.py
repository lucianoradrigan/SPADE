"""P1 (round 2026-10-26): FmuAdapter's FMI 3.0 and Model Exchange paths, and PLECS FMUs in Plant Studio.

- FMI 3.0 Co-Simulation and Model Exchange: test doubles of fmpy's FMU3Slave / FMU3Model (no
  third-party binary is executed), with an analytic state, time events and a state event.
- Model Exchange on real FMUs (marker fmu): OpenModelica builds of SPADE's bucks
  (scripts/fmu/build_fmu_variants.mos -> data/fmu/variants/): the averaged buck against the native
  plant, and the switching buck -- which aborts in Co-Simulation (S-15) -- running and converging.
- Plant Studio with a "PLECS" FMU: an OpenModelica FMU whose modelDescription generationTool is
  relabelled "PLECS (test double)", to exercise PLECS detection and the external-FMU publication
  path. It is NOT a PLECS export; no PLECS-exported FMU has been run here.
"""

import io
import math
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from driveflow.integrations.fmu import adapter as fa

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = Path(os.environ.get("SPADE_FMU_DIR", ROOT / "data" / "fmu")) / "variants"


# --------------------------------------------------------------------------- FMI 3.0 test doubles
class _Base:
    def __init__(self):
        self.u, self.x, self.t, self.counter = 0.0, 0.0, 0.0, 0

    def instantiate(self, *a, **k): pass
    def enterInitializationMode(self, startTime=0.0, **k): self.t = startTime
    def exitInitializationMode(self): pass
    def terminate(self): pass
    def freeInstance(self): pass

    def setFloat64(self, vr, values):
        for r, v in zip(vr, values):
            if r == 0:
                self.u = v

    def getFloat64(self, vr, nValues=None):
        return [self.x for _ in vr]

    def getInt32(self, vr, nValues=None):
        return [self.counter for _ in vr]


class FakeFMU3Slave(_Base):
    """x' = u - x integrated exactly per step."""

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=True):
        self.x = self.u + (self.x - self.u) * math.exp(-communicationStepSize)
        self.t = currentCommunicationPoint + communicationStepSize
        return False, False, False, self.t


class FakeFMU3Model(_Base):
    """x' = u - x; time event every 0.25 s increments `counter`; event indicator z = x - 0.5 (state event)."""

    def __init__(self):
        super().__init__()
        self.next_time = 0.25
        self.crossings = []
        self.mode = "init"

    def updateDiscreteStates(self):
        if self.t >= self.next_time - 1e-12:
            self.counter += 1
            self.next_time += 0.25
        if self.mode == "event" and abs(self.x - 0.5) < 1e-6:
            self.crossings.append(self.t)
        return False, False, False, False, True, self.next_time

    def enterEventMode(self): self.mode = "event"
    def enterContinuousTimeMode(self): self.mode = "ct"
    def setTime(self, time): self.t = time
    def getContinuousStates(self, p, n): p[0] = self.x
    def setContinuousStates(self, p, n): self.x = p[0]
    def getContinuousStateDerivatives(self, p, n): p[0] = self.u - self.x
    def getEventIndicators(self, p, n): p[0] = self.x - 0.5
    def completedIntegratorStep(self, noSetFMUStatePriorToCurrentPoint=True): return False, False


VARS = (fa.Variable("u", 0, "input", "continuous", None, None, None, 0.0, "Float64"),
        fa.Variable("y", 1, "output", "continuous", None, None, None, None, "Float64"),
        fa.Variable("counter", 2, "output", "discrete", None, None, None, None, "Int32"))


def _fmi3_adapter(model, interface, **kw):
    a = fa.FmuAdapter(interface=interface, **kw)
    a.info = fa.FmuInfo("fake3.fmu", "0" * 64, "Fake3", "{g}", "3.0", "CoSimulation", ("x86_64-linux",), VARS,
                        fmu_types=("CoSimulation", "ModelExchange"), generation_tool="test double", n_states=1, n_event_indicators=1,
                        needs_completed_integrator_step=True)
    a.interface = fa.INTERFACES[interface]
    a._make_slave = lambda: model
    return a


def test_fmi3_co_simulation_path_steps_and_reads_float64():
    a = _fmi3_adapter(FakeFMU3Slave(), "cs")
    a.reset()
    for k in range(10):
        a.set_inputs({"u": 1.0})
        a.do_step(k * 0.1, 0.1)
    assert a.get_outputs()["y"] == pytest.approx(1 - math.exp(-1.0), rel=1e-12)
    assert a.stats()["interface"] == "CoSimulation" and a.stats()["fmi_version"] == "3.0"


def test_fmi3_model_exchange_rk4_time_events_and_located_state_event():
    m = FakeFMU3Model()
    a = _fmi3_adapter(m, "me", me_substeps=10, me_event_tol=1e-10)
    a.reset()
    for k in range(20):  # 2 s
        a.set_inputs({"u": 1.0})
        a.do_step(k * 0.1, 0.1)
    out = a.get_outputs()
    assert out["y"] == pytest.approx(1 - math.exp(-2.0), abs=1e-7)          # RK4, events in between
    assert out["counter"] == 8                                             # time events at 0.25 s steps, hit exactly
    assert len(m.crossings) == 1 and m.crossings[0] == pytest.approx(math.log(2), abs=1e-8)  # x = 0.5 at t = ln 2
    assert a.stats()["events"] >= 9


# --------------------------------------------------------------------------- real FMUs (OpenModelica builds)
def _need(name):
    pytest.importorskip("fmpy")
    p = VARIANTS / f"{name}.fmu"
    if not p.exists():
        pytest.skip(f"{p} not built (scripts/fmu/build_fmu_variants.mos)")
    return p


def _run(path, n, dt, sub, duty=0.5, out="v_out"):
    a = fa.FmuAdapter(interface="me", me_substeps=sub)
    a.load(path)
    a.reset()
    y = []
    for k in range(n):
        a.set_inputs({"duty": duty})
        a.do_step(k * dt, dt)
        y.append(a.get_outputs(check_range=False)[out])
    a.close()
    return np.array(y)


@pytest.mark.fmu
def test_model_exchange_averaged_buck_matches_the_native_plant():
    from driveflow.sim.buck_system import TAU, BuckSystem

    y = _run(_need("BuckAveraged_fmi2_me"), 500, TAU, 10)
    nat = BuckSystem()
    nat.reset()
    vn = np.array([nat.simulate(0.5).v_out for _ in range(500)])
    assert np.sqrt(np.mean((y - vn) ** 2)) / np.ptp(vn) < 1e-8   # measured 5.0e-11


@pytest.mark.fmu
def test_model_exchange_runs_the_switching_buck_and_converges():
    """BuckSwitching aborts in Co-Simulation with CVODE (S-15); through Model Exchange with event
    location it runs 10 ms and converges in the substep count."""
    from driveflow.sim.buck_system import TAU

    p = _need("BuckSwitching_fmi2_me")
    y4, y40 = _run(p, 500, TAU, 4), _run(p, 500, TAU, 40)
    assert np.all(np.isfinite(y4)) and len(y4) == 500
    assert np.max(np.abs(y4 - y40)) < 0.02          # V; measured 2.8e-3
    assert abs(y40[-1] - 23.75) < 0.1                # around the averaged steady state (native 23.753 V), ripple included


# --------------------------------------------------------------------------- Plant Studio, PLECS-labelled FMU
def _relabel_as_plecs(src: Path, dst: Path) -> Path:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "modelDescription.xml":
                import re

                data = re.sub(rb'generationTool="[^"]*"', b'generationTool="PLECS (test double: OpenModelica FMU relabelled)"', data, count=1)
            zout.writestr(item, data)
    return dst


def test_upload_rejects_non_fmus(tmp_path):
    from driveflow.integrations.fmu import gym_profile as gp

    with pytest.raises(ValueError, match="not a zip"):
        gp.save_upload(b"not a zip", "x.fmu")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="no modelDescription"):
        gp.save_upload(buf.getvalue(), "x.fmu")
    with pytest.raises(ValueError, match="file name"):
        gp.save_upload(buf.getvalue(), "../evil.fmu.exe")


@pytest.mark.fmu
def test_a_plecs_labelled_model_exchange_fmu_goes_from_upload_to_publication(tmp_path, monkeypatch):
    from driveflow.envs import registry
    from driveflow.integrations.fmu import gym_profile as gp

    monkeypatch.setenv("SPADE_FMU_DIR", str(tmp_path / "empty"))
    src = _relabel_as_plecs(_need("BuckAveraged_fmi2_me"), tmp_path / "PlecsBuck.fmu")
    gp.save_upload(src.read_bytes(), "PlecsBuck.fmu")
    cand = next(c for c in gp.candidates() if c["name"] == "PlecsBuck")
    assert cand["is_plecs"] and cand["b3"] is None and cand["publishable"] and cand["interfaces"] == ["ModelExchange"]
    p = gp.new_profile("plecs_buck", "PLECS buck (test double)", "PlecsBuck")
    m = gp.suggest_mapping(p)
    m["observations"] = [{"output": "i_L", "unit": "A", "scale": 15.0}, {"output": "v_out", "unit": "V", "scale": 48.0}]
    m["reference"], m["safety"] = {"output": "v_out", "value": 24.0}, {"output": "i_L", "rated": 15.0}
    p = gp.set_mapping(p, m)
    with pytest.raises(ValueError, match="does not offer CS"):
        gp.set_gym(p, {"tau_s": 2e-5, "duration_s": 0.004, "parameters": {}, "reward_terms": None, "interface": "cs"})
    p = gp.set_gym(p, {"tau_s": 2e-5, "duration_s": 0.004, "parameters": {}, "reward_terms": None, "interface": "me", "me_substeps": 5})
    assert gp.readiness(p)["Circuit"][0] == "Validated" and "no native comparison" in gp.readiness(p)["Circuit"][1]
    rep = gp.verify(p, log=lambda _m: None)
    assert rep["status"] == "passed", [c for c in rep["checks"] if c["status"] != "pass"]
    assert "ModelExchange" in next(c for c in rep["checks"] if c["id"] == 3)["detail"]
    p = gp.record_verification(p.profile_id, p.version, rep, "run_test")
    env_id = gp.publish(p)
    registry.register_all()
    env = registry.make(env_id)
    obs, _ = env.reset(seed=0)
    assert env.unwrapped.metadata["fmu"]["interface"] == "ModelExchange" and "PLECS" in env.unwrapped.metadata["fmu"]["generation_tool"]
    env.close()


def test_plant_studio_offers_the_plecs_fmu_engine_with_its_requirements():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "driveflow" / "viz" / "app.py"), default_timeout=300)
    at.run()
    at.switch_page("pages/plecs_plant_studio.py").run()
    at.button(key="plecs_goal_gym").click().run()
    at.radio(key="fmu_stage").set_value("2. Engine").run()
    at.radio(key="fmu_engine").set_value("PLECS FMU").run()
    assert not at.exception
    text = " ".join(str(c.value) for c in at.caption) + " ".join(str(m.value) for m in at.markdown)
    assert "Model Exchange" in text and "PLECS must be installed" in text and "No PLECS-exported FMU has been run yet" in text
    at.radio(key="fmu_stage").set_value("3. Circuit").run()
    assert not at.exception and at.get("file_uploader")
