"""B2 (round PLECS/OM): the generic FmuAdapter. Unit tests use a fake slave (no fmpy binary needed beyond
reading a model description); the integration test (marker fmu) drives the real BuckAveraged FMU."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from driveflow.integrations.fmu import adapter as fa

FMU_DIR = Path(os.environ.get("SPADE_FMU_DIR", Path(__file__).resolve().parents[2] / "data" / "fmu"))


class FakeSlave:
    """x' = u - x (one state), y = x; fails on request; NaN on request."""

    def __init__(self, fail_at=None, nan_at=None):
        self.x, self.u, self.t, self.fail_at, self.nan_at = 0.0, 0.0, 0.0, fail_at, nan_at
        self.freed = False

    def instantiate(self): pass
    def setupExperiment(self, startTime=0.0): self.t = startTime
    def enterInitializationMode(self): pass
    def exitInitializationMode(self): pass
    def terminate(self): pass
    def freeInstance(self): self.freed = True

    def setReal(self, vr, values):
        for r, v in zip(vr, values):
            if r == 0:
                self.u = v
            elif r == 1:
                self.x = v

    def doStep(self, currentCommunicationPoint, communicationStepSize):
        if self.fail_at is not None and currentCommunicationPoint >= self.fail_at:
            return 3
        self.x += communicationStepSize * (self.u - self.x)
        self.t = currentCommunicationPoint + communicationStepSize
        return 0

    def getReal(self, vr):
        return [float("nan") if (self.nan_at is not None and self.t >= self.nan_at) else self.x for _ in vr]


INFO = fa.FmuInfo("fake.fmu", "0" * 64, "Fake", "{g}", "2.0", "CoSimulation", ("linux64",),
                  (fa.Variable("u", 0, "input", "continuous", None, None, None, 0.0),
                   fa.Variable("y", 1, "output", "continuous", "V", -1.0, 10.0, None),
                   fa.Variable("k", 2, "parameter", "fixed", None, None, None, 1.0)))


def _adapter(**kw):
    a = fa.FmuAdapter(slave_factory=lambda info, d: FakeSlave(**kw))
    a.info, a._dir = INFO, None
    return a


def test_steps_inputs_outputs_and_stats():
    a = _adapter()
    a.reset(seed=3)
    for k in range(10):
        a.set_inputs({"u": 1.0})
        a.do_step(k * 0.1, 0.1)
    y = a.get_outputs()["y"]
    assert y == pytest.approx(1 - 0.9 ** 10) and a.time == pytest.approx(1.0) and a.seed == 3
    st = a.stats()
    assert st["n_steps"] == 10 and st["p50_ms"] is not None and st["solver_failures"] == 0


def test_reset_is_reproducible_and_reinstantiates():
    a = _adapter()
    outs = []
    for _ in range(2):
        a.reset(seed=0)
        for k in range(5):
            a.set_inputs({"u": 2.0})
            a.do_step(k * 0.1, 0.1)
        outs.append(a.get_outputs()["y"])
    assert outs[0] == outs[1]


@pytest.mark.parametrize("kw, step, code", [({"fail_at": 0.2}, 3, "solver"), ({"nan_at": 0.2}, 3, "nan")])
def test_typed_errors(kw, step, code):
    a = _adapter(**kw)
    a.reset()
    with pytest.raises(fa.FmuError) as e:
        for k in range(step):
            a.set_inputs({"u": 1.0})
            a.do_step(k * 0.1, 0.1)
            a.get_outputs()
    assert e.value.code == code
    if code == "solver":
        assert a.stats()["solver_failures"] == 1


def test_out_of_range_unknown_input_and_non_monotone_time():
    a = _adapter()
    a.reset()
    a.set_inputs({"u": 1000.0})
    a.do_step(0.0, 0.5)
    with pytest.raises(fa.FmuError, match="out_of_range"):
        a.get_outputs()
    with pytest.raises(fa.FmuError, match="not inputs"):
        a.set_inputs({"y": 1.0})
    with pytest.raises(fa.FmuError, match="non-monotone"):
        a.do_step(0.0, 0.1)
    with pytest.raises(fa.FmuError, match="nan"):
        a.set_inputs({"u": math.nan})


def test_platform_mismatch_is_reported(tmp_path, monkeypatch):
    pytest.importorskip("fmpy")
    p = FMU_DIR / "BuckAveraged.fmu"
    if not p.exists():
        pytest.skip("BuckAveraged.fmu not built")
    monkeypatch.setattr(fa, "current_platform", lambda: "win64")
    with pytest.raises(fa.FmuError) as e:
        fa.FmuAdapter().load(p)
    assert e.value.code == "platform" and "Not supported" in str(e.value)


@pytest.mark.fmu
def test_real_buck_averaged_fmu_against_the_native_plant():
    pytest.importorskip("fmpy")
    p = FMU_DIR / "BuckAveraged.fmu"
    if not p.exists():
        pytest.skip("BuckAveraged.fmu not built")
    from driveflow.sim.buck_system import TAU, BuckSystem

    a = fa.FmuAdapter()
    info = a.load(p)
    assert [v.name for v in info.inputs] == ["duty"] and {v.name for v in info.outputs} == {"v_out", "i_L"}
    runs = []
    for _ in range(2):
        a.reset(seed=0)
        v = []
        for k in range(500):
            a.set_inputs({"duty": 0.5})
            a.do_step(k * TAU, TAU)
            v.append(a.get_outputs()["v_out"])
        runs.append(np.array(v))
    a.close()
    nat = BuckSystem()
    nat.reset()
    vn = np.array([nat.simulate(0.5).v_out for _ in range(500)])
    assert np.array_equal(runs[0], runs[1])
    assert np.max(np.abs(runs[0] - vn)) < 0.01 * np.ptp(vn)
