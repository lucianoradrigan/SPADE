"""FmuAdapter: one FMI 2.0 Co-Simulation FMU driven step by step through fmpy (task B2, D6).

- ``load(path)``: reads modelDescription.xml -- inputs, outputs, parameters, causality, units,
  declared ranges, FMU type, GUID and the binary platforms in the archive. Unlike a .plecs file this
  introspection is real, so the GUI mapping can propose roles from it.
- ``reset(params, initial_state, seed)``: frees and re-instantiates the FMU from the extracted
  archive (more reproducible across FMUs than fmi2Reset), sets parameters and start values, runs
  initialisation, t = 0. ``seed`` is recorded; the FMUs here are deterministic.
- ``set_inputs(dict)``, ``do_step(t, dt)``, ``get_outputs()``, ``close()``.
- Typed errors (FmuError.code): load, platform, solver, nan, out_of_range.
- Instrumentation: wall time of every doStep (p50, p99) and the count of solver failures.

fmpy is the optional extra ``[fmu]`` (D3); importing this module without it raises FmuError("load").
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

CODES = ("load", "platform", "solver", "nan", "out_of_range")


class FmuError(RuntimeError):
    def __init__(self, code: str, message: str):
        assert code in CODES
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclasses.dataclass(frozen=True)
class Variable:
    name: str
    value_reference: int
    causality: str
    variability: str
    unit: str | None
    minimum: float | None
    maximum: float | None
    start: float | None


@dataclasses.dataclass(frozen=True)
class FmuInfo:
    path: str
    sha256: str
    model_name: str
    guid: str
    fmi_version: str
    fmu_type: str
    platforms: tuple
    variables: tuple

    def by_causality(self, causality: str) -> list[Variable]:
        return [v for v in self.variables if v.causality == causality]

    @property
    def inputs(self):
        return self.by_causality("input")

    @property
    def outputs(self):
        return self.by_causality("output")

    @property
    def parameters(self):
        return self.by_causality("parameter")


def _fmpy():
    try:
        import fmpy  # noqa: F401
        from fmpy import read_model_description
        from fmpy.fmi2 import FMU2Slave

        return fmpy, read_model_description, FMU2Slave
    except ImportError:
        raise FmuError("load", "fmpy is not installed: pip install \"driveflow[fmu]\"") from None


def _num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def inspect_fmu(path) -> FmuInfo:
    """Model description and archive platforms, without loading the binary."""
    fmpy, read_md, _ = _fmpy()
    p = Path(path)
    if not p.is_file():
        raise FmuError("load", f"no such FMU: {p}")
    try:
        md = read_md(str(p))
        with zipfile.ZipFile(p) as z:
            platforms = tuple(sorted({n.split("/")[1] for n in z.namelist() if n.startswith("binaries/") and n.count("/") >= 2 and n.split("/")[1]}))
    except Exception as exc:
        raise FmuError("load", f"cannot read {p.name}: {exc}") from None
    fmu_type = "CoSimulation" if md.coSimulation is not None else ("ModelExchange" if md.modelExchange is not None else "unknown")
    variables = tuple(Variable(v.name, int(v.valueReference), v.causality, v.variability, v.unit, _num(v.min), _num(v.max), _num(v.start))
                      for v in md.modelVariables if v.type == "Real")
    return FmuInfo(str(p), hashlib.sha256(p.read_bytes()).hexdigest(), md.modelName, md.guid, md.fmiVersion, fmu_type, platforms, variables)


def current_platform() -> str:
    fmpy, _, _ = _fmpy()
    return fmpy.platform


class FmuAdapter:
    def __init__(self, slave_factory=None):
        """slave_factory(info, unzip_dir) -> an object with fmpy's FMU2Slave methods (tests inject a fake)."""
        self._factory = slave_factory
        self.info: FmuInfo | None = None
        self._dir: str | None = None
        self._slave = None
        self._t = 0.0
        self._inputs: dict = {}
        self.step_wall_s: list[float] = []
        self.solver_failures = 0
        self.seed = None

    # ------------------------------------------------------------------ life cycle
    def load(self, path) -> FmuInfo:
        info = inspect_fmu(path)
        if info.fmi_version != "2.0" or info.fmu_type != "CoSimulation":
            raise FmuError("load", f"{Path(path).name} is FMI {info.fmi_version} {info.fmu_type}; an FMI 2.0 Co-Simulation FMU is required")
        if self._factory is None:
            plat = current_platform()
            if plat not in info.platforms:
                raise FmuError("platform", f"{Path(path).name} has binaries for {list(info.platforms) or 'no platform'}, not {plat}: Not supported on this machine")
        self.info = info
        self._dir = tempfile.mkdtemp(prefix="spade_fmu_")
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(self._dir)
        except Exception as exc:
            raise FmuError("load", f"cannot extract {Path(path).name}: {exc}") from None
        return info

    def _make_slave(self):
        if self._factory is not None:
            return self._factory(self.info, self._dir)
        _, read_md, FMU2Slave = _fmpy()
        md = read_md(self.info.path)
        return FMU2Slave(guid=md.guid, unzipDirectory=self._dir, modelIdentifier=md.coSimulation.modelIdentifier, instanceName="spade")

    def reset(self, params: dict | None = None, initial_state: dict | None = None, seed: int | None = None, start_time: float = 0.0) -> None:
        if self.info is None:
            raise FmuError("load", "load() an FMU first")
        self._free()
        self.seed = seed
        try:
            s = self._make_slave()
            s.instantiate()
            s.setupExperiment(startTime=start_time)
            values = {**(initial_state or {}), **(params or {})}
            if values:
                vr = self._vrs(values)
                s.setReal(vr, [float(values[n]) for n in values])
            s.enterInitializationMode()
            s.exitInitializationMode()
        except FmuError:
            raise
        except Exception as exc:
            raise FmuError("solver", f"initialisation failed: {exc}") from None
        self._slave, self._t, self._inputs = s, float(start_time), {}

    def _vrs(self, values: dict) -> list[int]:
        idx = {v.name: v.value_reference for v in self.info.variables}
        unknown = [n for n in values if n not in idx]
        if unknown:
            raise FmuError("load", f"unknown variable(s) {unknown} in {self.info.model_name}")
        return [idx[n] for n in values]

    def set_inputs(self, values: dict) -> None:
        names = {v.name for v in self.info.inputs}
        bad = [n for n in values if n not in names]
        if bad:
            raise FmuError("load", f"{bad} are not inputs of {self.info.model_name} (inputs: {sorted(names)})")
        for n, x in values.items():
            if not math.isfinite(float(x)):
                raise FmuError("nan", f"input {n} = {x!r}")
        self._slave.setReal(self._vrs(values), [float(x) for x in values.values()])
        self._inputs.update(values)

    def do_step(self, t: float, dt: float) -> None:
        if self._slave is None:
            raise FmuError("load", "reset() before do_step()")
        if abs(t - self._t) > 1e-12 * max(1.0, abs(t)):
            raise FmuError("solver", f"non-monotone step: FMU at t = {self._t!r}, asked to step from {t!r}")
        t0 = time.perf_counter()
        try:
            status = self._slave.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
        except Exception as exc:
            self.solver_failures += 1
            raise FmuError("solver", f"doStep at t = {t:g} s failed: {exc}") from None
        finally:
            self.step_wall_s.append(time.perf_counter() - t0)
        if status not in (None, 0):
            self.solver_failures += 1
            raise FmuError("solver", f"doStep at t = {t:g} s returned status {status}")
        self._t = t + dt

    def get_outputs(self, check_range: bool = True) -> dict:
        outs = self.info.outputs
        vals = self._slave.getReal([v.value_reference for v in outs])
        res = dict(zip([v.name for v in outs], (float(x) for x in vals)))
        for v in outs:
            x = res[v.name]
            if not math.isfinite(x):
                raise FmuError("nan", f"output {v.name} is {x!r} at t = {self._t:g} s")
            if check_range and ((v.minimum is not None and x < v.minimum) or (v.maximum is not None and x > v.maximum)):
                raise FmuError("out_of_range", f"output {v.name} = {x:g} outside its declared [{v.minimum}, {v.maximum}]")
        return res

    @property
    def time(self) -> float:
        return self._t

    def stats(self) -> dict:
        w = np.asarray(self.step_wall_s)
        return {"n_steps": int(len(w)), "p50_ms": float(np.percentile(w, 50) * 1e3) if len(w) else None,
                "p99_ms": float(np.percentile(w, 99) * 1e3) if len(w) else None, "solver_failures": self.solver_failures}

    def _free(self) -> None:
        if self._slave is not None:
            try:
                self._slave.terminate()
            except Exception:
                pass
            try:
                self._slave.freeInstance()
            except Exception:
                pass
            self._slave = None

    def close(self) -> None:
        self._free()
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
